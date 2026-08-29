"""JEPA world model over image and action histories."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import prod

import torch
from torch import nn

from tdmpc_like.config import ACTION_HISTORY_LEN, OBS_HISTORY_LEN, WorldModelConfig
from tdmpc_like.history import append_history


class ImageHistoryEncoder(nn.Module):
    """Encodes a masked image-history window into an observation tensor."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.observation_shape = config.observation_shape
        layers: list[nn.Module] = []
        in_channels = OBS_HISTORY_LEN * config.channels
        for out_channels in config.cnn_channels:
            layers.extend(
                (
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                )
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, OBS_HISTORY_LEN * config.channels, config.height, config.width)
            encoded = self.cnn(dummy)
        self.to_observation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(encoded.shape[1:]), config.observation_dim),
        )

    def forward(self, obs_history: torch.Tensor, obs_valid_mask: torch.Tensor) -> torch.Tensor:
        expected = (OBS_HISTORY_LEN, *self.observation_shape)
        if obs_history.shape[1:] != expected:
            raise ValueError(f"obs_history must have shape [batch, {expected}].")
        _validate_mask(obs_valid_mask, obs_history.shape[:2], obs_history.device, "obs_valid_mask")
        masked = obs_history * obs_valid_mask[:, :, None, None, None]
        stacked = masked.flatten(start_dim=1, end_dim=2)
        return self.to_observation(self.cnn(stacked))


class LatentEncoder(nn.Module):
    """Fuses the observation tensor and preceding-action history into latent state."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.observation_dim = config.observation_dim
        self.action_history_dim = config.action_history_dim
        self.latent_dim = config.latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(config.observation_dim + config.action_history_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.latent_dim),
        )

    def forward(
        self, observation: torch.Tensor, action_history_tensor: torch.Tensor
    ) -> torch.Tensor:
        batch = observation.shape[0]
        if observation.shape != (batch, self.observation_dim):
            raise ValueError(
                f"observation must have shape [batch, {self.observation_dim}]."
            )
        if action_history_tensor.shape != (batch, self.action_history_dim):
            raise ValueError(
                "action_history_tensor must have shape "
                f"[batch, {self.action_history_dim}]."
            )
        if action_history_tensor.device != observation.device:
            raise ValueError("observation and action history must be on the same device.")
        return self.mlp(torch.cat((observation, action_history_tensor), dim=-1))


class LatentDynamics(nn.Module):
    """Predicts the next latent state from [latent, action] tokens."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.action_dim = config.action_dim
        self.position = nn.Parameter(torch.empty(1, 2, config.model_dim))
        self.z_projection = nn.Linear(config.latent_dim, config.model_dim)
        self.action_projection = nn.Linear(config.action_dim, config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.latent_dim),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        if latent.shape != (batch, self.latent_dim):
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}].")
        if action.shape != (batch, self.action_dim):
            raise ValueError(f"action must have shape [batch, {self.action_dim}].")
        if action.device != latent.device:
            raise ValueError("latent and action must be on the same device.")
        tokens = torch.cat(
            (
                self.z_projection(latent)[:, None],
                self.action_projection(action)[:, None],
            ),
            dim=1,
        )
        encoded = self.transformer(tokens + self.position)
        return self.output(encoded[:, 0])


@dataclass(frozen=True)
class HeadOutput:
    reward: torch.Tensor
    value: torch.Tensor
    q1: torch.Tensor | None = None
    q2: torch.Tensor | None = None


class WorldHeads(nn.Module):
    """Predicts transition reward and twin action-conditioned Q values."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.action_dim = config.action_dim
        self.reward_head = _scalar_head(config, config.latent_dim + config.action_dim)
        self.q1_head = _scalar_head(config, config.latent_dim + config.action_dim)
        self.q2_head = _scalar_head(config, config.latent_dim + config.action_dim)

    @property
    def value_head(self) -> nn.Sequential:
        """Compatibility alias for the first Q head."""
        return self.q1_head

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> HeadOutput:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}].")
        action = action.flatten(start_dim=1)
        if action.shape != (latent.shape[0], self.action_dim):
            raise ValueError(f"action must have shape [batch, {self.action_dim}].")
        q1, q2 = self.q_values(latent, action)
        return HeadOutput(self.reward_head(torch.cat((latent, action), dim=-1)), torch.minimum(q1, q2), q1, q2)

    def reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action.flatten(start_dim=1)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}].")
        if action.shape != (latent.shape[0], self.action_dim):
            raise ValueError(f"action must have shape [batch, {self.action_dim}].")
        return self.reward_head(torch.cat((latent, action), dim=-1))

    def value(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.q_values(latent, action)
        return torch.minimum(q1, q2)

    def q_values(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must have shape [batch, {self.latent_dim}].")
        action = action.flatten(start_dim=1)
        if action.shape != (latent.shape[0], self.action_dim):
            raise ValueError(f"action must have shape [batch, {self.action_dim}].")
        inputs = torch.cat((latent, action), dim=-1)
        return self.q1_head(inputs), self.q2_head(inputs)


@dataclass(frozen=True)
class ActionEvaluation:
    next_z: torch.Tensor
    heads: HeadOutput
    score: torch.Tensor


@dataclass(frozen=True)
class RolloutOutput:
    latents: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    scores: torch.Tensor
    final_z: torch.Tensor
    final_action_history: torch.Tensor
    final_action_valid_mask: torch.Tensor


class WorldModel(nn.Module):
    """Observation encoder, 128D latent encoder, Transformer dynamics, and task heads."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ImageHistoryEncoder(config)
        self.latent_encoder = LatentEncoder(config)
        self.dynamics = LatentDynamics(config)
        self.heads = WorldHeads(config)
        self.ema_encoder = deepcopy(self.encoder).requires_grad_(False)
        self.ema_latent_encoder = deepcopy(self.latent_encoder).requires_grad_(False)
        self.ema_dynamics = deepcopy(self.dynamics).requires_grad_(False)
        self.ema_heads = deepcopy(self.heads).requires_grad_(False)
        self._set_ema_eval()

    def train(self, mode: bool = True) -> WorldModel:
        super().train(mode)
        self._set_ema_eval()
        return self

    def _set_ema_eval(self) -> None:
        self.ema_encoder.eval()
        self.ema_latent_encoder.eval()
        self.ema_dynamics.eval()
        self.ema_heads.eval()

    def action_history_tensor(
        self,
        action_history: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = action_history.shape[0]
        if action_history.shape != (batch, ACTION_HISTORY_LEN, self.config.action_dim):
            raise ValueError(
                f"action_history must have shape [batch, {ACTION_HISTORY_LEN}, "
                f"{self.config.action_dim}]."
            )
        if action_valid_mask is None:
            action_valid_mask = torch.ones(
                action_history.shape[:2], dtype=torch.bool, device=action_history.device
            )
        _validate_mask(
            action_valid_mask,
            action_history.shape[:2],
            action_history.device,
            "action_valid_mask",
        )
        return (action_history * action_valid_mask[:, :, None]).flatten(start_dim=1)

    def encode_observation_online(
        self, obs_history: torch.Tensor, obs_valid_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(obs_history, obs_valid_mask)

    def encode_online(
        self,
        obs_history: torch.Tensor,
        obs_valid_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observation = self.encoder(obs_history, obs_valid_mask)
        ah = self._prepare_action_history(action_history, action_valid_mask)
        return self.latent_encoder(observation, ah)

    @torch.no_grad()
    def encode(
        self,
        obs_history: torch.Tensor,
        obs_valid_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_ema(
            obs_history, obs_valid_mask, action_history, action_valid_mask
        )

    @torch.no_grad()
    def encode_ema(
        self,
        obs_history: torch.Tensor,
        obs_valid_mask: torch.Tensor,
        action_history: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observation = self.ema_encoder(obs_history, obs_valid_mask)
        ah = self._prepare_action_history(action_history, action_valid_mask)
        return self.ema_latent_encoder(observation, ah)

    def _prepare_action_history(
        self,
        action_history: torch.Tensor,
        action_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_history.ndim == 2:
            if action_valid_mask is not None:
                raise ValueError("action_valid_mask is only used with unflattened action history.")
            return action_history
        return self.action_history_tensor(action_history, action_valid_mask)

    def predict_next_online(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(z, action.flatten(start_dim=1))

    @torch.no_grad()
    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predict_next_ema(z, action)

    @torch.no_grad()
    def predict_next_ema(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.ema_dynamics(z, action.flatten(start_dim=1))

    def predict_heads_online(self, z: torch.Tensor, action: torch.Tensor) -> HeadOutput:
        return self.heads(z, action)

    @torch.no_grad()
    def predict_heads(self, z: torch.Tensor, action: torch.Tensor) -> HeadOutput:
        return self.ema_heads(z, action)

    def evaluate_action_online(self, z: torch.Tensor, action: torch.Tensor) -> ActionEvaluation:
        next_z = self.predict_next_online(z, action)
        heads = self.heads(z, action)
        return ActionEvaluation(next_z, heads, heads.reward)

    @torch.no_grad()
    def evaluate_action(self, z: torch.Tensor, action: torch.Tensor) -> ActionEvaluation:
        return self.evaluate_action_ema(z, action)

    @torch.no_grad()
    def evaluate_action_ema(self, z: torch.Tensor, action: torch.Tensor) -> ActionEvaluation:
        next_z = self.predict_next_ema(z, action)
        heads = self.ema_heads(z, action)
        return ActionEvaluation(next_z, heads, heads.reward)

    @torch.no_grad()
    def update_target(self, ema: float | None = None) -> None:
        ema = self.config.target_ema if ema is None else ema
        if not 0.0 <= ema < 1.0:
            raise ValueError("ema must be in [0, 1).")
        for target_module, online_module in (
            (self.ema_encoder, self.encoder),
            (self.ema_latent_encoder, self.latent_encoder),
            (self.ema_dynamics, self.dynamics),
            (self.ema_heads, self.heads),
        ):
            for target, online in zip(
                target_module.parameters(), online_module.parameters(), strict=True
            ):
                target.lerp_(online, 1.0 - ema)
            for target, online in zip(
                target_module.buffers(), online_module.buffers(), strict=True
            ):
                target.copy_(online)

    def rollout_online(
        self,
        z: torch.Tensor,
        action_history: torch.Tensor,
        actions: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> RolloutOutput:
        return self._rollout(z, action_history, actions, action_valid_mask, online=True)

    @torch.no_grad()
    def rollout(
        self,
        z: torch.Tensor,
        action_history: torch.Tensor,
        actions: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> RolloutOutput:
        return self.rollout_ema(z, action_history, actions, action_valid_mask)

    @torch.no_grad()
    def rollout_ema(
        self,
        z: torch.Tensor,
        action_history: torch.Tensor,
        actions: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
    ) -> RolloutOutput:
        return self._rollout(z, action_history, actions, action_valid_mask, online=False)

    def _rollout(
        self,
        z: torch.Tensor,
        action_history: torch.Tensor,
        actions: torch.Tensor,
        action_valid_mask: torch.Tensor | None,
        *,
        online: bool,
    ) -> RolloutOutput:
        if actions.ndim < 3 or actions.shape[0] != z.shape[0]:
            raise ValueError("actions must have shape [batch, horizon, *action_shape].")
        actions = actions.flatten(start_dim=2)
        if actions.shape[2] != self.config.action_dim or actions.shape[1] == 0:
            raise ValueError("actions must have a positive horizon and match action_shape.")
        if action_valid_mask is None:
            action_valid_mask = torch.ones(
                action_history.shape[:2], dtype=torch.bool, device=action_history.device
            )
        latents, rewards, values, scores = [], [], [], []
        evaluate = self.evaluate_action_online if online else self.evaluate_action_ema
        for action in actions.unbind(dim=1):
            evaluation = evaluate(z, action)
            z = evaluation.next_z
            action_history, action_valid_mask = append_history(
                action_history, action_valid_mask, action
            )
            latents.append(z)
            rewards.append(evaluation.heads.reward)
            values.append(evaluation.heads.value)
            scores.append(evaluation.heads.reward)
        return RolloutOutput(
            torch.stack(latents, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(values, dim=1),
            torch.stack(scores, dim=1),
            z,
            action_history,
            action_valid_mask,
        )


def _scalar_head(config: WorldModelConfig, input_dim: int | None = None) -> nn.Sequential:
    input_dim = config.latent_dim if input_dim is None else input_dim
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, config.model_dim),
        nn.GELU(),
        nn.Linear(config.model_dim, 1),
    )


def _validate_mask(
    mask: torch.Tensor, shape: torch.Size | tuple[int, ...], device: torch.device, name: str
) -> None:
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean with shape {tuple(shape)}.")
    if mask.device != device:
        raise ValueError(f"{name} must be on the same device as its input.")
