"""Masked sequence losses and optimization for the image-history world model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from trans_wm.config import ACTION_HISTORY_LEN, OBS_HISTORY_LEN
from trans_wm.history import history_windows, previous_history_windows
from trans_wm.model import WorldModel
from utils.replay_buffer import EpisodeBatch, RolloutReplayBuffer


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 100.0
    observation_weight: float = 1.0
    reward_weight: float = 1.0
    value_weight: float = 1.0
    vae_reconstruction_weight: float = 1.0
    vae_kl_weight: float = 1e-4

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided.")
        if any(
            weight < 0.0
            for weight in (
                self.observation_weight,
                self.reward_weight,
                self.value_weight,
                self.vae_reconstruction_weight,
                self.vae_kl_weight,
            )
        ):
            raise ValueError("Loss weights must be non-negative.")


@dataclass(frozen=True)
class TensorEpisodeBatch:
    observations: torch.Tensor  # [B, T + 1, C, H, W]
    actions: torch.Tensor  # [B, T, action_dim]
    rewards: torch.Tensor  # [B, T, 1]
    terminated: torch.Tensor  # [B, T]
    transition_valid: torch.Tensor  # [B, T]
    state_valid: torch.Tensor  # [B, T + 1]


@dataclass(frozen=True)
class TensorTransitionBatch:
    current_observations: torch.Tensor  # [B, 10, C, H, W]
    next_observations: torch.Tensor  # [B, 10, C, H, W]
    current_obs_valid: torch.Tensor  # [B, 10]
    next_obs_valid: torch.Tensor  # [B, 10]
    action_history: torch.Tensor  # [B, 9, action_dim]
    action_valid: torch.Tensor  # [B, 9]
    action: torch.Tensor  # [B, action_dim]
    reward: torch.Tensor  # [B, 1]
    terminated: torch.Tensor  # [B, 1]


@dataclass(frozen=True)
class WorldModelLosses:
    total: torch.Tensor
    observation: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    vae_reconstruction: torch.Tensor
    vae_kl: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": self.total.detach().item(),
            "observation": self.observation.detach().item(),
            "reward": self.reward.detach().item(),
            "value": self.value.detach().item(),
            "vae_reconstruction": self.vae_reconstruction.detach().item(),
            "vae_kl": self.vae_kl.detach().item(),
        }


def tensor_episode_batch(batch: EpisodeBatch, model: WorldModel) -> TensorEpisodeBatch:
    """Loads images as BCHW tensors and constructs state/transition masks."""
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    source = batch.images if batch.images is not None else batch.obs
    observations = _image_tensor(source, model.config.observation_shape, device, dtype)
    actions = torch.as_tensor(batch.action, device=device, dtype=dtype).flatten(start_dim=2)
    rewards = torch.as_tensor(batch.reward, device=device, dtype=dtype).unsqueeze(-1)
    terminated = torch.as_tensor(batch.terminated, device=device, dtype=torch.bool)
    lengths = torch.as_tensor(batch.lengths, device=device)
    if actions.shape[2] != model.config.action_dim:
        raise ValueError("Replay actions do not match model action_shape.")
    time = actions.shape[1]
    transition_valid = torch.arange(time, device=device)[None] < lengths[:, None]
    state_valid = torch.arange(time + 1, device=device)[None] <= lengths[:, None]
    return TensorEpisodeBatch(
        observations, actions, rewards, terminated, transition_valid, state_valid
    )


def sample_transition_batch(
    batch: EpisodeBatch,
    model: WorldModel,
    batch_size: int,
    rng: np.random.Generator,
) -> TensorTransitionBatch:
    """Samples transitions with their exact image and action histories."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    source = batch.images if batch.images is not None else batch.obs
    if source.ndim != 5:
        raise ValueError("Trans-WM training requires image observations.")
    lengths = np.asarray(batch.lengths, dtype=np.int64)
    cumulative = np.cumsum(lengths)
    flat_indices = rng.integers(0, int(cumulative[-1]), size=batch_size)
    episode_indices = np.searchsorted(cumulative, flat_indices, side="right")
    starts = np.concatenate((np.zeros(1, dtype=np.int64), cumulative[:-1]))
    time_indices = flat_indices - starts[episode_indices]

    current_observations = np.zeros((batch_size, OBS_HISTORY_LEN, *source.shape[2:]), dtype=source.dtype)
    next_observations = np.zeros_like(current_observations)
    current_obs_valid = np.zeros((batch_size, OBS_HISTORY_LEN), dtype=bool)
    next_obs_valid = np.zeros_like(current_obs_valid)
    action_shape = batch.action.shape[2:]
    action_history = np.zeros(
        (batch_size, ACTION_HISTORY_LEN, *action_shape), dtype=batch.action.dtype
    )
    action_valid = np.zeros((batch_size, ACTION_HISTORY_LEN), dtype=bool)
    actions = np.empty((batch_size, *action_shape), dtype=batch.action.dtype)
    rewards = np.empty((batch_size, 1), dtype=batch.reward.dtype)
    terminated = np.empty((batch_size, 1), dtype=bool)

    for sample, (episode, timestep) in enumerate(zip(episode_indices, time_indices, strict=True)):
        current_start = max(0, int(timestep) - OBS_HISTORY_LEN + 1)
        current = source[episode, current_start : timestep + 1]
        current_observations[sample, -len(current) :] = current
        current_obs_valid[sample, -len(current) :] = True

        next_start = max(0, int(timestep) - OBS_HISTORY_LEN + 2)
        following = source[episode, next_start : timestep + 2]
        next_observations[sample, -len(following) :] = following
        next_obs_valid[sample, -len(following) :] = True

        action_start = max(0, int(timestep) - ACTION_HISTORY_LEN)
        previous_actions = batch.action[episode, action_start:timestep]
        if len(previous_actions):
            action_history[sample, -len(previous_actions) :] = previous_actions
            action_valid[sample, -len(previous_actions) :] = True
        actions[sample] = batch.action[episode, timestep]
        rewards[sample, 0] = batch.reward[episode, timestep]
        terminated[sample, 0] = batch.terminated[episode, timestep]

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    return TensorTransitionBatch(
        _image_tensor(current_observations, model.config.observation_shape, device, dtype),
        _image_tensor(next_observations, model.config.observation_shape, device, dtype),
        torch.as_tensor(current_obs_valid, device=device),
        torch.as_tensor(next_obs_valid, device=device),
        torch.as_tensor(action_history, device=device, dtype=dtype).flatten(start_dim=2),
        torch.as_tensor(action_valid, device=device),
        torch.as_tensor(actions, device=device, dtype=dtype).flatten(start_dim=1),
        torch.as_tensor(rewards, device=device, dtype=dtype),
        torch.as_tensor(terminated, device=device),
    )


def encode_sequence(
    model: WorldModel, batch: TensorEpisodeBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns one z per state plus the exact image windows and masks used."""
    obs_windows, obs_masks = history_windows(
        batch.observations, batch.state_valid, OBS_HISTORY_LEN
    )
    batch_size, states = batch.observations.shape[:2]
    latents = model.encode_online(obs_windows.flatten(0, 1), obs_masks.flatten(0, 1))
    return (
        latents.reshape(batch_size, states, model.config.latent_dim),
        obs_windows,
        obs_masks,
    )


def vae_kl_loss(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Standard-normal posterior KL, reduced over latent dimensions per sample."""
    if mean.shape != log_variance.shape or mean.ndim != 2:
        raise ValueError("mean and log_variance must have shape [batch, latent_dim].")
    return -0.5 * (
        1.0 + log_variance - mean.square() - log_variance.exp()
    ).mean(dim=-1)


def bellman_target(
    reward: torch.Tensor,
    next_value: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Returns r(t) + gamma * V_target(z(t+1)), without terminal bootstrap."""
    if reward.shape != next_value.shape or reward.shape != terminated.shape:
        raise ValueError("reward, next_value, and terminated shapes must match.")
    if terminated.dtype != torch.bool:
        raise ValueError("terminated must be boolean.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    return reward + gamma * (~terminated) * next_value


def world_model_loss(
    model: WorldModel,
    batch: TensorEpisodeBatch,
    config: TrainingConfig,
) -> WorldModelLosses:
    """Computes task losses plus same-window VAE reconstruction and KL."""
    obs_windows, obs_masks = history_windows(
        batch.observations, batch.state_valid, OBS_HISTORY_LEN
    )
    batch_size, states = batch.observations.shape[:2]
    flat_windows = obs_windows.flatten(0, 1)
    flat_masks = obs_masks.flatten(0, 1)
    posterior = model.encoder.posterior(flat_windows, flat_masks)
    latents = posterior.mean.reshape(batch_size, states, model.config.latent_dim)
    vae_reconstruction = model.heads.observation_head(posterior.rsample())
    vae_reconstruction_per_state = (
        (vae_reconstruction - flat_windows).flatten(1).square().mean(dim=1)
    )
    vae_kl_per_state = vae_kl_loss(posterior.mean, posterior.log_variance)
    action_windows, action_masks = previous_history_windows(
        batch.actions, batch.transition_valid, ACTION_HISTORY_LEN
    )
    current_z = latents[:, :-1].flatten(0, 1)
    reward_target = batch.rewards.flatten(0, 1)

    current_heads = model.predict_heads_online(current_z)
    predicted_next_z = model.predict_next_online(
        current_z,
        action_windows.flatten(0, 1),
        batch.actions.flatten(0, 1),
        action_masks.flatten(0, 1),
    )
    predicted_next_heads = model.predict_heads_online(predicted_next_z)

    observation_target = obs_windows[:, 1:].flatten(0, 1)
    observation_per_item = (
        (predicted_next_heads.observation - observation_target).flatten(1).square().mean(dim=1)
    )
    reward_per_item = (predicted_next_heads.reward - reward_target).square().squeeze(-1)

    with torch.no_grad():
        ema_next_z = model.ema_encoder(
            obs_windows[:, 1:].flatten(0, 1), obs_masks[:, 1:].flatten(0, 1)
        )
        target_next_value = model.ema_heads.value_head(ema_next_z)
        value_target = bellman_target(
            reward_target,
            target_next_value,
            batch.terminated.flatten(0, 1).unsqueeze(-1),
            model.config.gamma,
        )
    current_value_error = (current_heads.value - value_target).square()
    predicted_next_value_error = (
        predicted_next_heads.value - target_next_value
    ).square()
    value_per_item = (0.5 * (current_value_error + predicted_next_value_error)).squeeze(-1)

    valid = batch.transition_valid.flatten().to(dtype=latents.dtype)
    observation_loss = _masked_mean(observation_per_item, valid)
    reward_loss = _masked_mean(reward_per_item, valid)
    value_loss = _masked_mean(value_per_item, valid)
    state_valid = batch.state_valid.flatten().to(dtype=latents.dtype)
    vae_reconstruction_loss = _masked_mean(vae_reconstruction_per_state, state_valid)
    vae_kl = _masked_mean(vae_kl_per_state, state_valid)
    total = (
        config.observation_weight * observation_loss
        + config.reward_weight * reward_loss
        + config.value_weight * value_loss
        + config.vae_reconstruction_weight * vae_reconstruction_loss
        + config.vae_kl_weight * vae_kl
    )
    return WorldModelLosses(
        total, observation_loss, reward_loss, value_loss, vae_reconstruction_loss, vae_kl
    )


def transition_world_model_loss(
    model: WorldModel,
    batch: TensorTransitionBatch,
    config: TrainingConfig,
) -> WorldModelLosses:
    """Computes task and same-window VAE objectives on sampled transitions."""
    current_posterior = model.encoder.posterior(
        batch.current_observations, batch.current_obs_valid
    )
    next_posterior = model.encoder.posterior(
        batch.next_observations, batch.next_obs_valid
    )
    current_z = current_posterior.mean
    current_heads = model.predict_heads_online(current_z)
    predicted_next_z = model.predict_next_online(
        current_z, batch.action_history, batch.action, batch.action_valid
    )
    predicted_next_heads = model.predict_heads_online(predicted_next_z)

    observation_loss = (
        (predicted_next_heads.observation - batch.next_observations)
        .flatten(1)
        .square()
        .mean()
    )
    reward_loss = (predicted_next_heads.reward - batch.reward).square().mean()
    current_vae_reconstruction = model.heads.observation_head(current_posterior.rsample())
    next_vae_reconstruction = model.heads.observation_head(next_posterior.rsample())
    vae_reconstruction_loss = 0.5 * (
        (current_vae_reconstruction - batch.current_observations).square().mean()
        + (next_vae_reconstruction - batch.next_observations).square().mean()
    )
    vae_kl = 0.5 * (
        vae_kl_loss(current_posterior.mean, current_posterior.log_variance).mean()
        + vae_kl_loss(next_posterior.mean, next_posterior.log_variance).mean()
    )
    with torch.no_grad():
        ema_next_z = model.ema_encoder(batch.next_observations, batch.next_obs_valid)
        target_next_value = model.ema_heads.value_head(ema_next_z)
        value_target = bellman_target(
            batch.reward,
            target_next_value,
            batch.terminated,
            model.config.gamma,
        )
    value_loss = 0.5 * (
        (current_heads.value - value_target).square().mean()
        + (predicted_next_heads.value - target_next_value).square().mean()
    )
    total = (
        config.observation_weight * observation_loss
        + config.reward_weight * reward_loss
        + config.value_weight * value_loss
        + config.vae_reconstruction_weight * vae_reconstruction_loss
        + config.vae_kl_weight * vae_kl
    )
    return WorldModelLosses(
        total, observation_loss, reward_loss, value_loss, vae_reconstruction_loss, vae_kl
    )


class WorldModelTrainer:
    """Optimizes the world model without owning or assuming a policy."""

    def __init__(self, model: WorldModel, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_batch(self, batch: EpisodeBatch) -> dict[str, float]:
        self.model.train()
        tensor_batch = tensor_episode_batch(batch, self.model)
        self.optimizer.zero_grad(set_to_none=True)
        losses = world_model_loss(self.model, tensor_batch, self.config)
        losses.total.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()
        self.model.update_target()
        return losses.detached()

    def train_transitions(
        self,
        batch: EpisodeBatch,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Trains from sampled transitions without materializing full sequence windows."""
        self.model.train()
        sampled = sample_transition_batch(batch, self.model, batch_size, rng)
        self.optimizer.zero_grad(set_to_none=True)
        losses = transition_world_model_loss(self.model, sampled, self.config)
        losses.total.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()
        self.model.update_target()
        return losses.detached()

    def fit(
        self,
        replay_buffer: RolloutReplayBuffer,
        rollouts: int,
        *,
        batch_size: int = 8,
        epochs_per_rollout: int = 10,
        sample_rollouts: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[dict[str, float]]:
        if rollouts <= 0 or epochs_per_rollout <= 0:
            raise ValueError("rollouts and epochs_per_rollout must be positive.")
        rng = rng or np.random.default_rng()
        history = []
        for _ in range(rollouts):
            batch = replay_buffer.sample(sample_rollouts)
            metrics = {}
            for _ in range(epochs_per_rollout):
                metrics = self.train_transitions(batch, batch_size, rng)
            history.append(metrics)
        return history


def _image_tensor(
    array: np.ndarray,
    observation_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if array.ndim != 5:
        raise ValueError("Image observations must have shape [batch, time, C, H, W] or HWC.")
    channels, height, width = observation_shape
    if array.shape[2:] == (channels, height, width):
        tensor = torch.as_tensor(array, device=device, dtype=dtype)
    elif array.shape[2:] == (height, width, channels):
        tensor = torch.as_tensor(array, device=device, dtype=dtype).permute(0, 1, 4, 2, 3)
    else:
        raise ValueError("Replay images do not match model observation_shape.")
    if np.issubdtype(array.dtype, np.integer):
        tensor = tensor / float(np.iinfo(array.dtype).max)
    return tensor


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape != valid.shape:
        raise ValueError("Loss values and validity mask must have identical shapes.")
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("A training batch must contain at least one valid transition.")
    return (values * valid).sum() / count
