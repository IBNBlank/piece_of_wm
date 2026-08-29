"""A compact Dreamer-v1 learner with multi-frame observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dreamer_like.model import RSSM, RSSMState, Actor, ValueModel, ImageHistoryEncoder, ImageHistoryDecoder
from dreamer_like.config import WorldModelConfig
from dreamer_like.training import lambda_return, rssm_kl_loss


@dataclass(frozen=True)
class DreamerLoss:
    world_model: torch.Tensor
    actor: torch.Tensor
    value: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.world_model + self.actor + self.value


class DreamerV1(nn.Module):
    """End-to-end Dreamer learner; observations are fixed-length frame histories."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ImageHistoryEncoder(config)
        self.decoder = ImageHistoryDecoder(config, self.encoder.encoded_shape)
        self.rssm = RSSM(config)
        feature_dim = config.rssm_hidden_dim + config.rssm_stochastic_dim
        self.reward = nn.Sequential(nn.Linear(feature_dim + config.action_dim, config.model_dim), nn.ELU(), nn.Linear(config.model_dim, 1))
        self.continue_head = nn.Sequential(nn.Linear(feature_dim, config.model_dim), nn.ELU(), nn.Linear(config.model_dim, 1))
        self.actor = Actor(feature_dim, config.action_dim, config.model_dim)
        self.value = ValueModel(feature_dim, config.model_dim)

    def observe(self, frames: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor) -> tuple[list[RSSMState], torch.Tensor]:
        """Infer ``s_t`` from ``o_t`` using the preceding action ``a_{t-1}``."""
        if frames.ndim != 6 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch, time, 3, channels, height, width].")
        batch, time = frames.shape[:2]
        if actions.shape[:2] != (batch, time):
            raise ValueError("actions must have shape [batch, time, action_dim].")
        embeddings = self.encoder(frames.flatten(0, 1), masks.flatten(0, 1)).reshape(batch, time, -1)
        state = self.rssm.initial(batch, device=frames.device, dtype=frames.dtype)
        previous_action = torch.zeros(batch, self.config.action_dim, device=frames.device, dtype=frames.dtype)
        states, kls = [], []
        for embedding, action in zip(embeddings.unbind(1), actions.unbind(1), strict=True):
            state, prior, posterior = self.rssm.observe_step(state, previous_action, embedding)
            states.append(state)
            kls.append(rssm_kl_loss(prior[0], prior[1], posterior[0], posterior[1]))
            previous_action = action
        return states, torch.stack(kls, dim=1)

    def imagine(self, start: RSSMState, horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = start
        features, actions, rewards, continues = [], [], [], []
        for _ in range(horizon):
            action = self.actor(state.features)
            state = self.rssm.imagine_step(state, action)
            feature = state.features
            reward = self.reward(torch.cat((feature, action), dim=-1))
            features.append(feature)
            actions.append(action)
            rewards.append(reward)
            continues.append(self.continue_head(feature).sigmoid())
        return torch.stack(features, 1), torch.stack(rewards, 1), torch.stack(continues, 1)

    def loss(self, frames: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, *, discount: float = 0.99, lambda_: float = 0.95) -> DreamerLoss:
        states, kl = self.observe(frames, masks, actions)
        start = states[-1]
        features, imagined_rewards, continues = self.imagine(start, actions.shape[1])
        values = self.value(features)
        targets = lambda_return(imagined_rewards, values, continues, discount, lambda_)
        actor_loss = -targets.mean()
        value_loss = (values - targets.detach()).square().mean()
        observed_features = torch.stack([s.features for s in states], dim=1)
        observed_rewards = self.reward(torch.cat((observed_features, actions), dim=-1))
        reconstructed = self.decoder(torch.stack([s.z for s in states], dim=1).flatten(0, 1))
        target_frames = frames.flatten(0, 1)
        reconstruction_loss = (reconstructed - target_frames).square().mean()
        observed_rewards = observed_rewards[..., 0]
        world_model_loss = kl.mean() + (observed_rewards - rewards[..., 0]).square().mean() + reconstruction_loss
        return DreamerLoss(world_model_loss, actor_loss, value_loss)
