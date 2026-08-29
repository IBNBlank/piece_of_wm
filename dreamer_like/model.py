"""Neural components used by the multi-frame Dreamer-v1 learner."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from dreamer_like.config import OBS_HISTORY_LEN, WorldModelConfig


@dataclass(frozen=True)
class RSSMState:
    h: torch.Tensor
    z: torch.Tensor

    @property
    def features(self) -> torch.Tensor:
        return torch.cat((self.h, self.z), dim=-1)


class RSSM(nn.Module):
    """Recurrent state-space model with Gaussian prior and posterior."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.rssm_hidden_dim
        self.stochastic_dim = config.rssm_stochastic_dim
        self.action_dim = config.action_dim
        self.cell = nn.GRUCell(self.stochastic_dim + self.action_dim, self.hidden_dim)
        self.prior = nn.Linear(self.hidden_dim, 2 * self.stochastic_dim)
        self.posterior = nn.Linear(self.hidden_dim + config.observation_dim, 2 * self.stochastic_dim)

    def initial(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> RSSMState:
        return RSSMState(torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype), torch.zeros(batch, self.stochastic_dim, device=device, dtype=dtype))

    @staticmethod
    def _stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = values.chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def imagine_step(self, state: RSSMState, action: torch.Tensor) -> RSSMState:
        h = self.cell(torch.cat((state.z, action), dim=-1), state.h)
        mean, log_std = self._stats(self.prior(h))
        return RSSMState(h, mean + log_std.exp() * torch.randn_like(mean))

    def observe_step(self, state: RSSMState, action: torch.Tensor, observation: torch.Tensor) -> tuple[RSSMState, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Advance with the preceding action, then condition z on the observation."""
        h = self.cell(torch.cat((state.z, action), dim=-1), state.h)
        prior = self._stats(self.prior(h))
        posterior = self._stats(self.posterior(torch.cat((h, observation), dim=-1)))
        z = posterior[0] + posterior[1].exp() * torch.randn_like(posterior[0])
        return RSSMState(h, z), prior, posterior


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, model_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(feature_dim, model_dim), nn.ELU(), nn.Linear(model_dim, model_dim), nn.ELU())
        self.mean = nn.Linear(model_dim, action_dim)
        self.log_std = nn.Linear(model_dim, action_dim)

    def forward(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        hidden = self.body(features)
        mean = self.mean(hidden)
        if deterministic:
            return torch.tanh(mean)
        std = self.log_std(hidden).clamp(-5.0, 2.0).exp()
        return torch.tanh(mean + std * torch.randn_like(mean))


class ValueModel(nn.Module):
    def __init__(self, feature_dim: int, model_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, model_dim), nn.ELU(), nn.Linear(model_dim, model_dim), nn.ELU(), nn.Linear(model_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ImageHistoryEncoder(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.observation_shape = config.observation_shape
        layers: list[nn.Module] = []
        in_channels = OBS_HISTORY_LEN * config.channels
        for out_channels in config.cnn_channels:
            layers.extend((nn.Conv2d(in_channels, out_channels, 4, 2, 1), nn.GELU()))
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            encoded = self.cnn(torch.zeros(1, OBS_HISTORY_LEN * config.channels, config.height, config.width))
        self.encoded_shape = tuple(encoded.shape[1:])
        self.to_observation = nn.Sequential(nn.Flatten(), nn.Linear(prod(self.encoded_shape), config.observation_dim))

    def forward(self, obs_history: torch.Tensor, obs_valid_mask: torch.Tensor) -> torch.Tensor:
        expected = (OBS_HISTORY_LEN, *self.observation_shape)
        if obs_history.ndim != 5 or tuple(obs_history.shape[1:]) != expected:
            raise ValueError(f"obs_history must have shape [batch, {expected}].")
        if obs_valid_mask.shape != obs_history.shape[:2]:
            raise ValueError("obs_valid_mask must have shape [batch, 3].")
        stacked = (obs_history * obs_valid_mask[:, :, None, None, None]).flatten(1, 2)
        return self.to_observation(self.cnn(stacked))


class ImageHistoryDecoder(nn.Module):
    def __init__(self, config: WorldModelConfig, encoded_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.observation_shape = config.observation_shape
        self.encoded_shape = encoded_shape
        self.from_latent = nn.Linear(config.rssm_stochastic_dim, prod(encoded_shape))
        channels = list(reversed(config.cnn_channels))
        layers: list[nn.Module] = []
        for index, in_channels in enumerate(channels):
            out_channels = channels[index + 1] if index + 1 < len(channels) else OBS_HISTORY_LEN * config.channels
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            if index + 1 < len(channels):
                layers.append(nn.GELU())
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(self.from_latent(z).reshape(z.shape[0], *self.encoded_shape))
        if decoded.shape[-2:] != self.observation_shape[-2:]:
            decoded = F.interpolate(decoded, size=self.observation_shape[-2:], mode="bilinear", align_corners=False)
        return decoded.reshape(z.shape[0], OBS_HISTORY_LEN, *self.observation_shape)
