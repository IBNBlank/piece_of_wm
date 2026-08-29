"""Configuration for the image-history world model."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


OBS_HISTORY_LEN = 3


@dataclass(frozen=True)
class WorldModelConfig:
    """Architecture and training constants for channel-first image observations."""

    observation_shape: tuple[int, int, int]
    action_shape: tuple[int, ...]
    observation_dim: int = 128
    latent_dim: int = 128
    model_dim: int = 256
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    rssm_hidden_dim: int = 256
    rssm_stochastic_dim: int = 128

    def __post_init__(self) -> None:
        if len(self.observation_shape) != 3 or any(size <= 0 for size in self.observation_shape):
            raise ValueError("observation_shape must be positive (channels, height, width).")
        if not self.action_shape or any(size <= 0 for size in self.action_shape):
            raise ValueError("action_shape must contain positive dimensions.")
        if not self.cnn_channels or any(size <= 0 for size in self.cnn_channels):
            raise ValueError("cnn_channels must contain positive dimensions.")
        if self.observation_dim <= 0 or self.model_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        if self.rssm_hidden_dim <= 0 or self.rssm_stochastic_dim <= 0:
            raise ValueError("RSSM dimensions must be positive.")
        if self.latent_dim != 128:
            raise ValueError("latent_dim is fixed at 128 for dreamer_like.")
        if self.rssm_stochastic_dim != self.latent_dim:
            raise ValueError("rssm_stochastic_dim must match latent_dim for shared reward/decoder heads.")

    @property
    def channels(self) -> int:
        return self.observation_shape[0]

    @property
    def height(self) -> int:
        return self.observation_shape[1]

    @property
    def width(self) -> int:
        return self.observation_shape[2]

    @property
    def action_dim(self) -> int:
        return prod(self.action_shape)
