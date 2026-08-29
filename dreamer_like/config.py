"""Configuration for the image-history world model."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


OBS_HISTORY_LEN = 3
ACTION_HISTORY_LEN = 2


@dataclass(frozen=True)
class WorldModelConfig:
    """Architecture and training constants for channel-first image observations."""

    observation_shape: tuple[int, int, int]
    action_shape: tuple[int, ...]
    observation_dim: int = 128
    latent_dim: int = 128
    model_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    feedforward_dim: int = 512
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.0
    target_ema: float = 0.99

    def __post_init__(self) -> None:
        if len(self.observation_shape) != 3 or any(size <= 0 for size in self.observation_shape):
            raise ValueError("observation_shape must be positive (channels, height, width).")
        if not self.action_shape or any(size <= 0 for size in self.action_shape):
            raise ValueError("action_shape must contain positive dimensions.")
        if not self.cnn_channels or any(size <= 0 for size in self.cnn_channels):
            raise ValueError("cnn_channels must contain positive dimensions.")
        if self.observation_dim <= 0 or self.model_dim <= 0 or self.feedforward_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        if self.latent_dim != 128:
            raise ValueError("latent_dim is fixed at 128 for dreamer_like.")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers and num_heads must be positive.")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.target_ema < 1.0:
            raise ValueError("target_ema must be in [0, 1).")

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

    @property
    def action_history_dim(self) -> int:
        return ACTION_HISTORY_LEN * self.action_dim
