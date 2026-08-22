"""Value-function modules shared by the world models."""

from __future__ import annotations

import torch
from torch import nn


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Monte Carlo return-to-go for one complete online episode."""
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty one-dimensional tensor.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    returns = torch.empty_like(rewards)
    running = rewards.new_zeros(())
    for timestep in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[timestep] + gamma * running
        returns[timestep] = running
    return returns


def lambda_returns(
    rewards: torch.Tensor,
    bootstrap_values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Computes finite-horizon TD(lambda) targets from aligned next-state values."""
    if rewards.ndim != 1 or bootstrap_values.shape != rewards.shape:
        raise ValueError("rewards and bootstrap_values must be one-dimensional and aligned.")
    if rewards.numel() == 0:
        raise ValueError("rewards must not be empty.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1].")
    returns = torch.empty_like(rewards)
    next_return = bootstrap_values[-1]
    for timestep in range(len(rewards) - 1, -1, -1):
        bootstrap = bootstrap_values[timestep]
        returns[timestep] = rewards[timestep] + gamma * (
            (1.0 - lambda_) * bootstrap + lambda_ * next_return
        )
        next_return = returns[timestep]
    return returns


class EnsembleValueHead(nn.Module):
    """Independent critics with explicit pessimistic and planning aggregates."""

    def __init__(self, input_dim: int, hidden_dim: int, num_critics: int) -> None:
        super().__init__()
        self.critics = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_critics)
        )

    def all_values(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cat([critic(latent) for critic in self.critics], dim=-1)

    def minimum(self, latent: torch.Tensor) -> torch.Tensor:
        return self.all_values(latent).min(dim=-1, keepdim=True).values

    def mean(self, latent: torch.Tensor) -> torch.Tensor:
        return self.all_values(latent).mean(dim=-1, keepdim=True)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.minimum(latent)
