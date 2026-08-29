"""Loss utilities for Dreamer v1."""

from __future__ import annotations

import torch


def rssm_kl_loss(
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q(z|h,o) || p(z|h)), reduced over the stochastic dimension."""
    prior_var = (2.0 * prior_log_std).exp()
    posterior_var = (2.0 * posterior_log_std).exp()
    return 0.5 * ((posterior_var + (posterior_mean - prior_mean).square()) / prior_var - 1.0 + 2.0 * (prior_log_std - posterior_log_std)).mean(dim=-1)


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    continuation: torch.Tensor,
    discount: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute TD(lambda) targets over imagined time."""
    if reward.shape != value.shape or reward.shape != continuation.shape:
        raise ValueError("reward, value, and continuation must have identical shapes.")
    target = value[:, -1]
    returns: list[torch.Tensor] = []
    for index in range(reward.shape[1] - 1, -1, -1):
        target = reward[:, index] + discount * continuation[:, index] * ((1.0 - lambda_) * value[:, index] + lambda_ * target)
        returns.append(target)
    return torch.stack(returns[::-1], dim=1)
