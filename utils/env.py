"""Environment helpers shared by data collection and PETS training."""

from __future__ import annotations

from typing import Callable, Tuple

import gymnasium as gym
import numpy as np
import torch


def make_env(env_id: str, *, render_mode: str | None = None) -> gym.Env:
    """Creates an environment and validates that PETS can sample continuous actions."""
    env = gym.make(env_id, render_mode=render_mode)
    if not isinstance(env.action_space, gym.spaces.Box):
        env.close()
        raise TypeError(
            f"{env_id} has {type(env.action_space).__name__} actions; "
            "this PETS implementation requires a continuous gymnasium.spaces.Box action space."
        )
    return env


def space_shapes(env: gym.Env) -> Tuple[tuple[int, ...], tuple[int, ...]]:
    """Returns observation and action shapes for a Box-to-Box environment."""
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            f"{type(env.observation_space).__name__} observations are unsupported; "
            "only gymnasium.spaces.Box observations can be stored by this project."
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("PETS requires a gymnasium.spaces.Box action space.")
    return tuple(env.observation_space.shape), tuple(env.action_space.shape)


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    """Resets an environment and normalizes observations to float32 arrays."""
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def no_termination(_actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """Model-rollout termination for tasks such as Pendulum without terminal states.

    PETS learns state deltas and rewards, but not terminal flags. The planner therefore
    rolls out for its fixed horizon. Real environment termination is still respected by
    the collection loops.
    """
    return torch.zeros((next_obs.shape[0], 1), dtype=torch.bool, device=next_obs.device)


TerminationFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
