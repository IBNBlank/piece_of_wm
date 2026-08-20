"""Environment helpers for data collection."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def make_env(env_id: str, *, render_mode: str | None = None) -> gym.Env:
    """Creates an environment with continuous actions for collection."""
    env = gym.make(env_id, render_mode=render_mode)
    if not isinstance(env.action_space, gym.spaces.Box):
        env.close()
        raise TypeError(
            f"{env_id} has {type(env.action_space).__name__} actions; "
            "this collector requires a continuous gymnasium.spaces.Box action space."
        )
    return env


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    """Resets an environment and normalizes observations to float32 arrays."""
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)
