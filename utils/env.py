"""Environment helpers for data collection."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def observation_to_array(observation: object) -> np.ndarray:
    """Convert Gym observations (including robotics dicts) to a stable vector."""
    if isinstance(observation, dict):
        parts = [observation_to_array(observation[key]).reshape(-1) for key in sorted(observation)]
        return np.concatenate(parts).astype(np.float32, copy=False)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def make_env(env_id: str, *, render_mode: str | None = None) -> gym.Env:
    """Creates an environment with continuous actions for collection."""
    if env_id == "FetchPickAndPlace-v4":
        try:
            import gymnasium_robotics  # noqa: F401  # registers robotics environments
        except ImportError as error:
            raise ImportError(
                "Robotics environments require gymnasium-robotics[mujoco]; run ./venv.sh."
            ) from error
        version = getattr(gymnasium_robotics, "__version__", "0.0.0")
        import mujoco

        robotics_version = tuple(int(part) for part in version.split(".")[:3])
        mujoco_version = tuple(int(part) for part in mujoco.__version__.split(".")[:3])
        if robotics_version < (1, 4, 3) and mujoco_version >= (3, 3, 0):
            raise RuntimeError(
                f"FetchPickAndPlace-v4 with gymnasium-robotics {version} requires "
                f"mujoco<3.3 (found {mujoco.__version__}); run ./venv.sh to update the environment."
            )
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
    return observation_to_array(obs)
