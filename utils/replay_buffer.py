"""Sequence-preserving replay buffer for multi-step world-model training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METADATA_FILE = "dataset.json"
ROLLOUT_FILE_TEMPLATE = "rollout_{rollout_index:06d}.npz"
FORMAT = "episode-rollout-replay-buffer-v1"
DEFAULT_SAMPLE_ROLLOUTS = 2


@dataclass(frozen=True)
class EpisodeBatch:
    """One complete episode per environment, padded on the time axis."""

    obs: np.ndarray  # (N, T + 1, *obs_shape)
    action: np.ndarray  # (N, T, *action_shape)
    reward: np.ndarray  # (N, T)
    terminated: np.ndarray  # (N, T)
    truncated: np.ndarray  # (N, T)
    lengths: np.ndarray  # (N,)
    images: np.ndarray | None = None  # (N, T + 1, H, W, C)

    @property
    def valid(self) -> np.ndarray:
        return np.arange(self.action.shape[1])[None] < self.lengths[:, None]

    @property
    def next_obs(self) -> np.ndarray:
        return self.obs[:, 1:]

    @property
    def num_transitions(self) -> int:
        return int(self.lengths.sum())


class RolloutReplayBuffer:
    """RAM-resident FIFO of ``NUM_ENVS`` complete episodes per rollout file."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        source_dir: str | Path | None = None,
        max_rollouts: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.storage_dir = Path(run_dir) / "replay_buffer"
        self._rng = np.random.default_rng(seed)
        self._batches: list[EpisodeBatch] = []
        self._filenames: list[str] = []
        metadata_path = self.storage_dir / METADATA_FILE

        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("format") != FORMAT:
                raise ValueError(f"Unsupported replay-buffer format: {metadata.get('format')!r}.")
            self.max_rollouts = self._capacity(
                max_rollouts if max_rollouts is not None else metadata.get("max_rollouts")
            )
            self._filenames = list(metadata["rollout_files"])
            self._batches = [self._read(self.storage_dir / name) for name in self._filenames]
            self._metadata = metadata
            self._evict_excess()
            return

        if source_dir is None:
            raise FileNotFoundError(f"Replay buffer not found at {self.storage_dir}; source_dir is required.")
        source_dir = Path(source_dir)
        source_metadata = self._read_metadata(source_dir)
        source_files = source_metadata.get("rollout_files") or ["replay_buffer.npz"]
        self.max_rollouts = self._capacity(max_rollouts if max_rollouts is not None else len(source_files))
        num_envs = int(source_metadata.get("num_envs", 1))
        self._metadata = {"num_envs": num_envs, **source_metadata}
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for filename in source_files[-self.max_rollouts :]:
            self._append(self._read_source(source_dir / filename, num_envs))
        self._save_metadata()

    @property
    def num_rollouts(self) -> int:
        return len(self._batches)

    @property
    def num_stored(self) -> int:
        return sum(batch.num_transitions for batch in self._batches)

    def rng_state(self) -> dict:
        return self._rng.bit_generator.state

    def load_rng_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state

    def add_rollout(self, batch: EpisodeBatch) -> None:
        """Adds a complete ``NUM_ENVS``-episode rollout and evicts the oldest."""
        self._append(self._validate_and_copy(batch))
        self._evict_excess()
        self._save_metadata()

    def sample(self, num_rollouts: int = DEFAULT_SAMPLE_ROLLOUTS) -> EpisodeBatch:
        """Returns complete episodes from random rollouts directly from RAM."""
        if not self._batches:
            raise RuntimeError("Cannot sample an empty replay buffer.")
        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive.")
        indices = self._rng.choice(
            self.num_rollouts,
            size=num_rollouts,
            replace=self.num_rollouts < num_rollouts,
        )
        return self._combine_batches([self._batches[index] for index in indices])

    def _append(self, batch: EpisodeBatch) -> None:
        filename = ROLLOUT_FILE_TEMPLATE.format(rollout_index=self._next_index())
        self._write(self.storage_dir / filename, batch)
        self._filenames.append(filename)
        self._batches.append(batch)

    def _evict_excess(self) -> None:
        while len(self._batches) > self.max_rollouts:
            self._batches.pop(0)
            (self.storage_dir / self._filenames.pop(0)).unlink()

    def _next_index(self) -> int:
        indices = [int(Path(name).stem.removeprefix("rollout_")) for name in self._filenames]
        return max(indices, default=-1) + 1

    def _save_metadata(self) -> None:
        metadata = {
            **self._metadata,
            "format": FORMAT,
            "max_rollouts": self.max_rollouts,
            "rollout_files": self._filenames,
            "num_rollouts": self.num_rollouts,
            "num_transitions": self.num_stored,
        }
        (self.storage_dir / METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._metadata = metadata

    @staticmethod
    def _capacity(value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("max_rollouts must be a positive integer.")
        return value

    @staticmethod
    def _read_metadata(directory: Path) -> dict:
        path = directory / METADATA_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    @classmethod
    def _read_source(cls, path: Path, num_envs: int) -> EpisodeBatch:
        with np.load(path) as data:
            arrays = {name: np.array(data[name], copy=True) for name in data.files}
        if "lengths" in arrays:
            return cls._from_sequence(arrays)
        return cls._from_flat(arrays, num_envs)

    @classmethod
    def _read(cls, path: Path) -> EpisodeBatch:
        if not path.is_file():
            raise FileNotFoundError(f"Rollout file not found: {path}")
        return cls._read_source(path, 1)

    @classmethod
    def _from_sequence(cls, arrays: dict[str, np.ndarray]) -> EpisodeBatch:
        required = ("obs", "action", "reward", "terminated", "truncated", "lengths")
        if missing := [name for name in required if name not in arrays]:
            raise ValueError(f"Sequence rollout is missing arrays: {missing}")
        return cls._validate_and_copy(
            EpisodeBatch(
                arrays["obs"], arrays["action"], arrays["reward"], arrays["terminated"],
                arrays["truncated"], arrays["lengths"], arrays.get("images"),
            )
        )

    @classmethod
    def _from_flat(cls, arrays: dict[str, np.ndarray], num_envs: int) -> EpisodeBatch:
        required = ("obs", "action", "next_obs", "reward", "terminated", "truncated")
        if missing := [name for name in required if name not in arrays]:
            raise ValueError(f"Transition rollout is missing arrays: {missing}")
        episode_indices = [[] for _ in range(num_envs)]
        active = [True] * num_envs
        index = 0
        while index < len(arrays["obs"]):
            for env_index, is_active in enumerate(active):
                if is_active:
                    if index >= len(arrays["obs"]):
                        raise ValueError("Rollout ended before every active environment episode completed.")
                    episode_indices[env_index].append(index)
                    active[env_index] = not bool(arrays["terminated"][index] or arrays["truncated"][index])
                    index += 1
            if not any(active):
                break
        if index != len(arrays["obs"]) or any(not indices for indices in episode_indices):
            raise ValueError("Rollout does not contain one complete episode per environment.")
        lengths = np.asarray([len(indices) for indices in episode_indices], dtype=np.int64)
        max_steps = int(lengths.max())
        obs = np.zeros((num_envs, max_steps + 1, *arrays["obs"].shape[1:]), dtype=arrays["obs"].dtype)
        action = np.zeros((num_envs, max_steps, *arrays["action"].shape[1:]), dtype=arrays["action"].dtype)
        reward = np.zeros((num_envs, max_steps), dtype=arrays["reward"].dtype)
        terminated = np.zeros((num_envs, max_steps), dtype=bool)
        truncated = np.zeros((num_envs, max_steps), dtype=bool)
        for env_index, indices in enumerate(episode_indices):
            length = len(indices)
            obs[env_index, :length] = arrays["obs"][indices]
            obs[env_index, length] = arrays["next_obs"][indices[-1]]
            action[env_index, :length] = arrays["action"][indices]
            reward[env_index, :length] = arrays["reward"][indices]
            terminated[env_index, :length] = arrays["terminated"][indices]
            truncated[env_index, :length] = arrays["truncated"][indices]
        images = None
        if "images" in arrays and "next_images" in arrays:
            images = np.zeros((num_envs, max_steps + 1, *arrays["images"].shape[1:]), dtype=arrays["images"].dtype)
            for env_index, indices in enumerate(episode_indices):
                length = len(indices)
                images[env_index, :length] = arrays["images"][indices]
                images[env_index, length] = arrays["next_images"][indices[-1]]
        return cls._validate_and_copy(EpisodeBatch(obs, action, reward, terminated, truncated, lengths, images))

    @staticmethod
    def _validate_and_copy(batch: EpisodeBatch) -> EpisodeBatch:
        if not isinstance(batch, EpisodeBatch):
            raise TypeError("add_rollout expects an EpisodeBatch.")
        copied = EpisodeBatch(
            *(np.array(value, copy=True) for value in (
                batch.obs, batch.action, batch.reward, batch.terminated, batch.truncated, batch.lengths
            )),
            images=None if batch.images is None else np.array(batch.images, copy=True),
        )
        if copied.obs.shape[:2] != (copied.action.shape[0], copied.action.shape[1] + 1):
            raise ValueError("obs must have shape (num_envs, max_steps + 1, ...).")
        if any(array.shape != copied.action.shape[:2] for array in (copied.reward, copied.terminated, copied.truncated)):
            raise ValueError("Transition targets must have shape (num_envs, max_steps).")
        if copied.lengths.shape != (copied.action.shape[0],) or np.any(copied.lengths <= 0):
            raise ValueError("lengths must contain one positive length per environment.")
        last = copied.lengths - 1
        if np.any(last >= copied.action.shape[1]) or not np.all(copied.terminated[np.arange(len(last)), last] | copied.truncated[np.arange(len(last)), last]):
            raise ValueError("Each episode must end with terminated or truncated.")
        if copied.images is not None and copied.images.shape[:2] != copied.obs.shape[:2]:
            raise ValueError("images must align with obs on environment and time axes.")
        return copied

    @staticmethod
    def _combine_batches(batches: list[EpisodeBatch]) -> EpisodeBatch:
        max_steps = max(batch.action.shape[1] for batch in batches)
        num_envs = sum(len(batch.lengths) for batch in batches)
        first = batches[0]
        obs = np.zeros((num_envs, max_steps + 1, *first.obs.shape[2:]), dtype=first.obs.dtype)
        action = np.zeros((num_envs, max_steps, *first.action.shape[2:]), dtype=first.action.dtype)
        reward = np.zeros((num_envs, max_steps), dtype=first.reward.dtype)
        terminated = np.zeros((num_envs, max_steps), dtype=bool)
        truncated = np.zeros((num_envs, max_steps), dtype=bool)
        lengths = np.empty(num_envs, dtype=first.lengths.dtype)
        images = None
        if any((batch.images is None) != (first.images is None) for batch in batches):
            raise ValueError("Cannot combine rollouts with and without images.")
        if first.images is not None:
            images = np.zeros((num_envs, max_steps + 1, *first.images.shape[2:]), dtype=first.images.dtype)
        offset = 0
        for batch in batches:
            count = len(batch.lengths)
            steps = batch.action.shape[1]
            obs[offset : offset + count, : steps + 1] = batch.obs
            action[offset : offset + count, :steps] = batch.action
            reward[offset : offset + count, :steps] = batch.reward
            terminated[offset : offset + count, :steps] = batch.terminated
            truncated[offset : offset + count, :steps] = batch.truncated
            lengths[offset : offset + count] = batch.lengths
            if images is not None:
                images[offset : offset + count, : steps + 1] = batch.images
            offset += count
        return EpisodeBatch(obs, action, reward, terminated, truncated, lengths, images)

    @staticmethod
    def _write(path: Path, batch: EpisodeBatch) -> None:
        arrays = {"obs": batch.obs, "action": batch.action, "reward": batch.reward,
                  "terminated": batch.terminated, "truncated": batch.truncated, "lengths": batch.lengths}
        if batch.images is not None:
            arrays["images"] = batch.images
        np.savez_compressed(path, **arrays)
