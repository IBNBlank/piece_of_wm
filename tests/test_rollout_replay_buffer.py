"""Tests for sequence-preserving, multi-environment replay rollouts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.replay_buffer import EpisodeBatch, METADATA_FILE, ROLLOUT_FILE_TEMPLATE, RolloutReplayBuffer


def _episode_batch(value: float, lengths: tuple[int, int] = (2, 3)) -> EpisodeBatch:
    num_envs = len(lengths)
    max_steps = max(lengths)
    obs = np.zeros((num_envs, max_steps + 1, 2), dtype=np.float32)
    action = np.zeros((num_envs, max_steps, 1), dtype=np.float32)
    reward = np.zeros((num_envs, max_steps), dtype=np.float32)
    terminated = np.zeros((num_envs, max_steps), dtype=bool)
    truncated = np.zeros((num_envs, max_steps), dtype=bool)
    for env_index, length in enumerate(lengths):
        obs[env_index, : length + 1] = value + env_index
        action[env_index, :length] = value + env_index
        reward[env_index, :length] = value + env_index
        truncated[env_index, length - 1] = True
    return EpisodeBatch(obs, action, reward, terminated, truncated, np.asarray(lengths, dtype=np.int64))


def _flat_rollout() -> dict[str, np.ndarray]:
    """Two interleaved environment episodes with lengths two and three."""
    values = np.asarray([1.0, 10.0, 2.0, 11.0, 12.0], dtype=np.float32)
    return {
        "obs": np.stack((values, values), axis=1),
        "action": values[:, None],
        "next_obs": np.stack((values + 0.5, values + 0.5), axis=1),
        "reward": values,
        "terminated": np.zeros(5, dtype=bool),
        "truncated": np.asarray([False, False, True, False, True]),
        "trajectory_indices": np.empty(0, dtype=np.float64),
    }


class RolloutReplayBufferTest(unittest.TestCase):
    def test_imports_flat_multi_env_rollout_as_complete_episodes_in_ram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "dataset"
            source_dir.mkdir()
            filename = ROLLOUT_FILE_TEMPLATE.format(rollout_index=0)
            np.savez_compressed(source_dir / filename, **_flat_rollout())
            (source_dir / METADATA_FILE).write_text(
                json.dumps({"format": "rollout-npz-v1", "rollout_files": [filename], "num_envs": 2}),
                encoding="utf-8",
            )

            buffer = RolloutReplayBuffer(root / "runs" / "trial", source_dir=source_dir, max_rollouts=2, seed=4)
            batch = buffer.sample(1)

            self.assertEqual(batch.obs.shape, (2, 4, 2))
            self.assertEqual(batch.action.shape, (2, 3, 1))
            np.testing.assert_array_equal(batch.lengths, [2, 3])
            np.testing.assert_array_equal(batch.valid, [[True, True, False], [True, True, True]])
            np.testing.assert_array_equal(batch.obs[0, :3, 0], [1.0, 2.0, 2.5])
            np.testing.assert_array_equal(batch.obs[1, :4, 0], [10.0, 11.0, 12.0, 12.5])
            self.assertNotIsInstance(batch.obs, np.memmap)
            (buffer.storage_dir / filename).unlink()
            self.assertEqual(buffer.sample(1).num_transitions, 5)

    def test_add_rollout_evicts_oldest_sequence_file_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_dir = root / "dataset"
            source_dir.mkdir()
            filename = ROLLOUT_FILE_TEMPLATE.format(rollout_index=0)
            np.savez_compressed(source_dir / filename, **_flat_rollout())
            (source_dir / METADATA_FILE).write_text(
                json.dumps({"format": "rollout-npz-v1", "rollout_files": [filename], "num_envs": 2}),
                encoding="utf-8",
            )

            run_dir = root / "runs" / "trial"
            buffer = RolloutReplayBuffer(run_dir, source_dir=source_dir, max_rollouts=2, seed=9)
            buffer.add_rollout(_episode_batch(20.0))
            buffer.add_rollout(_episode_batch(30.0))

            metadata = json.loads((run_dir / "replay_buffer" / METADATA_FILE).read_text(encoding="utf-8"))
            self.assertEqual(metadata["num_rollouts"], 2)
            self.assertEqual(metadata["num_transitions"], 10)
            self.assertFalse((run_dir / "replay_buffer" / filename).exists())
            persisted_file = run_dir / "replay_buffer" / metadata["rollout_files"][0]
            with np.load(persisted_file) as data:
                self.assertEqual(data["obs"].shape, (2, 4, 2))
                self.assertEqual(data["lengths"].shape, (2,))

            restored = RolloutReplayBuffer(run_dir, max_rollouts=2, seed=9)
            self.assertEqual(restored.num_rollouts, 2)
            restored_batch = restored.sample()
            self.assertEqual(restored_batch.obs.shape, (4, 4, 2))
            self.assertEqual(restored_batch.num_transitions, 10)
            self.assertTrue(
                np.all(restored_batch.truncated[np.arange(4), restored_batch.lengths - 1])
            )


if __name__ == "__main__":
    unittest.main()
