"""Tests for resumable training checkpoints."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from trans_wm import train as trans_wm_train
from trans_wm_le import train as trans_wm_le_train
from utils.replay_buffer import RolloutReplayBuffer


class _Stateful:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self) -> dict:
        return {"state": torch.tensor([1.0])}

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


class TrainingCheckpointTest(unittest.TestCase):
    def test_value_gamma_is_fixed_for_both_models(self) -> None:
        self.assertEqual(trans_wm_train.VALUE_GAMMA, 0.95)
        self.assertEqual(trans_wm_le_train.VALUE_GAMMA, 0.95)

    def test_rolling_checkpoints_keep_latest_two_and_resolve_latest(self) -> None:
        for module in (trans_wm_train, trans_wm_le_train):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                (output_dir / "checkpoint_best.pt").touch()
                with mock.patch.object(
                    module, "_save_checkpoint", side_effect=lambda path, *args: path.touch()
                ):
                    for rollout in (2, 10, 1_000_000):
                        module._save_rolling_checkpoint(
                            output_dir, *(None for _ in range(8)), rollout, 0.0, 0
                        )

                self.assertEqual(
                    sorted(path.name for path in output_dir.glob("checkpoint_*.pt")),
                    ["checkpoint_000010.pt", "checkpoint_1000000.pt", "checkpoint_best.pt"],
                )
                self.assertEqual(
                    module._resolve_resume_checkpoint(output_dir).name,
                    "checkpoint_1000000.pt",
                )

                legacy_dir = output_dir / "legacy"
                legacy_dir.mkdir()
                legacy_checkpoint = legacy_dir / "checkpoint.pt"
                legacy_checkpoint.touch()
                self.assertEqual(
                    module._resolve_resume_checkpoint(legacy_dir), legacy_checkpoint
                )

    def test_checkpoint_restores_training_state_and_rngs(self) -> None:
        cases = (
            (
                trans_wm_train,
                trans_wm_train.WorldModelConfig(
                    observation_shape=(3, 32, 32), action_shape=(1,), cnn_channels=(4,)
                ),
                trans_wm_train.TrainingConfig(),
            ),
            (
                trans_wm_le_train,
                trans_wm_le_train.WorldModelConfig(
                    observation_shape=(3, 32, 32), action_shape=(1,), cnn_channels=(4,)
                ),
                trans_wm_le_train.TrainingConfig(),
            ),
        )
        for module, model_config, training_config in cases:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint_000010.pt"
                model = _Stateful()
                trainer = SimpleNamespace(optimizer=_Stateful(), value_optimizer=_Stateful())
                rng = np.random.default_rng(1)
                replay_buffer = RolloutReplayBuffer.__new__(RolloutReplayBuffer)
                replay_buffer._rng = np.random.default_rng(2)
                policy_generator = torch.Generator().manual_seed(3)
                evaluation_generator = torch.Generator().manual_seed(4)

                module._save_checkpoint(
                    path,
                    model,
                    trainer,
                    model_config,
                    training_config,
                    rng,
                    replay_buffer,
                    policy_generator,
                    evaluation_generator,
                    10,
                    -123.0,
                    2,
                )
                expected = (
                    rng.integers(1_000_000),
                    replay_buffer._rng.integers(1_000_000),
                    torch.randint(1_000_000, (), generator=policy_generator),
                    torch.randint(1_000_000, (), generator=evaluation_generator),
                )

                rng.random(10)
                replay_buffer._rng.random(10)
                torch.rand(10, generator=policy_generator)
                torch.rand(10, generator=evaluation_generator)
                restored = module._restore_checkpoint(
                    path,
                    model,
                    trainer,
                    model_config,
                    training_config,
                    rng,
                    replay_buffer,
                    policy_generator,
                    evaluation_generator,
                    torch.device("cpu"),
                )

                self.assertEqual(restored, (10, -123.0, 2))
                actual = (
                    rng.integers(1_000_000),
                    replay_buffer._rng.integers(1_000_000),
                    torch.randint(1_000_000, (), generator=policy_generator),
                    torch.randint(1_000_000, (), generator=evaluation_generator),
                )
                for actual_value, expected_value in zip(actual, expected):
                    self.assertEqual(actual_value, expected_value)


if __name__ == "__main__":
    unittest.main()
