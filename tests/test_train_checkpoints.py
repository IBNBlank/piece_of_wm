"""Tests for resumable training checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from trans_wm import train as trans_wm_train
from trans_wm_le import train as trans_wm_le_train
from utils import training_runtime
from utils.replay_buffer import RolloutReplayBuffer


class _Stateful:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self) -> dict:
        return {"state": torch.tensor([1.0])}

    def load_state_dict(self, state: dict, strict: bool = True) -> SimpleNamespace:
        self.loaded = state
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class TrainingCheckpointTest(unittest.TestCase):
    def test_pretraining_entrypoint_never_creates_a_gym_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                verbose=False,
                seed=0,
                device="cpu",
                data_dir=Path(directory) / "dataset",
                output_dir=Path(directory) / "run",
                num_envs=1,
                max_steps=2,
                replay_capacity=None,
                pretrained_checkpoint=None,
                target_ema=0.99,
                learning_rate=1e-4,
                weight_decay=1e-5,
                grad_clip_norm=10.0,
                jepa_weight=1.0,
                sigreg_weight=0.2,
                value_weight=1.0,
                sigreg_projections=8,
                sigreg_frequencies=4,
                sigreg_max_frequency=5.0,
                planning_horizon=10,
                sample_rollouts=1,
                pretrain=True,
            )
            first_batch = SimpleNamespace(
                images=np.zeros((1, 3, 32, 32, 3), dtype=np.uint8),
                action=np.zeros((1, 2, 1), dtype=np.float32),
            )
            replay_buffer = SimpleNamespace(
                sample=mock.Mock(return_value=first_batch),
                num_rollouts=1,
            )
            model = mock.Mock()
            model.to.return_value = model
            with (
                mock.patch.object(trans_wm_le_train, "parse_args", return_value=args),
                mock.patch.object(trans_wm_le_train, "configure_logging"),
                mock.patch.object(trans_wm_le_train, "_validate_positive_args"),
                mock.patch.object(trans_wm_le_train, "seed_everything"),
                mock.patch.object(trans_wm_le_train, "_rollout_files", return_value=[]),
                mock.patch.object(trans_wm_le_train, "_validate_dataset_metadata"),
                mock.patch.object(
                    trans_wm_le_train,
                    "OfflineRolloutDataset",
                    return_value=replay_buffer,
                ),
                mock.patch.object(trans_wm_le_train, "WorldModel", return_value=model),
                mock.patch.object(trans_wm_le_train, "WorldModelTrainer", return_value=mock.Mock()),
                mock.patch.object(trans_wm_le_train, "_write_config"),
                mock.patch.object(trans_wm_le_train, "_run_pretraining") as run_pretraining,
                mock.patch.object(trans_wm_le_train, "make_env") as make_env,
            ):
                trans_wm_le_train.main()

            run_pretraining.assert_called_once()
            self.assertEqual(run_pretraining.call_args.args[4].value_weight, 0.0)
            make_env.assert_not_called()

    def test_training_cli_has_no_gamma(self) -> None:
        for module in (trans_wm_train, trans_wm_le_train):
            with self.subTest(module=module.__name__), mock.patch(
                "sys.argv", ["train", "--data-dir", "dataset"]
            ):
                self.assertFalse(hasattr(module.parse_args(), "gamma"))

    def test_training_particle_defaults(self) -> None:
        for module in (trans_wm_train, trans_wm_le_train):
            with self.subTest(module=module.__name__), mock.patch(
                "sys.argv", ["train", "--data-dir", "dataset"]
            ):
                args = module.parse_args()
                self.assertEqual(args.particle_updates, 5)
                self.assertEqual(args.num_particles, 1000)
                self.assertEqual(args.particle_temperature, 2.0)
                self.assertEqual(args.planning_horizon, 8)

    def test_rolling_checkpoints_keep_latest_two_and_resolve_latest(self) -> None:
        for module in (trans_wm_train, trans_wm_le_train):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                (output_dir / "checkpoint_best.pt").touch()
                with mock.patch.object(
                    training_runtime,
                    "save_checkpoint",
                    side_effect=lambda path, *args, **kwargs: path.touch(),
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
                trainer = SimpleNamespace(optimizer=_Stateful())
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

                self.assertEqual(restored.rollout, 10)
                self.assertEqual(restored.best_online_return, -123.0)
                self.assertEqual(restored.checks_without_improvement, 2)
                actual = (
                    rng.integers(1_000_000),
                    replay_buffer._rng.integers(1_000_000),
                    torch.randint(1_000_000, (), generator=policy_generator),
                    torch.randint(1_000_000, (), generator=evaluation_generator),
                )
                for actual_value, expected_value in zip(actual, expected):
                    self.assertEqual(actual_value, expected_value)

                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["phase"], "training")
                with self.assertRaisesRegex(ValueError, "pretraining checkpoint"):
                    module._load_pretrained_checkpoint(
                        path,
                        model,
                        model_config,
                        torch.device("cpu"),
                    )
                with self.assertRaisesRegex(ValueError, "Checkpoint phase"):
                    module._restore_checkpoint(
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
                        expected_phase="pretrain",
                    )

    def test_pretrained_checkpoint_loads_only_world_training_state(self) -> None:
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
                path = Path(directory) / "checkpoint_best.pt"
                model = _Stateful()
                trainer = SimpleNamespace(optimizer=_Stateful())
                replay_buffer = RolloutReplayBuffer.__new__(RolloutReplayBuffer)
                replay_buffer._rng = np.random.default_rng(2)
                module._save_checkpoint(
                    path,
                    model,
                    trainer,
                    model_config,
                    training_config,
                    np.random.default_rng(1),
                    replay_buffer,
                    torch.Generator().manual_seed(3),
                    torch.Generator().manual_seed(4),
                    10,
                    -float("inf"),
                    0,
                    phase="pretrain",
                    best_validation_loss=0.25,
                )

                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["phase"], "pretrain")
                self.assertEqual(checkpoint["epoch"], 10)
                self.assertNotIn("rollout", checkpoint)
                self.assertEqual(checkpoint["best_validation_loss"], 0.25)
                self.assertEqual(
                    checkpoint["validation_metric_version"],
                    training_runtime.VALIDATION_METRIC_VERSION,
                )
                module._load_pretrained_checkpoint(
                    path,
                    model,
                    model_config,
                    torch.device("cpu"),
                )
                self.assertIsNotNone(model.loaded)
                self.assertIsNone(trainer.optimizer.loaded)

    def test_pretraining_only_runs_replay_updates(self) -> None:
        for module in (trans_wm_train, trans_wm_le_train):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                trainer = SimpleNamespace(train_epoch=mock.Mock(return_value={"total": 1.0}))
                replay_buffer = SimpleNamespace(batches=(object(),), num_stored=8)
                args = SimpleNamespace(
                    resume=None,
                    epochs=3,
                    batch_size=8,
                    checkpoint_epochs=1,
                    seed=0,
                    output_dir=output_dir,
                )
                with (
                    mock.patch.object(
                        training_runtime,
                        "evaluate_validation",
                        return_value={"total": 0.5},
                    ),
                    mock.patch.object(training_runtime, "save_checkpoint"),
                    mock.patch.object(
                        training_runtime,
                        "save_rolling_checkpoint",
                        return_value=output_dir / "checkpoint_000001.pt",
                    ),
                ):
                    module._run_pretraining(
                        args,
                        None,
                        trainer,
                        None,
                        None,
                        replay_buffer,
                        object(),
                        np.random.default_rng(1),
                        torch.Generator().manual_seed(2),
                        torch.Generator().manual_seed(3),
                        output_dir / "metrics.jsonl",
                        torch.device("cpu"),
                    )

                self.assertEqual(trainer.train_epoch.call_count, args.epochs)
                trainer.train_epoch.assert_called_with(
                    replay_buffer.batches,
                    args.batch_size,
                    mock.ANY,
                    on_update=mock.ANY,
                )

    def test_formal_model_restores_complete_pretrained_world_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_best.pt"
            source_config = trans_wm_le_train.WorldModelConfig(
                observation_shape=(3, 32, 32),
                action_shape=(1,),
                cnn_channels=(4,),
            )
            target_config = trans_wm_le_train._model_config_from_checkpoint(
                asdict(source_config),
                source_config.observation_shape,
                source_config.action_shape,
            )
            pretraining_config = trans_wm_le_train.TrainingConfig(value_weight=0.0)
            source_model = trans_wm_le_train.WorldModel(source_config)
            source_trainer = trans_wm_le_train.WorldModelTrainer(
                source_model, pretraining_config
            )
            replay_buffer = RolloutReplayBuffer.__new__(RolloutReplayBuffer)
            replay_buffer._rng = np.random.default_rng(2)
            trans_wm_le_train._save_checkpoint(
                path,
                source_model,
                source_trainer,
                source_config,
                pretraining_config,
                np.random.default_rng(1),
                replay_buffer,
                torch.Generator().manual_seed(3),
                torch.Generator().manual_seed(4),
                1,
                -float("inf"),
                0,
                phase="pretrain",
            )

            target_model = trans_wm_le_train.WorldModel(target_config)
            trans_wm_le_train._load_pretrained_checkpoint(
                path,
                target_model,
                target_config,
                torch.device("cpu"),
            )

            torch.testing.assert_close(
                next(target_model.encoder.parameters()),
                next(source_model.encoder.parameters()),
            )


if __name__ == "__main__":
    unittest.main()
