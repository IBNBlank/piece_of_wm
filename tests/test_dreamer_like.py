"""Tests for the variational latent world model, dynamics, heads, and losses."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

import dreamer_like.training as training_module

from dreamer_like import (
    ACTION_HISTORY_LEN,
    OBS_HISTORY_LEN,
    TrainingConfig,
    WorldModel,
    WorldModelConfig,
    WorldModelTrainer,
    append_history,
    sample_transition_batch,
    tensor_episode_batch,
    world_model_loss,
    vae_kl_loss,
)
from utils.replay_buffer import EpisodeBatch


def _config() -> WorldModelConfig:
    return WorldModelConfig(
        observation_shape=(3, 32, 32),
        action_shape=(2,),
        observation_dim=12,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        cnn_channels=(8, 16),
        dropout=0.0,
    )


def _batch() -> EpisodeBatch:
    lengths = np.asarray([2, 4], dtype=np.int64)
    obs = np.zeros((2, 5, 3), dtype=np.float32)
    images = np.zeros((2, 5, 32, 32, 3), dtype=np.uint8)
    action = np.zeros((2, 4, 2), dtype=np.float32)
    reward = np.zeros((2, 4), dtype=np.float32)
    terminated = np.zeros((2, 4), dtype=bool)
    truncated = np.zeros((2, 4), dtype=bool)
    for index, length in enumerate(lengths):
        images[index, : length + 1] = np.arange(length + 1)[:, None, None, None] * 20
        action[index, :length] = 0.1 + index
        reward[index, :length] = np.arange(length) + index
        terminated[index, length - 1] = True
    return EpisodeBatch(obs, action, reward, terminated, truncated, lengths, images)


class WorldModelShapeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = WorldModel(_config()).eval()

    def test_encoder_and_action_history_produce_128d_z(self) -> None:
        images = torch.randn(4, OBS_HISTORY_LEN, 3, 32, 32)
        mask = torch.ones(4, OBS_HISTORY_LEN, dtype=torch.bool)
        actions = torch.randn(4, ACTION_HISTORY_LEN, 2)
        action_mask = torch.ones(4, ACTION_HISTORY_LEN, dtype=torch.bool)

        observation = self.model.encode_observation_online(images, mask)
        z = self.model.encode(images, mask, actions, action_mask)

        self.assertEqual(observation.shape, (4, 12))
        self.assertEqual(z.shape, (4, 128))
        first_conv = self.model.encoder.cnn[0]
        self.assertEqual(first_conv.in_channels, OBS_HISTORY_LEN * 3)

    def test_encoder_ignores_masked_image_values(self) -> None:
        images = torch.randn(2, OBS_HISTORY_LEN, 3, 32, 32)
        mask = torch.tensor([[False, True, True], [True] * OBS_HISTORY_LEN])
        actions = torch.randn(2, ACTION_HISTORY_LEN, 2)
        changed = images.clone()
        changed[~mask] = 10000.0

        torch.testing.assert_close(
            self.model.encode(images, mask, actions),
            self.model.encode(changed, mask, actions),
        )

    def test_dynamics_uses_latent_and_current_action_tokens(self) -> None:
        z = torch.randn(4, 128)
        action = torch.randn(4, 2)
        captured: list[torch.Size] = []

        handle = self.model.dynamics.transformer.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].shape)
        )
        try:
            next_z = self.model.predict_next_online(z, action)
        finally:
            handle.remove()

        self.assertEqual(next_z.shape, (4, 128))
        self.assertEqual(captured, [torch.Size((4, 2, 16))])

    def test_action_history_is_one_flattened_ah_tensor(self) -> None:
        actions = torch.randn(2, ACTION_HISTORY_LEN, 2)
        mask = torch.tensor([[False, True], [True] * ACTION_HISTORY_LEN])

        ah = self.model.action_history_tensor(actions, mask)

        self.assertEqual(ah.shape, (2, ACTION_HISTORY_LEN * 2))
        self.assertTrue(torch.equal(ah[0, :2], torch.zeros(2)))
        torch.testing.assert_close(ah[0, -2:], actions[0, -1].flatten())
        images = torch.randn(2, OBS_HISTORY_LEN, 3, 32, 32)
        image_mask = torch.ones(2, OBS_HISTORY_LEN, dtype=torch.bool)
        torch.testing.assert_close(
            self.model.encode_online(images, image_mask, actions, mask),
            self.model.encode_online(images, image_mask, ah),
        )

    def test_heads_read_one_z_and_reconstruct_image_history(self) -> None:
        output = self.model.predict_heads(torch.randn(3, 128), torch.randn(3, 2))

        self.assertEqual(output.observation.shape, (3, OBS_HISTORY_LEN, 3, 32, 32))
        self.assertEqual(output.reward.shape, (3, 1))
        self.assertFalse(hasattr(output, "value"))
        self.assertFalse(hasattr(self.model.heads, "value_head"))

    def test_action_score_is_predicted_reward(self) -> None:
        z = torch.randn(2, 128)
        action = torch.randn(2, 2)

        result = self.model.evaluate_action(z, action)

        torch.testing.assert_close(result.score, result.heads.reward)
        self.assertEqual(result.score.shape, (2, 1))

    def test_rollout_maintains_only_action_history(self) -> None:
        z = torch.randn(2, 128)
        action_history = torch.randn(2, ACTION_HISTORY_LEN, 2)
        action_mask = torch.tensor([[False, True], [True] * ACTION_HISTORY_LEN])
        actions = torch.randn(2, 3, 2)

        output = self.model.rollout(z, action_history, actions, action_mask)

        self.assertEqual(output.latents.shape, (2, 3, 128))
        self.assertEqual(
            output.observations.shape, (2, 3, OBS_HISTORY_LEN, 3, 32, 32)
        )
        self.assertEqual(output.rewards.shape, (2, 3, 1))
        self.assertFalse(hasattr(output, "values"))
        self.assertEqual(output.scores.shape, (2, 3, 1))
        torch.testing.assert_close(output.scores, output.rewards)
        self.assertEqual(output.final_z.shape, (2, 128))
        self.assertEqual(output.final_action_history.shape, (2, ACTION_HISTORY_LEN, 2))
        torch.testing.assert_close(output.final_action_history, actions[:, -2:])


class WorldModelTrainingTest(unittest.TestCase):
    def test_validation_epoch_visits_every_transition_once(self) -> None:
        trainer = WorldModelTrainer(WorldModel(_config()), TrainingConfig())
        batch = _batch()
        with mock.patch.object(
            training_module,
            "transition_batch_from_indices",
            wraps=training_module.transition_batch_from_indices,
        ) as build_batch:
            metrics = trainer.evaluate_transitions(
                batch, batch_size=4, rng=np.random.default_rng(7)
            )

        visited = np.concatenate([call.args[2] for call in build_batch.call_args_list])
        np.testing.assert_array_equal(np.sort(visited), np.arange(batch.num_transitions))
        self.assertEqual(build_batch.call_count, 2)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_epoch_visits_every_transition_once_and_keeps_partial_batch(self) -> None:
        trainer = WorldModelTrainer(WorldModel(_config()), TrainingConfig())
        batches = [_batch(), _batch()]
        on_update = mock.Mock()
        with (
            mock.patch.object(
                training_module,
                "transition_batch_from_indices",
                wraps=training_module.transition_batch_from_indices,
            ) as build_batch,
            mock.patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step,
        ):
            metrics = trainer.train_epoch(
                batches,
                batch_size=5,
                rng=np.random.default_rng(7),
                on_update=on_update,
            )

        for batch in batches:
            visited = np.concatenate(
                [call.args[2] for call in build_batch.call_args_list if call.args[0] is batch]
            )
            np.testing.assert_array_equal(np.sort(visited), np.arange(6))
        self.assertEqual(step.call_count, 3)
        self.assertEqual(on_update.call_count, 3)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_value_head_is_absent(self) -> None:
        model = WorldModel(_config())
        self.assertFalse(hasattr(model.heads, "value"))
        self.assertFalse(hasattr(model.heads, "value_head"))

    def test_reward_is_conditioned_on_current_latent_and_action(self) -> None:
        model = WorldModel(_config())
        z = torch.randn(2, model.config.latent_dim)
        low = model.heads.reward(z, -torch.ones(2, model.config.action_dim))
        high = model.heads.reward(z, torch.ones(2, model.config.action_dim))
        self.assertFalse(torch.equal(low, high))
        self.assertEqual(
            model.heads.reward_head[1].in_features,
            model.config.latent_dim + model.config.action_dim,
        )

    def test_transition_sampler_preserves_frame_history_alignment(self) -> None:
        model = WorldModel(_config())
        sampled = sample_transition_batch(
            _batch(), model, batch_size=6, rng=np.random.default_rng(4), planning_horizon=10
        )

        self.assertEqual(sampled.current_observations.shape, (6, 3, 3, 32, 32))
        self.assertEqual(sampled.action_history.shape, (6, 2, 2))
        torch.testing.assert_close(
            sampled.current_observations[:, 1:], sampled.next_observations[:, :-1]
        )
        self.assertTrue(
            torch.equal(sampled.current_obs_valid[:, 1:], sampled.next_obs_valid[:, :-1])
        )

    def test_transition_sampler_builds_masked_multistep_targets(self) -> None:
        model = WorldModel(_config())
        sampled = training_module.transition_batch_from_indices(
            _batch(), model, np.asarray([0, 2]), planning_horizon=3
        )

        self.assertEqual(sampled.actions.shape, (2, 3, 2))
        self.assertEqual(sampled.target_observations.shape, (2, 3, 3, 3, 32, 32))
        torch.testing.assert_close(
            sampled.next_returns.squeeze(-1),
            torch.tensor([[1.0, 0.0, 0.0], [9.0, 7.0, 4.0]]),
        )
        torch.testing.assert_close(
            sampled.transition_valid,
            torch.tensor([[True, True, False], [True, True, True]]),
        )

    def test_transition_loss_recursively_predicts_each_horizon_step(self) -> None:
        model = WorldModel(_config())
        config = TrainingConfig(planning_horizon=3)
        sampled = training_module.transition_batch_from_indices(
            _batch(), model, np.asarray([2]), planning_horizon=3
        )

        with mock.patch.object(
            model, "predict_next_online", wraps=model.predict_next_online
        ) as predict_next:
            training_module.transition_world_model_loss(model, sampled, config)

        self.assertEqual(predict_next.call_count, 3)

    def test_multistep_vae_uses_each_aligned_state_posterior(self) -> None:
        model = WorldModel(_config())
        sampled = training_module.transition_batch_from_indices(
            _batch(), model, np.asarray([2]), planning_horizon=3
        )
        with mock.patch.object(
            model, "posterior_online", wraps=model.posterior_online
        ) as posterior:
            training_module.transition_world_model_loss(
                model, sampled, TrainingConfig(planning_horizon=3)
            )

        self.assertEqual(posterior.call_count, 4)
        next_history, next_valid = append_history(
            sampled.action_history, sampled.action_valid, sampled.action
        )
        torch.testing.assert_close(posterior.call_args_list[1].args[2], next_history)
        torch.testing.assert_close(posterior.call_args_list[1].args[3], next_valid)

    def test_vae_kl_is_zero_for_standard_normal_posterior(self) -> None:
        mean = torch.zeros(2, 8)
        log_variance = torch.zeros(2, 8)

        loss = vae_kl_loss(mean, log_variance)

        torch.testing.assert_close(loss, torch.zeros(2))

    def test_image_batch_conversion_and_training_losses(self) -> None:
        torch.manual_seed(8)
        model = WorldModel(_config())
        config = TrainingConfig(grad_clip_norm=10.0)
        batch = _batch()
        tensor_batch = tensor_episode_batch(batch, model)

        self.assertEqual(tensor_batch.observations.shape, (2, 5, 3, 32, 32))
        self.assertGreaterEqual(tensor_batch.observations.min().item(), 0.0)
        self.assertLessEqual(tensor_batch.observations.max().item(), 1.0)
        losses = world_model_loss(model, tensor_batch, config)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertIsNotNone(model.encoder.to_observation[1].weight.grad)
        self.assertIsNotNone(model.latent_encoder.to_statistics[0].weight.grad)
        self.assertIsNotNone(model.dynamics.output[1].weight.grad)
        model.zero_grad(set_to_none=True)

        metrics = WorldModelTrainer(model, config).train_batch(batch)

        self.assertEqual(
            set(metrics),
            {"total", "observation", "reward", "vae_reconstruction", "vae_kl"},
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

        sampled_metrics = WorldModelTrainer(model, config).train_transitions(
            batch, batch_size=3, rng=np.random.default_rng(5)
        )
        self.assertTrue(all(np.isfinite(value) for value in sampled_metrics.values()))

    def test_policy_facing_apis_use_frozen_ema_modules_without_grad(self) -> None:
        model = WorldModel(_config())
        image = torch.randn(2, OBS_HISTORY_LEN, 3, 32, 32)
        image_mask = torch.ones(2, OBS_HISTORY_LEN, dtype=torch.bool)
        action_history = torch.randn(2, ACTION_HISTORY_LEN, 2)
        z = model.encode_ema(image, image_mask, action_history)
        action = torch.randn(2, 2)

        evaluation = model.evaluate_action(z, action)

        self.assertFalse(evaluation.next_z.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.ema_encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.ema_latent_encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.ema_dynamics.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.ema_heads.parameters()))

    def test_ema_update_covers_encoder_dynamics_and_heads(self) -> None:
        model = WorldModel(_config())
        with torch.no_grad():
            next(model.encoder.parameters()).add_(1.0)
            next(model.latent_encoder.parameters()).add_(1.0)
            next(model.dynamics.parameters()).add_(1.0)
            next(model.heads.parameters()).add_(1.0)

        model.update_target(ema=0.0)

        for ema_module, online_module in (
            (model.ema_encoder, model.encoder),
            (model.ema_latent_encoder, model.latent_encoder),
            (model.ema_dynamics, model.dynamics),
            (model.ema_heads, model.heads),
        ):
            for ema_parameter, online_parameter in zip(
                ema_module.parameters(), online_module.parameters(), strict=True
            ):
                torch.testing.assert_close(ema_parameter, online_parameter)


if __name__ == "__main__":
    unittest.main()
