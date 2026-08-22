"""Tests for the action-conditioned latent JEPA world model."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

import trans_wm_le.training as training_module

from trans_wm_le import (
    ACTION_HISTORY_LEN,
    OBS_HISTORY_LEN,
    TrainingConfig,
    WorldModel,
    WorldModelConfig,
    WorldModelTrainer,
    append_history,
    discounted_returns,
    sample_transition_batch,
    sigreg_loss,
    tensor_episode_batch,
    transition_world_model_loss,
    world_model_loss,
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


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        grad_clip_norm=10.0,
        sigreg_projections=8,
        sigreg_frequencies=4,
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

    def test_cnn_output_is_observation_and_obs_plus_ah_produces_64d_latent(self) -> None:
        images = torch.randn(4, OBS_HISTORY_LEN, 3, 32, 32)
        obs_mask = torch.ones(4, OBS_HISTORY_LEN, dtype=torch.bool)
        action_history = torch.randn(4, ACTION_HISTORY_LEN, 2)
        action_mask = torch.ones(4, ACTION_HISTORY_LEN, dtype=torch.bool)

        observation = self.model.encode_observation_online(images, obs_mask)
        latent = self.model.encode(images, obs_mask, action_history, action_mask)

        self.assertEqual(observation.shape, (4, 12))
        self.assertEqual(latent.shape, (4, 64))
        changed = action_history.clone()
        changed[:, -1] += 1.0
        self.assertFalse(
            torch.equal(latent, self.model.encode(images, obs_mask, changed, action_mask))
        )

    def test_dynamics_uses_latent_and_current_action_tokens(self) -> None:
        latent = torch.randn(4, 64)
        action = torch.randn(4, 2)
        captured: list[torch.Size] = []

        handle = self.model.dynamics.transformer.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].shape)
        )
        try:
            next_latent = self.model.predict_next_online(latent, action)
        finally:
            handle.remove()

        self.assertEqual(next_latent.shape, (4, 64))
        self.assertEqual(captured, [torch.Size((4, 2, 16))])

    def test_decoder_and_observation_head_are_absent(self) -> None:
        heads = self.model.predict_heads(torch.randn(3, 64), torch.randn(3, 2))

        self.assertEqual(heads.reward.shape, (3, 1))
        self.assertEqual(heads.value.shape, (3, 1))
        self.assertFalse(hasattr(heads, "observation"))
        self.assertFalse(hasattr(self.model.heads, "observation_head"))

    def test_rollout_updates_history_but_dynamics_only_receives_action(self) -> None:
        latent = torch.randn(2, 64)
        action_history = torch.randn(2, ACTION_HISTORY_LEN, 2)
        action_mask = torch.tensor([[False] * 7 + [True] * 2, [True] * 9])
        actions = torch.randn(2, 3, 2)

        output = self.model.rollout(latent, action_history, actions, action_mask)

        self.assertEqual(output.latents.shape, (2, 3, 64))
        self.assertEqual(output.rewards.shape, (2, 3, 1))
        self.assertEqual(output.values.shape, (2, 3, 1))
        self.assertEqual(output.final_action_history.shape, (2, 9, 2))
        torch.testing.assert_close(output.final_action_history[:, -3:], actions)


class WorldModelTrainingTest(unittest.TestCase):
    def test_validation_epoch_visits_every_transition_once(self) -> None:
        trainer = WorldModelTrainer(WorldModel(_config()), _training_config())
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
        trainer = WorldModelTrainer(WorldModel(_config()), _training_config())
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

    def test_discounted_returns_are_aligned_with_current_state(self) -> None:
        returns = discounted_returns(torch.tensor([1.0, 2.0, 3.0]), gamma=0.5)
        torch.testing.assert_close(returns, torch.tensor([2.75, 3.5, 3.0]))

    def test_replay_update_does_not_train_value_head(self) -> None:
        model = WorldModel(_config())
        trainer = WorldModelTrainer(model, _training_config())
        before = [parameter.detach().clone() for parameter in model.heads.value_head.parameters()]
        trainer.train_transitions(_batch(), batch_size=3, rng=np.random.default_rng(1))
        for previous, current in zip(before, model.heads.value_head.parameters(), strict=True):
            torch.testing.assert_close(previous, current, rtol=0.0, atol=0.0)

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
        sampled = sample_transition_batch(
            _batch(), WorldModel(_config()), batch_size=6, rng=np.random.default_rng(4)
        )

        torch.testing.assert_close(
            sampled.current_observations[:, 1:], sampled.next_observations[:, :-1]
        )
        self.assertTrue(
            torch.equal(sampled.current_obs_valid[:, 1:], sampled.next_obs_valid[:, :-1])
        )

    def test_jepa_target_uses_action_history_with_current_action_appended(self) -> None:
        model = WorldModel(_config())
        sampled = sample_transition_batch(
            _batch(), model, batch_size=3, rng=np.random.default_rng(2)
        )
        captured: list[torch.Tensor] = []
        handle = model.ema_latent_encoder.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[1].detach().clone())
        )
        try:
            transition_world_model_loss(model, sampled, _training_config())
        finally:
            handle.remove()
        next_history, next_mask = append_history(
            sampled.action_history, sampled.action_valid, sampled.action
        )

        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(
            captured[0], model.action_history_tensor(next_history, next_mask)
        )

    def test_sigreg_is_finite_and_backpropagates(self) -> None:
        latents = torch.randn(16, 64, requires_grad=True)

        loss = sigreg_loss(latents, num_projections=8, num_frequencies=4)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.isfinite(latents.grad).all())

    def test_full_and_sampled_training_have_new_losses_and_gradients(self) -> None:
        torch.manual_seed(8)
        model = WorldModel(_config())
        config = _training_config()
        batch = _batch()
        losses = world_model_loss(model, tensor_episode_batch(batch, model), config)

        losses.total.backward()

        self.assertIsNotNone(model.encoder.to_observation[1].weight.grad)
        self.assertIsNotNone(model.latent_encoder.mlp[0].weight.grad)
        self.assertIsNotNone(model.dynamics.output[-1].weight.grad)
        model.zero_grad(set_to_none=True)
        trainer = WorldModelTrainer(model, config)
        full_metrics = trainer.train_batch(batch)
        sampled_metrics = trainer.train_transitions(
            batch, batch_size=3, rng=np.random.default_rng(5)
        )
        expected_metrics = {"total", "jepa", "sigreg", "reward", "value"}
        self.assertEqual(set(full_metrics), expected_metrics)
        self.assertEqual(set(sampled_metrics), expected_metrics)
        self.assertTrue(all(np.isfinite(value) for value in full_metrics.values()))
        self.assertTrue(all(np.isfinite(value) for value in sampled_metrics.values()))

    def test_policy_apis_and_all_ema_modules_are_frozen(self) -> None:
        model = WorldModel(_config())
        images = torch.randn(2, OBS_HISTORY_LEN, 3, 32, 32)
        image_mask = torch.ones(2, OBS_HISTORY_LEN, dtype=torch.bool)
        action_history = torch.randn(2, ACTION_HISTORY_LEN, 2)
        action_mask = torch.ones(2, ACTION_HISTORY_LEN, dtype=torch.bool)
        latent = model.encode(images, image_mask, action_history, action_mask)
        evaluation = model.evaluate_action(latent, torch.randn(2, 2))

        self.assertFalse(evaluation.next_z.requires_grad)
        for module in (
            model.ema_encoder,
            model.ema_latent_encoder,
            model.ema_dynamics,
            model.ema_heads,
        ):
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))


if __name__ == "__main__":
    unittest.main()
