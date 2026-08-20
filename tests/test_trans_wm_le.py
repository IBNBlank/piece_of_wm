"""Tests for the action-conditioned latent JEPA world model."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from trans_wm_le import (
    ACTION_HISTORY_LEN,
    OBS_HISTORY_LEN,
    TrainingConfig,
    WorldModel,
    WorldModelConfig,
    WorldModelTrainer,
    append_history,
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
        cnn_channels=(8, 16),
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

    def test_dynamics_is_mlp_of_latent_and_current_action(self) -> None:
        latent = torch.randn(4, 64)
        action = torch.randn(4, 2)

        next_latent = self.model.predict_next_online(latent, action)

        self.assertEqual(next_latent.shape, (4, 64))
        self.assertEqual(self.model.dynamics.mlp[0].in_features, 66)
        self.assertFalse(hasattr(self.model.dynamics, "transformer"))

    def test_decoder_and_observation_head_are_absent(self) -> None:
        heads = self.model.predict_heads(torch.randn(3, 64))

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
        self.assertIsNotNone(model.dynamics.mlp[-1].weight.grad)
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
