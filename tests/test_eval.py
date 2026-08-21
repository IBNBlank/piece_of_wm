"""Tests for online world-model action scoring."""

from __future__ import annotations

import unittest

import torch

from eval import score_particles, select_particle_action
from trans_wm import WorldModel as TransWorldModel
from trans_wm import WorldModelConfig as TransWorldModelConfig
from trans_wm_le import WorldModel as LatentWorldModel
from trans_wm_le import WorldModelConfig as LatentWorldModelConfig
from utils.particle_policy import ParticlePolicy


class OnlineEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.action_history = torch.zeros(2, 9, 1)
        self.action_valid = torch.zeros(2, 9, dtype=torch.bool)
        self.particles = torch.linspace(-2.0, 2.0, 100).reshape(1, 100, 1).expand(2, -1, -1)

    def test_trans_wm_scores_particle_actions_with_ema_reward_and_value(self) -> None:
        model = TransWorldModel(
            TransWorldModelConfig(
                observation_shape=(3, 32, 32),
                action_shape=(1,),
                latent_dim=8,
                model_dim=8,
                num_layers=1,
                num_heads=1,
                feedforward_dim=16,
                cnn_channels=(4,),
            )
        ).eval()
        latent = torch.randn(2, 8)

        scores, rewards, values = score_particles(
            "trans_wm",
            model,
            latent,
            self.action_history,
            self.action_valid,
            self.particles,
        )

        self.assertEqual(scores.shape, (2, 100))
        self.assertEqual(rewards.shape, (2, 100))
        self.assertEqual(values.shape, (2, 100))
        torch.testing.assert_close(scores, rewards + model.config.gamma * values)

    def test_trans_wm_le_selects_a_bounded_particle_action(self) -> None:
        model = LatentWorldModel(
            LatentWorldModelConfig(
                observation_shape=(3, 32, 32),
                action_shape=(1,),
                observation_dim=8,
                model_dim=8,
                cnn_channels=(4,),
            )
        ).eval()
        latent = torch.randn(2, 64)

        action, predicted_reward, predicted_value = select_particle_action(
            "trans_wm_le",
            model,
            latent,
            self.action_history,
            self.action_valid,
            ParticlePolicy(),
            particle_updates=2,
            particle_sigma=0.1,
            planning_horizon=3,
            generator=torch.Generator().manual_seed(4),
        )

        self.assertEqual(action.shape, (2, 1))
        self.assertTrue(torch.all(action >= -2.0))
        self.assertTrue(torch.all(action <= 2.0))
        self.assertEqual(predicted_reward.shape, (2,))
        self.assertEqual(predicted_value.shape, (2,))


if __name__ == "__main__":
    unittest.main()
