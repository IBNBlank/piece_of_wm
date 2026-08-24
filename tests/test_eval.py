"""Tests for online world-model action scoring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval import _particle_sigma, score_particles, select_particle_action
from trans_wm import WorldModel as TransWorldModel
from trans_wm import WorldModelConfig as TransWorldModelConfig
from trans_wm_le import WorldModel as LatentWorldModel
from trans_wm_le import WorldModelConfig as LatentWorldModelConfig
from utils.particle_policy import ParticlePolicy


class OnlineEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.particles = torch.linspace(-2.0, 2.0, 100).reshape(
            1, 100, 1, 1
        ).expand(2, -1, -1, -1)

    def test_trans_wm_scores_particle_actions_with_predicted_reward(self) -> None:
        model = TransWorldModel(
            TransWorldModelConfig(
                observation_shape=(3, 32, 32),
                action_shape=(1,),
                model_dim=8,
                num_layers=1,
                num_heads=1,
                feedforward_dim=16,
                cnn_channels=(4,),
            )
        ).eval()
        latent = torch.randn(2, 128)

        scores, rewards = score_particles(
            model,
            latent,
            self.particles,
        )

        self.assertEqual(scores.shape, (2, 100))
        self.assertEqual(rewards.shape, (2, 100))
        torch.testing.assert_close(scores, rewards)

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
        latent = torch.randn(2, 128)

        action, predicted_reward = select_particle_action(
            model,
            latent,
            ParticlePolicy(horizon=3),
            particle_updates=2,
            particle_sigma=0.1,
            particle_temperature=1.0,
            generator=torch.Generator().manual_seed(4),
        )

        self.assertEqual(action.shape, (2, 1))
        self.assertTrue(torch.all(action >= -2.0))
        self.assertTrue(torch.all(action <= 2.0))
        self.assertEqual(predicted_reward.shape, (2,))

    def test_particle_sigma_decreases_to_inverse_horizon_floor(self) -> None:
        self.assertAlmostEqual(_particle_sigma(0.1, 25, 0), 0.1)
        self.assertAlmostEqual(_particle_sigma(0.1, 25, 1), 0.06)
        self.assertAlmostEqual(_particle_sigma(0.1, 25, 2), 0.04)
        self.assertAlmostEqual(_particle_sigma(0.1, 25, 9), 0.04)

    def test_particle_horizon_scores_action_sequences(self) -> None:
        class Reward:
            def __call__(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                return action[:, :1]

        model = SimpleNamespace(
            ema_heads=SimpleNamespace(reward=Reward()),
            predict_next_ema=lambda latent, action: latent,
        )
        particles = torch.tensor([[[[1.0], [2.0], [3.0]], [[-1.0], [4.0], [2.0]]]])

        scores, first_rewards = score_particles(
            model,
            torch.zeros(1, 2),
            particles,
        )

        torch.testing.assert_close(scores, torch.tensor([[6.0, 5.0]]))
        torch.testing.assert_close(first_rewards, torch.tensor([[1.0, -1.0]]))


if __name__ == "__main__":
    unittest.main()
