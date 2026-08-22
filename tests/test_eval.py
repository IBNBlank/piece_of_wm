"""Tests for online world-model action scoring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

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
        self.particles = torch.linspace(-2.0, 2.0, 100).reshape(
            1, 100, 1, 1
        ).expand(2, -1, -1, -1)

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
            ParticlePolicy(horizon=3),
            particle_updates=2,
            particle_sigma=0.1,
            generator=torch.Generator().manual_seed(4),
        )

        self.assertEqual(action.shape, (2, 1))
        self.assertTrue(torch.all(action >= -2.0))
        self.assertTrue(torch.all(action <= 2.0))
        self.assertEqual(predicted_reward.shape, (2,))
        self.assertEqual(predicted_value.shape, (2,))

    def test_particle_horizon_scores_action_sequences(self) -> None:
        class Reward:
            def __call__(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                return action[:, :1]

        class Value:
            def mean(self, latent: torch.Tensor) -> torch.Tensor:
                return torch.zeros((len(latent), 1), dtype=latent.dtype)

        model = SimpleNamespace(
            config=SimpleNamespace(gamma=0.5),
            ema_heads=SimpleNamespace(reward=Reward(), value_head=Value()),
            predict_next_ema=lambda latent, action: latent,
        )
        particles = torch.tensor([[[[1.0], [2.0], [3.0]], [[-1.0], [4.0], [2.0]]]])

        scores, first_rewards, _ = score_particles(
            "trans_wm_le",
            model,
            torch.zeros(1, 2),
            torch.zeros(1, 9, 1),
            torch.zeros(1, 9, dtype=torch.bool),
            particles,
        )

        torch.testing.assert_close(scores, torch.tensor([[2.75, 1.5]]))
        torch.testing.assert_close(first_rewards, torch.tensor([[1.0, -1.0]]))


if __name__ == "__main__":
    unittest.main()
