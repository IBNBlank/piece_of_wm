"""Tests for lambda returns and pessimistic ensemble values."""

from __future__ import annotations

import unittest

import torch

from eval import _predict_ema_bootstrap_value, score_particles
from trans_wm import ACTION_HISTORY_LEN
from trans_wm import WorldModel as TransWorldModel
from trans_wm import WorldModelConfig as TransWorldModelConfig
from trans_wm_le import WorldModel as LatentWorldModel
from trans_wm_le import WorldModelConfig as LatentWorldModelConfig
from utils.value import EnsembleValueHead, lambda_returns


class ValueTest(unittest.TestCase):
    def test_lambda_returns_mix_world_model_bootstrap_and_multistep_return(self) -> None:
        rewards = torch.tensor([1.0, 2.0, 3.0])
        bootstrap_values = torch.tensor([10.0, 20.0, 0.0])

        returns = lambda_returns(rewards, bootstrap_values, gamma=0.5, lambda_=0.5)

        torch.testing.assert_close(returns, torch.tensor([5.4375, 7.75, 3.0]))
        torch.testing.assert_close(
            lambda_returns(rewards, bootstrap_values, gamma=0.5, lambda_=1.0),
            torch.tensor([2.75, 3.5, 3.0]),
        )

    def test_ensemble_exposes_minimum_and_mean_aggregates(self) -> None:
        head = EnsembleValueHead(input_dim=3, hidden_dim=4, num_critics=3)
        latent = torch.randn(5, 3)
        all_values = head.all_values(latent)

        torch.testing.assert_close(
            head.minimum(latent), all_values.min(dim=-1, keepdim=True).values
        )
        torch.testing.assert_close(
            head.mean(latent), all_values.mean(dim=-1, keepdim=True)
        )
        torch.testing.assert_close(head(latent), head.minimum(latent))
        all_values.square().mean().backward()
        for critic in head.critics:
            self.assertTrue(all(parameter.grad is not None for parameter in critic.parameters()))

    def test_minimum_loss_updates_only_the_worst_critic(self) -> None:
        head = EnsembleValueHead(input_dim=3, hidden_dim=4, num_critics=2)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.critics[0][-1].bias.fill_(-2.0)
            head.critics[1][-1].bias.fill_(1.0)

        head.minimum(torch.zeros(5, 3)).square().mean().backward()

        self.assertNotEqual(head.critics[0][-1].bias.grad.item(), 0.0)
        self.assertEqual(head.critics[1][-1].bias.grad.item(), 0.0)

    def test_world_model_bootstrap_uses_worst_ema_critic(self) -> None:
        cases = (
            (
                "trans_wm",
                TransWorldModel(
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
                ),
            ),
            (
                "trans_wm_le",
                LatentWorldModel(
                    LatentWorldModelConfig(
                        observation_shape=(3, 32, 32),
                        action_shape=(1,),
                        observation_dim=8,
                        model_dim=8,
                        num_layers=1,
                        num_heads=1,
                        feedforward_dim=16,
                        cnn_channels=(4,),
                    )
                ),
            ),
        )
        for model_name, model in cases:
            with self.subTest(model=model_name), torch.no_grad():
                for head in (model.heads.value_head, model.ema_heads.value_head):
                    for parameter in head.parameters():
                        parameter.zero_()
                model.heads.value_head.critics[0][-1].bias.fill_(-9.0)
                model.heads.value_head.critics[1][-1].bias.fill_(-8.0)
                model.ema_heads.value_head.critics[0][-1].bias.fill_(4.0)
                model.ema_heads.value_head.critics[1][-1].bias.fill_(-3.0)
                latent = torch.randn(1, model.config.latent_dim)
                action = torch.zeros(1, 1)
                action_history = torch.zeros(1, ACTION_HISTORY_LEN, 1)
                action_valid = torch.zeros(1, ACTION_HISTORY_LEN, dtype=torch.bool)

                value = _predict_ema_bootstrap_value(
                    model_name,
                    model,
                    latent,
                    action_history,
                    action_valid,
                    action,
                )

                torch.testing.assert_close(value, torch.tensor([[-3.0]]))

                _, _, planning_value = score_particles(
                    model_name,
                    model,
                    latent,
                    action_history,
                    action_valid,
                    torch.zeros(1, 1, 1, 1),
                )
                torch.testing.assert_close(planning_value, torch.tensor([[0.5]]))


if __name__ == "__main__":
    unittest.main()
