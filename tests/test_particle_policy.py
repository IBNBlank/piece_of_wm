"""Tests for Pendulum action-particle generation and updates."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from utils.particle_policy import (
    INITIAL_PARTICLE_SIGMA,
    NUM_ACTION_PARTICLES,
    PENDULUM_ACTION_MIDPOINT,
    ParticlePolicy,
)


class ParticlePolicyTest(unittest.TestCase):
    def test_init_particles_is_gaussian_and_bounded(self) -> None:
        particles = ParticlePolicy(num_particles=32, horizon=3).init_particles(
            3, generator=torch.Generator().manual_seed(4)
        )

        self.assertEqual(particles.shape, (3, 32, 3, 1))
        self.assertTrue(torch.all(particles >= -2.0))
        self.assertTrue(torch.all(particles <= 2.0))

    def test_gaussian_init_uses_action_midpoint_and_quarter_range_sigma(self) -> None:
        policy = ParticlePolicy(num_particles=2, horizon=2)
        noise = torch.tensor([[[[-3.0], [-1.0]], [[1.0], [3.0]]]])

        with mock.patch("torch.randn", return_value=noise) as randn:
            particles = policy.init_particles(1)

        self.assertEqual(PENDULUM_ACTION_MIDPOINT, 0.0)
        self.assertEqual(INITIAL_PARTICLE_SIGMA, 1.0)
        torch.testing.assert_close(
            particles, torch.tensor([[[[-2.0], [-1.0]], [[1.0], [2.0]]]])
        )
        self.assertEqual(randn.call_args.args[0], (1, 2, 2, 1))

    def test_default_particle_count_is_one_thousand(self) -> None:
        self.assertEqual(NUM_ACTION_PARTICLES, 1000)
        self.assertEqual(ParticlePolicy().num_particles, 1000)
        self.assertEqual(ParticlePolicy().horizon, 8)

    def test_update_resamples_high_score_particles_without_noise(self) -> None:
        policy = ParticlePolicy(num_particles=32, horizon=3)
        particles = torch.linspace(-2.0, 2.0, 96).reshape(1, 32, 3, 1)
        scores = torch.full((1, 32), -100.0)
        scores[:, 24] = 100.0

        updated = policy.update_particles(
            particles,
            scores,
            sigma=0.0,
            generator=torch.Generator().manual_seed(5),
        )

        self.assertTrue(torch.equal(updated, particles[:, 24:25].expand_as(updated)))

    def test_temperature_scales_resampling_logits(self) -> None:
        policy = ParticlePolicy(num_particles=2, horizon=1)
        particles = torch.tensor([[[[-1.0]], [[1.0]]]])
        scores = torch.tensor([[0.0, 2.0]])
        parent_indices = torch.tensor([[0, 1]])

        with mock.patch("torch.multinomial", return_value=parent_indices) as multinomial:
            policy.update_particles(particles, scores, sigma=0.0, temperature=0.5)

        torch.testing.assert_close(
            multinomial.call_args.args[0], torch.softmax(scores / 0.5, dim=1)
        )

    def test_temperature_must_be_positive(self) -> None:
        policy = ParticlePolicy(num_particles=2, horizon=1)
        particles = torch.zeros(1, 2, 1, 1)
        scores = torch.zeros(1, 2)

        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            policy.update_particles(particles, scores, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
