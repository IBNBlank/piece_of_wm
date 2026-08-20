"""Tests for Pendulum action-particle generation and updates."""

from __future__ import annotations

import unittest

import torch

from utils.particle_policy import ParticlePolicy


class ParticlePolicyTest(unittest.TestCase):
    def test_init_particles_is_uniformly_bounded(self) -> None:
        particles = ParticlePolicy().init_particles(3, generator=torch.Generator().manual_seed(4))

        self.assertEqual(particles.shape, (3, 100, 1))
        self.assertTrue(torch.all(particles >= -2.0))
        self.assertTrue(torch.all(particles <= 2.0))

    def test_update_resamples_high_score_particles_without_noise(self) -> None:
        policy = ParticlePolicy()
        particles = torch.linspace(-2.0, 2.0, 100).reshape(1, 100, 1)
        scores = torch.full((1, 100), -100.0)
        scores[:, 75] = 100.0

        updated = policy.update_particles(
            particles,
            scores,
            sigma=0.0,
            generator=torch.Generator().manual_seed(5),
        )

        self.assertTrue(torch.equal(updated, particles[:, 75:76].expand_as(updated)))


if __name__ == "__main__":
    unittest.main()
