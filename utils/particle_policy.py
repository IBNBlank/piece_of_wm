"""Particle resampling primitives for Pendulum-v1 action selection."""

from __future__ import annotations

import torch


PENDULUM_ACTION_DIM = 1
PENDULUM_ACTION_LOW = -2.0
PENDULUM_ACTION_HIGH = 2.0
NUM_ACTION_PARTICLES = 100
DEFAULT_PARTICLE_SIGMA = 0.1


class ParticlePolicy:
    """Generates and updates batches of one-dimensional Pendulum action particles.

    The policy has no world-model dependency. Callers evaluate particles with their
    own reward and value models, then pass those scores to :meth:`update_particles`.
    """

    def __init__(self, num_particles: int = NUM_ACTION_PARTICLES, horizon: int = 1) -> None:
        if isinstance(num_particles, bool) or not isinstance(num_particles, int):
            raise TypeError("num_particles must be an integer.")
        if num_particles <= 0:
            raise ValueError("num_particles must be positive.")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer.")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.num_particles = num_particles
        self.horizon = horizon

    def init_particles(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Returns uniformly sampled particles for each batch element."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("Particle dtype must be floating point.")
        return PENDULUM_ACTION_LOW + (PENDULUM_ACTION_HIGH - PENDULUM_ACTION_LOW) * torch.rand(
            (batch_size, self.num_particles, self.horizon, PENDULUM_ACTION_DIM),
            dtype=dtype,
            device=device,
            generator=generator,
        )

    def update_particles(
        self,
        particles: torch.Tensor,
        scores: torch.Tensor,
        *,
        sigma: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Resamples particles by score and perturbs each result with Gaussian noise.

        ``scores`` has shape ``(batch, num_particles)`` and is converted to resampling weights
        with ``softmax``. ``sigma`` is the Gaussian standard deviation; omitting it
        uses ``0.1``.
        """
        self._validate_particles(particles)
        self._validate_scores(scores, particles)
        sigma = DEFAULT_PARTICLE_SIGMA if sigma is None else sigma
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")

        parent_indices = torch.multinomial(
            torch.softmax(scores, dim=1),
            self.num_particles,
            replacement=True,
            generator=generator,
        )
        resampled = torch.gather(
            particles,
            dim=1,
            index=parent_indices[..., None, None].expand(
                -1, -1, self.horizon, PENDULUM_ACTION_DIM
            ),
        )
        if sigma == 0.0:
            return resampled
        noise = torch.randn(
            resampled.shape,
            dtype=resampled.dtype,
            device=resampled.device,
            generator=generator,
        )
        return (resampled + sigma * noise).clamp(PENDULUM_ACTION_LOW, PENDULUM_ACTION_HIGH)

    def _validate_particles(self, particles: torch.Tensor) -> None:
        if not isinstance(particles, torch.Tensor):
            raise TypeError("particles must be a torch.Tensor.")
        if particles.ndim != 4 or particles.shape[1:] != (
            self.num_particles,
            self.horizon,
            PENDULUM_ACTION_DIM,
        ):
            raise ValueError(
                f"particles must have shape (batch, {self.num_particles}, "
                f"{self.horizon}, 1)."
            )
        if not torch.is_floating_point(particles):
            raise TypeError("particles must use a floating-point dtype.")

    @staticmethod
    def _validate_scores(scores: torch.Tensor, particles: torch.Tensor) -> None:
        if not isinstance(scores, torch.Tensor):
            raise TypeError("scores must be a torch.Tensor.")
        if scores.shape != particles.shape[:2]:
            raise ValueError("scores must have shape (batch, num_particles).")
        if scores.device != particles.device:
            raise ValueError("scores and particles must be on the same device.")
        if not torch.is_floating_point(scores):
            raise TypeError("scores must use a floating-point dtype.")
        if not torch.isfinite(scores).all():
            raise ValueError("scores must be finite for particle resampling.")
