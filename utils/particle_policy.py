"""Particle resampling primitives for bounded continuous Gym actions."""

from __future__ import annotations

import torch


FETCH_ACTION_DIM = 4
FETCH_ACTION_LOW = -1.0
FETCH_ACTION_HIGH = 1.0
NUM_ACTION_PARTICLES = 1000
DEFAULT_PARTICLE_SIGMA = 0.1
DEFAULT_PARTICLE_TEMPERATURE = 2.0
DEFAULT_PLANNING_HORIZON = 20


class ParticlePolicy:
    """Generates and updates batches of Fetch pick-and-place action particles.

    The policy has no world-model dependency. Callers evaluate particles with their
    own reward model, then pass those scores to :meth:`update_particles`.
    """

    def __init__(
        self,
        num_particles: int = NUM_ACTION_PARTICLES,
        horizon: int = DEFAULT_PLANNING_HORIZON,
        action_dim: int = FETCH_ACTION_DIM,
        action_low: float | torch.Tensor = FETCH_ACTION_LOW,
        action_high: float | torch.Tensor = FETCH_ACTION_HIGH,
    ) -> None:
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
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        self.action_dim = action_dim
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32).reshape(-1)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32).reshape(-1)
        if self.action_low.numel() == 1:
            self.action_low = self.action_low.repeat(action_dim)
        if self.action_high.numel() == 1:
            self.action_high = self.action_high.repeat(action_dim)
        if self.action_low.numel() != action_dim or self.action_high.numel() != action_dim:
            raise ValueError("Action bounds must match action_dim.")

    def init_particles(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Samples bounded Gaussian action sequences independently at every step."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("Particle dtype must be floating point.")
        low = self.action_low.to(device=device, dtype=dtype)
        high = self.action_high.to(device=device, dtype=dtype)
        particles = (low + high) / 2 + (high - low) / 4 * torch.randn(
            (batch_size, self.num_particles, self.horizon, self.action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return particles.clamp(low, high)

    def update_particles(
        self,
        particles: torch.Tensor,
        scores: torch.Tensor,
        *,
        sigma: float | None = None,
        temperature: float = DEFAULT_PARTICLE_TEMPERATURE,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Resamples particles by score and perturbs each result with Gaussian noise.

        ``scores`` has shape ``(batch, num_particles)`` and is converted to resampling weights
        with ``softmax(scores / temperature)``. ``sigma`` is the Gaussian standard
        deviation; omitting it uses ``0.1``.
        """
        self._validate_particles(particles)
        self._validate_scores(scores, particles)
        sigma = DEFAULT_PARTICLE_SIGMA if sigma is None else sigma
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        parent_indices = torch.multinomial(
            torch.softmax(scores / temperature, dim=1),
            self.num_particles,
            replacement=True,
            generator=generator,
        )
        resampled = torch.gather(
            particles,
            dim=1,
            index=parent_indices[..., None, None].expand(
                -1, -1, self.horizon, self.action_dim
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
        low = self.action_low.to(device=resampled.device, dtype=resampled.dtype)
        high = self.action_high.to(device=resampled.device, dtype=resampled.dtype)
        return (resampled + sigma * noise).clamp(low, high)

    def _validate_particles(self, particles: torch.Tensor) -> None:
        if not isinstance(particles, torch.Tensor):
            raise TypeError("particles must be a torch.Tensor.")
        if particles.ndim != 4 or particles.shape[1:] != (
            self.num_particles,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"particles must have shape (batch, {self.num_particles}, "
                f"{self.horizon}, {self.action_dim})."
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
