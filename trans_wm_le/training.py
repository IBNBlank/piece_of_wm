"""JEPA, SIGReg, and task losses for the latent world model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn

from trans_wm_le.config import ACTION_HISTORY_LEN, OBS_HISTORY_LEN
from trans_wm_le.history import append_history, history_windows, previous_history_windows
from trans_wm_le.model import WorldModel
from utils.replay_buffer import EpisodeBatch, RolloutReplayBuffer
from utils.value import discounted_returns


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 100.0
    jepa_weight: float = 1.0
    sigreg_weight: float = 1.0
    reward_weight: float = 1.0
    value_weight: float = 1.0
    sigreg_projections: int = 256
    sigreg_frequencies: int = 17
    sigreg_max_frequency: float = 5.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided.")
        if any(
            weight < 0.0
            for weight in (
                self.jepa_weight,
                self.sigreg_weight,
                self.reward_weight,
                self.value_weight,
            )
        ):
            raise ValueError("Loss weights must be non-negative.")
        if self.sigreg_projections <= 0 or self.sigreg_frequencies <= 0:
            raise ValueError("SIGReg projections and frequencies must be positive.")
        if self.sigreg_max_frequency <= 0.0:
            raise ValueError("sigreg_max_frequency must be positive.")


@dataclass(frozen=True)
class TensorEpisodeBatch:
    observations: torch.Tensor  # [B, T + 1, C, H, W]
    actions: torch.Tensor  # [B, T, action_dim]
    rewards: torch.Tensor  # [B, T, 1]
    terminated: torch.Tensor  # [B, T]
    transition_valid: torch.Tensor  # [B, T]
    state_valid: torch.Tensor  # [B, T + 1]


@dataclass(frozen=True)
class TensorTransitionBatch:
    current_observations: torch.Tensor  # [B, 10, C, H, W]
    next_observations: torch.Tensor  # [B, 10, C, H, W]
    current_obs_valid: torch.Tensor  # [B, 10]
    next_obs_valid: torch.Tensor  # [B, 10]
    action_history: torch.Tensor  # [B, 9, action_dim]
    action_valid: torch.Tensor  # [B, 9]
    action: torch.Tensor  # [B, action_dim]
    reward: torch.Tensor  # [B, 1]
    terminated: torch.Tensor  # [B, 1]


@dataclass(frozen=True)
class WorldModelLosses:
    total: torch.Tensor
    jepa: torch.Tensor
    sigreg: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": self.total.detach().item(),
            "jepa": self.jepa.detach().item(),
            "sigreg": self.sigreg.detach().item(),
            "reward": self.reward.detach().item(),
            "value": self.value.detach().item(),
        }


def tensor_episode_batch(batch: EpisodeBatch, model: WorldModel) -> TensorEpisodeBatch:
    """Loads images as BCHW tensors and constructs state/transition masks."""
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    source = batch.images if batch.images is not None else batch.obs
    observations = _image_tensor(source, model.config.observation_shape, device, dtype)
    actions = torch.as_tensor(batch.action, device=device, dtype=dtype).flatten(start_dim=2)
    rewards = torch.as_tensor(batch.reward, device=device, dtype=dtype).unsqueeze(-1)
    terminated = torch.as_tensor(batch.terminated, device=device, dtype=torch.bool)
    lengths = torch.as_tensor(batch.lengths, device=device)
    if actions.shape[2] != model.config.action_dim:
        raise ValueError("Replay actions do not match model action_shape.")
    time = actions.shape[1]
    transition_valid = torch.arange(time, device=device)[None] < lengths[:, None]
    state_valid = torch.arange(time + 1, device=device)[None] <= lengths[:, None]
    return TensorEpisodeBatch(
        observations, actions, rewards, terminated, transition_valid, state_valid
    )


def sample_transition_batch(
    batch: EpisodeBatch,
    model: WorldModel,
    batch_size: int,
    rng: np.random.Generator,
) -> TensorTransitionBatch:
    """Samples transitions with their exact image and preceding-action histories."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    flat_indices = rng.integers(0, batch.num_transitions, size=batch_size)
    return transition_batch_from_indices(batch, model, flat_indices)


def transition_batch_from_indices(
    batch: EpisodeBatch,
    model: WorldModel,
    flat_indices: np.ndarray,
) -> TensorTransitionBatch:
    """Builds an aligned transition batch from flattened episode indices."""
    source = batch.images if batch.images is not None else batch.obs
    if source.ndim != 5:
        raise ValueError("Trans-WM-LE training requires image observations.")
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    if flat_indices.ndim != 1 or len(flat_indices) == 0:
        raise ValueError("flat_indices must be a non-empty one-dimensional array.")
    lengths = np.asarray(batch.lengths, dtype=np.int64)
    cumulative = np.cumsum(lengths)
    if np.any(flat_indices < 0) or np.any(flat_indices >= cumulative[-1]):
        raise IndexError("Transition index is outside the episode batch.")
    batch_size = len(flat_indices)
    episode_indices = np.searchsorted(cumulative, flat_indices, side="right")
    starts = np.concatenate((np.zeros(1, dtype=np.int64), cumulative[:-1]))
    time_indices = flat_indices - starts[episode_indices]

    current_observations = np.zeros(
        (batch_size, OBS_HISTORY_LEN, *source.shape[2:]), dtype=source.dtype
    )
    next_observations = np.zeros_like(current_observations)
    current_obs_valid = np.zeros((batch_size, OBS_HISTORY_LEN), dtype=bool)
    next_obs_valid = np.zeros_like(current_obs_valid)
    action_shape = batch.action.shape[2:]
    action_history = np.zeros(
        (batch_size, ACTION_HISTORY_LEN, *action_shape), dtype=batch.action.dtype
    )
    action_valid = np.zeros((batch_size, ACTION_HISTORY_LEN), dtype=bool)
    actions = np.empty((batch_size, *action_shape), dtype=batch.action.dtype)
    rewards = np.empty((batch_size, 1), dtype=batch.reward.dtype)
    terminated = np.empty((batch_size, 1), dtype=bool)

    for sample, (episode, timestep) in enumerate(zip(episode_indices, time_indices, strict=True)):
        current_start = max(0, int(timestep) - OBS_HISTORY_LEN + 1)
        current = source[episode, current_start : timestep + 1]
        current_observations[sample, -len(current) :] = current
        current_obs_valid[sample, -len(current) :] = True

        next_start = max(0, int(timestep) - OBS_HISTORY_LEN + 2)
        following = source[episode, next_start : timestep + 2]
        next_observations[sample, -len(following) :] = following
        next_obs_valid[sample, -len(following) :] = True

        action_start = max(0, int(timestep) - ACTION_HISTORY_LEN)
        previous_actions = batch.action[episode, action_start:timestep]
        if len(previous_actions):
            action_history[sample, -len(previous_actions) :] = previous_actions
            action_valid[sample, -len(previous_actions) :] = True
        actions[sample] = batch.action[episode, timestep]
        rewards[sample, 0] = batch.reward[episode, timestep]
        terminated[sample, 0] = batch.terminated[episode, timestep]

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    return TensorTransitionBatch(
        _image_tensor(current_observations, model.config.observation_shape, device, dtype),
        _image_tensor(next_observations, model.config.observation_shape, device, dtype),
        torch.as_tensor(current_obs_valid, device=device),
        torch.as_tensor(next_obs_valid, device=device),
        torch.as_tensor(action_history, device=device, dtype=dtype).flatten(start_dim=2),
        torch.as_tensor(action_valid, device=device),
        torch.as_tensor(actions, device=device, dtype=dtype).flatten(start_dim=1),
        torch.as_tensor(rewards, device=device, dtype=dtype),
        torch.as_tensor(terminated, device=device),
    )


def concatenate_transition_batches(
    batches: Sequence[TensorTransitionBatch],
) -> TensorTransitionBatch:
    """Concatenates transition tensors assembled from different rollout files."""
    return TensorTransitionBatch(
        *(
            torch.cat([getattr(batch, field.name) for batch in batches], dim=0)
            for field in fields(TensorTransitionBatch)
        )
    )


def encode_sequence(
    model: WorldModel, batch: TensorEpisodeBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns one action-conditioned latent per valid state and its image windows."""
    obs_windows, obs_masks = history_windows(
        batch.observations, batch.state_valid, OBS_HISTORY_LEN
    )
    action_windows, action_masks = _state_action_windows(batch)
    batch_size, states = batch.observations.shape[:2]
    latents = model.encode_online(
        obs_windows.flatten(0, 1),
        obs_masks.flatten(0, 1),
        action_windows.flatten(0, 1),
        action_masks.flatten(0, 1),
    )
    return latents.reshape(batch_size, states, model.config.latent_dim), obs_windows, obs_masks


def sigreg_loss(
    latents: torch.Tensor,
    num_projections: int = 256,
    num_frequencies: int = 17,
    max_frequency: float = 5.0,
) -> torch.Tensor:
    """Matches projected latent characteristic functions to a standard Gaussian."""
    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError("latents must have non-empty shape [samples, latent_dim].")
    if num_projections <= 0 or num_frequencies <= 0 or max_frequency <= 0.0:
        raise ValueError("SIGReg sampling parameters must be positive.")
    directions = torch.randn(
        latents.shape[1], num_projections, device=latents.device, dtype=latents.dtype
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(
        torch.finfo(latents.dtype).eps
    )
    projected = latents @ directions
    frequencies = torch.linspace(
        max_frequency / num_frequencies,
        max_frequency,
        num_frequencies,
        device=latents.device,
        dtype=latents.dtype,
    )
    phases = projected[:, :, None] * frequencies
    empirical_real = phases.cos().mean(dim=0)
    empirical_imag = phases.sin().mean(dim=0)
    gaussian_real = torch.exp(-0.5 * frequencies.square())[None]
    return ((empirical_real - gaussian_real).square() + empirical_imag.square()).mean()


def world_model_loss(
    model: WorldModel,
    batch: TensorEpisodeBatch,
    config: TrainingConfig,
) -> WorldModelLosses:
    obs_windows, obs_masks = history_windows(
        batch.observations, batch.state_valid, OBS_HISTORY_LEN
    )
    action_windows, action_masks = _state_action_windows(batch)
    batch_size, states = batch.observations.shape[:2]
    online_latents = model.encode_online(
        obs_windows.flatten(0, 1),
        obs_masks.flatten(0, 1),
        action_windows.flatten(0, 1),
        action_masks.flatten(0, 1),
    ).reshape(batch_size, states, model.config.latent_dim)
    current_z = online_latents[:, :-1].flatten(0, 1)
    predicted_next_z = model.predict_next_online(current_z, batch.actions.flatten(0, 1))
    reward_target = batch.rewards.flatten(0, 1)

    with torch.no_grad():
        target_next_z = model.encode_ema(
            obs_windows[:, 1:].flatten(0, 1),
            obs_masks[:, 1:].flatten(0, 1),
            action_windows[:, 1:].flatten(0, 1),
            action_masks[:, 1:].flatten(0, 1),
        )

    valid = batch.transition_valid.flatten().to(dtype=current_z.dtype)
    jepa = _masked_mean((predicted_next_z - target_next_z).square().mean(dim=-1), valid)
    reward = _masked_mean(
        (model.heads.reward(current_z, batch.actions.flatten(0, 1)) - reward_target)
        .square()
        .squeeze(-1),
        valid,
    )
    value = reward.new_zeros(())
    sigreg = sigreg_loss(
        online_latents[batch.state_valid],
        config.sigreg_projections,
        config.sigreg_frequencies,
        config.sigreg_max_frequency,
    )
    total = (
        config.jepa_weight * jepa
        + config.sigreg_weight * sigreg
        + config.reward_weight * reward
    )
    return WorldModelLosses(total, jepa, sigreg, reward, value)


def transition_world_model_loss(
    model: WorldModel,
    batch: TensorTransitionBatch,
    config: TrainingConfig,
) -> WorldModelLosses:
    current_z = model.encode_online(
        batch.current_observations,
        batch.current_obs_valid,
        batch.action_history,
        batch.action_valid,
    )
    next_action_history, next_action_valid = append_history(
        batch.action_history, batch.action_valid, batch.action
    )
    next_z = model.encode_online(
        batch.next_observations,
        batch.next_obs_valid,
        next_action_history,
        next_action_valid,
    )
    predicted_next_z = model.predict_next_online(current_z, batch.action)
    with torch.no_grad():
        target_next_z = model.encode_ema(
            batch.next_observations,
            batch.next_obs_valid,
            next_action_history,
            next_action_valid,
        )

    jepa = (predicted_next_z - target_next_z).square().mean()
    sigreg = sigreg_loss(
        torch.cat((current_z, next_z), dim=0),
        config.sigreg_projections,
        config.sigreg_frequencies,
        config.sigreg_max_frequency,
    )
    reward = (model.heads.reward(current_z, batch.action) - batch.reward).square().mean()
    value = reward.new_zeros(())
    total = (
        config.jepa_weight * jepa
        + config.sigreg_weight * sigreg
        + config.reward_weight * reward
    )
    return WorldModelLosses(total, jepa, sigreg, reward, value)


class WorldModelTrainer:
    def __init__(self, model: WorldModel, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        value_parameters = set(model.heads.value_head.parameters())
        self.optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and parameter not in value_parameters
            ),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            model.heads.value_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train_batch(self, batch: EpisodeBatch) -> dict[str, float]:
        self.model.train()
        tensor_batch = tensor_episode_batch(batch, self.model)
        self.optimizer.zero_grad(set_to_none=True)
        losses = world_model_loss(self.model, tensor_batch, self.config)
        losses.total.backward()
        self._finish_step()
        return losses.detached()

    def train_transitions(
        self,
        batch: EpisodeBatch,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        self.model.train()
        sampled = sample_transition_batch(batch, self.model, batch_size, rng)
        self.optimizer.zero_grad(set_to_none=True)
        losses = transition_world_model_loss(self.model, sampled, self.config)
        losses.total.backward()
        self._finish_step()
        return losses.detached()

    def train_epoch(
        self,
        batches: Sequence[EpisodeBatch],
        batch_size: int,
        rng: np.random.Generator,
        on_update: Callable[[dict[str, float]], None] | None = None,
    ) -> dict[str, float]:
        """Trains once on every valid transition, in shuffled minibatches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not batches:
            raise ValueError("An epoch requires at least one rollout batch.")
        totals: dict[str, float] = {}
        num_transitions = 0
        cumulative = np.cumsum([batch.num_transitions for batch in batches])
        indices = rng.permutation(int(cumulative[-1]))
        starts = np.concatenate((np.zeros(1, dtype=np.int64), cumulative[:-1]))
        for start in range(0, len(indices), batch_size):
            minibatch_indices = indices[start : start + batch_size]
            batch_indices = np.searchsorted(cumulative, minibatch_indices, side="right")
            parts = [
                transition_batch_from_indices(
                    batches[int(batch_index)],
                    self.model,
                    minibatch_indices[batch_indices == batch_index] - starts[batch_index],
                )
                for batch_index in np.unique(batch_indices)
            ]
            self.model.train()
            sampled = parts[0] if len(parts) == 1 else concatenate_transition_batches(parts)
            self.optimizer.zero_grad(set_to_none=True)
            losses = transition_world_model_loss(self.model, sampled, self.config)
            losses.total.backward()
            self._finish_step()
            metrics = losses.detached()
            if on_update is not None:
                on_update(metrics)
            count = len(minibatch_indices)
            num_transitions += count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * count
        return {name: value / num_transitions for name, value in totals.items()}

    def evaluate_transitions(
        self,
        batch: EpisodeBatch,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Evaluates every valid transition once, in shuffled minibatches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model.eval()
        totals: dict[str, float] = {}
        num_transitions = 0
        indices = rng.permutation(batch.num_transitions)
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                minibatch_indices = indices[start : start + batch_size]
                sampled = transition_batch_from_indices(
                    batch, self.model, minibatch_indices
                )
                metrics = transition_world_model_loss(
                    self.model, sampled, self.config
                ).detached()
                count = len(minibatch_indices)
                num_transitions += count
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * count
        return {name: value / num_transitions for name, value in totals.items()}

    def train_value_rollout(
        self, latents: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, float]:
        """Fits every online critic to aligned value targets from a real rollout."""
        if latents.ndim != 2 or latents.shape[1] != self.model.config.latent_dim:
            raise ValueError("latents must have shape [time, latent_dim].")
        returns = returns.reshape(-1, 1)
        if returns.shape[0] != latents.shape[0]:
            raise ValueError("latents and returns must have the same time dimension.")
        self.model.train()
        self.value_optimizer.zero_grad(set_to_none=True)
        prediction = self.model.heads.value_head.minimum(latents.detach())
        loss = (prediction - returns.detach()).square().mean()
        (self.config.value_weight * loss).backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.model.heads.value_head.parameters(), self.config.grad_clip_norm
            )
        self.value_optimizer.step()
        self.model.update_target()
        return {"value": loss.detach().item()}

    def _finish_step(self) -> None:
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()
        self.model.update_target()

    def fit(
        self,
        replay_buffer: RolloutReplayBuffer,
        rollouts: int,
        *,
        batch_size: int = 8,
        epochs_per_rollout: int = 10,
        sample_rollouts: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[dict[str, float]]:
        if rollouts <= 0 or epochs_per_rollout <= 0:
            raise ValueError("rollouts and epochs_per_rollout must be positive.")
        rng = rng or np.random.default_rng()
        history = []
        for _ in range(rollouts):
            batch = replay_buffer.sample(sample_rollouts)
            metrics = {}
            for _ in range(epochs_per_rollout):
                metrics = self.train_transitions(batch, batch_size, rng)
            history.append(metrics)
        return history


def _state_action_windows(
    batch: TensorEpisodeBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    padding = torch.zeros_like(batch.actions[:, :1])
    padded_actions = torch.cat((batch.actions, padding), dim=1)
    padded_valid = torch.cat(
        (batch.transition_valid, torch.zeros_like(batch.transition_valid[:, :1])), dim=1
    )
    return previous_history_windows(padded_actions, padded_valid, ACTION_HISTORY_LEN)


def _image_tensor(
    array: np.ndarray,
    observation_shape: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if array.ndim != 5:
        raise ValueError("Image observations must have shape [batch, time, C, H, W] or HWC.")
    channels, height, width = observation_shape
    if array.shape[2:] == (channels, height, width):
        tensor = torch.as_tensor(array, device=device, dtype=dtype)
    elif array.shape[2:] == (height, width, channels):
        tensor = torch.as_tensor(array, device=device, dtype=dtype).permute(0, 1, 4, 2, 3)
    else:
        raise ValueError("Replay images do not match model observation_shape.")
    if np.issubdtype(array.dtype, np.integer):
        tensor = tensor / float(np.iinfo(array.dtype).max)
    return tensor


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape != valid.shape:
        raise ValueError("Loss values and validity mask must have identical shapes.")
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("A training batch must contain at least one valid transition.")
    return (values * valid).sum() / count
