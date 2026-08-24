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


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 100.0
    jepa_weight: float = 1.0
    sigreg_weight: float = 0.2
    reward_weight: float = 1.0
    sigreg_projections: int = 256
    sigreg_frequencies: int = 17
    sigreg_max_frequency: float = 5.0
    planning_horizon: int = 20

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided.")
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive.")
        if any(
            weight < 0.0
            for weight in (
                self.jepa_weight,
                self.sigreg_weight,
                self.reward_weight,
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
    returns: torch.Tensor  # [B, T + 1, 1]
    terminated: torch.Tensor  # [B, T]
    transition_valid: torch.Tensor  # [B, T]
    state_valid: torch.Tensor  # [B, T + 1]


@dataclass(frozen=True)
class TensorTransitionBatch:
    observations: torch.Tensor  # [B, 3 + P, C, H, W]
    obs_valid: torch.Tensor  # [B, 3 + P]
    action_history: torch.Tensor  # [B, 2, action_dim]
    action_valid: torch.Tensor  # [B, 2]
    actions: torch.Tensor  # [B, P, action_dim]
    rewards: torch.Tensor  # [B, P, 1]
    next_returns: torch.Tensor  # [B, P, 1]
    terminated: torch.Tensor  # [B, P]
    transition_valid: torch.Tensor  # [B, P]

    @property
    def current_observations(self) -> torch.Tensor:
        return self.observations[:, :OBS_HISTORY_LEN]

    @property
    def current_obs_valid(self) -> torch.Tensor:
        return self.obs_valid[:, :OBS_HISTORY_LEN]

    @property
    def target_observations(self) -> torch.Tensor:
        return torch.stack(
            [
                self.observations[:, offset + 1 : offset + 1 + OBS_HISTORY_LEN]
                for offset in range(self.actions.shape[1])
            ],
            dim=1,
        )

    @property
    def target_obs_valid(self) -> torch.Tensor:
        return torch.stack(
            [
                self.obs_valid[:, offset + 1 : offset + 1 + OBS_HISTORY_LEN]
                for offset in range(self.actions.shape[1])
            ],
            dim=1,
        )

    @property
    def next_observations(self) -> torch.Tensor:
        return self.observations[:, 1 : 1 + OBS_HISTORY_LEN]

    @property
    def next_obs_valid(self) -> torch.Tensor:
        return self.obs_valid[:, 1 : 1 + OBS_HISTORY_LEN]

    @property
    def action(self) -> torch.Tensor:
        return self.actions[:, 0]

    @property
    def reward(self) -> torch.Tensor:
        return self.rewards[:, 0]


@dataclass(frozen=True)
class WorldModelLosses:
    total: torch.Tensor
    jepa: torch.Tensor
    sigreg: torch.Tensor
    reward: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": self.total.detach().item(),
            "jepa": self.jepa.detach().item(),
            "sigreg": self.sigreg.detach().item(),
            "reward": self.reward.detach().item(),
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
    returns = _returns_to_go(rewards, transition_valid)
    return TensorEpisodeBatch(
        observations, actions, rewards, returns, terminated, transition_valid, state_valid
    )


def sample_transition_batch(
    batch: EpisodeBatch,
    model: WorldModel,
    batch_size: int,
    rng: np.random.Generator,
    planning_horizon: int,
) -> TensorTransitionBatch:
    """Samples transitions with their exact image and preceding-action histories."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    flat_indices = rng.integers(0, batch.num_transitions, size=batch_size)
    return transition_batch_from_indices(
        batch, model, flat_indices, planning_horizon=planning_horizon
    )


def transition_batch_from_indices(
    batch: EpisodeBatch,
    model: WorldModel,
    flat_indices: np.ndarray,
    planning_horizon: int,
) -> TensorTransitionBatch:
    """Builds an aligned transition batch from flattened episode indices."""
    source = batch.images if batch.images is not None else batch.obs
    if source.ndim != 5:
        raise ValueError("Trans-WM-LE training requires image observations.")
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    if flat_indices.ndim != 1 or len(flat_indices) == 0:
        raise ValueError("flat_indices must be a non-empty one-dimensional array.")
    if planning_horizon <= 0:
        raise ValueError("planning_horizon must be positive.")
    lengths = np.asarray(batch.lengths, dtype=np.int64)
    cumulative = np.cumsum(lengths)
    if np.any(flat_indices < 0) or np.any(flat_indices >= cumulative[-1]):
        raise IndexError("Transition index is outside the episode batch.")
    batch_size = len(flat_indices)
    episode_indices = np.searchsorted(cumulative, flat_indices, side="right")
    starts = np.concatenate((np.zeros(1, dtype=np.int64), cumulative[:-1]))
    time_indices = flat_indices - starts[episode_indices]

    observations = np.zeros(
        (batch_size, OBS_HISTORY_LEN + planning_horizon, *source.shape[2:]),
        dtype=source.dtype,
    )
    obs_valid = np.zeros((batch_size, OBS_HISTORY_LEN + planning_horizon), dtype=bool)
    action_shape = batch.action.shape[2:]
    action_history = np.zeros(
        (batch_size, ACTION_HISTORY_LEN, *action_shape), dtype=batch.action.dtype
    )
    action_valid = np.zeros((batch_size, ACTION_HISTORY_LEN), dtype=bool)
    actions = np.zeros(
        (batch_size, planning_horizon, *action_shape), dtype=batch.action.dtype
    )
    rewards = np.zeros((batch_size, planning_horizon, 1), dtype=batch.reward.dtype)
    next_returns = np.zeros((batch_size, planning_horizon, 1), dtype=batch.reward.dtype)
    terminated = np.zeros((batch_size, planning_horizon), dtype=bool)
    transition_valid = np.zeros((batch_size, planning_horizon), dtype=bool)

    for sample, (episode, timestep) in enumerate(zip(episode_indices, time_indices, strict=True)):
        sequence_start = int(timestep) - OBS_HISTORY_LEN + 1
        source_start = max(0, sequence_start)
        source_end = min(int(lengths[episode]) + 1, int(timestep) + planning_horizon + 1)
        destination_start = source_start - sequence_start
        sequence = source[episode, source_start:source_end]
        observations[
            sample, destination_start : destination_start + len(sequence)
        ] = sequence
        obs_valid[sample, destination_start : destination_start + len(sequence)] = True

        action_start = max(0, int(timestep) - ACTION_HISTORY_LEN)
        previous_actions = batch.action[episode, action_start:timestep]
        if len(previous_actions):
            action_history[sample, -len(previous_actions) :] = previous_actions
            action_valid[sample, -len(previous_actions) :] = True
        rollout_length = min(planning_horizon, int(lengths[episode] - timestep))
        for offset in range(rollout_length):
            transition = int(timestep + offset)
            actions[sample, offset] = batch.action[episode, transition]
            rewards[sample, offset, 0] = batch.reward[episode, transition]
            next_returns[sample, offset, 0] = batch.reward[
                episode, transition + 1 : lengths[episode]
            ].sum()
            terminated[sample, offset] = batch.terminated[episode, transition]
            transition_valid[sample, offset] = True

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    return TensorTransitionBatch(
        _image_tensor(observations, model.config.observation_shape, device, dtype),
        torch.as_tensor(obs_valid, device=device),
        torch.as_tensor(action_history, device=device, dtype=dtype).flatten(start_dim=2),
        torch.as_tensor(action_valid, device=device),
        torch.as_tensor(actions, device=device, dtype=dtype).flatten(start_dim=2),
        torch.as_tensor(rewards, device=device, dtype=dtype),
        torch.as_tensor(next_returns, device=device, dtype=dtype),
        torch.as_tensor(terminated, device=device),
        torch.as_tensor(transition_valid, device=device),
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
    with torch.no_grad():
        target_latents = model.encode_ema(
            obs_windows.flatten(0, 1),
            obs_masks.flatten(0, 1),
            action_windows.flatten(0, 1),
            action_masks.flatten(0, 1),
        ).reshape(batch_size, states, model.config.latent_dim)

    rollout_z = online_latents[:, :-1]
    jepa_errors = []
    reward_errors = []
    rollout_valid = []
    rollout_steps = min(config.planning_horizon, batch.actions.shape[1])
    for offset in range(rollout_steps):
        step_actions = batch.actions[:, offset:]
        step_valid = batch.transition_valid[:, offset:]
        predicted_reward = model.heads.reward(
            rollout_z.flatten(0, 1), step_actions.flatten(0, 1)
        ).reshape(*step_actions.shape[:2], 1)
        predicted_next_z = model.predict_next_online(
            rollout_z.flatten(0, 1), step_actions.flatten(0, 1)
        ).reshape(*step_actions.shape[:2], model.config.latent_dim)
        jepa_errors.append(
            (predicted_next_z - target_latents[:, offset + 1 :]).square().mean(dim=-1)
        )
        reward_errors.append(
            (predicted_reward - batch.rewards[:, offset:]).square().squeeze(-1)
        )
        rollout_valid.append(step_valid)
        rollout_z = predicted_next_z[:, :-1]

    valid = torch.cat([item.flatten() for item in rollout_valid]).to(
        dtype=online_latents.dtype
    )
    jepa = _masked_mean(
        torch.cat([item.flatten() for item in jepa_errors]), valid
    )
    reward = _masked_mean(
        torch.cat([item.flatten() for item in reward_errors]), valid
    )
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
    return WorldModelLosses(total, jepa, sigreg, reward)


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
    target_histories = []
    target_history_valid = []
    action_history = batch.action_history
    action_valid = batch.action_valid
    for offset in range(batch.actions.shape[1]):
        action_history, action_valid = append_history(
            action_history, action_valid, batch.actions[:, offset]
        )
        target_histories.append(action_history)
        target_history_valid.append(action_valid)
    target_online_latents = []
    target_latents = []
    for offset, (target_history, target_valid) in enumerate(
        zip(target_histories, target_history_valid, strict=True)
    ):
        target_observation = batch.observations[
            :, offset + 1 : offset + 1 + OBS_HISTORY_LEN
        ]
        target_obs_valid = batch.obs_valid[
            :, offset + 1 : offset + 1 + OBS_HISTORY_LEN
        ]
        target_online_latents.append(
            model.encode_online(
                target_observation,
                target_obs_valid,
                target_history,
                target_valid,
            )
        )
        with torch.no_grad():
            target_latents.append(
                model.encode_ema(
                    target_observation,
                    target_obs_valid,
                    target_history,
                    target_valid,
                )
            )
    target_online_z = torch.stack(target_online_latents, dim=1)
    target_z = torch.stack(target_latents, dim=1)

    rollout_z = current_z
    jepa_errors = []
    reward_errors = []
    for offset in range(batch.actions.shape[1]):
        action = batch.actions[:, offset]
        reward_errors.append(
            (model.heads.reward(rollout_z, action) - batch.rewards[:, offset])
            .square()
            .squeeze(-1)
        )
        rollout_z = model.predict_next_online(rollout_z, action)
        jepa_errors.append((rollout_z - target_z[:, offset]).square().mean(dim=-1))
    valid = batch.transition_valid.flatten().to(dtype=current_z.dtype)
    jepa = _masked_mean(torch.stack(jepa_errors, dim=1).flatten(), valid)
    sigreg = sigreg_loss(
        torch.cat((current_z, target_online_z[batch.transition_valid]), dim=0),
        config.sigreg_projections,
        config.sigreg_frequencies,
        config.sigreg_max_frequency,
    )
    reward = _masked_mean(torch.stack(reward_errors, dim=1).flatten(), valid)
    total = (
        config.jepa_weight * jepa
        + config.sigreg_weight * sigreg
        + config.reward_weight * reward
    )
    return WorldModelLosses(total, jepa, sigreg, reward)


class WorldModelTrainer:
    def __init__(self, model: WorldModel, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
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
        sampled = sample_transition_batch(
            batch,
            self.model,
            batch_size,
            rng,
            planning_horizon=self.config.planning_horizon,
        )
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
                    planning_horizon=self.config.planning_horizon,
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
                    batch,
                    self.model,
                    minibatch_indices,
                    planning_horizon=self.config.planning_horizon,
                )
                metrics = transition_world_model_loss(
                    self.model, sampled, self.config
                ).detached()
                count = len(minibatch_indices)
                num_transitions += count
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + value * count
        return {name: value / num_transitions for name, value in totals.items()}

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


def _returns_to_go(rewards: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Returns undiscounted rewards following each state, with zero at episode end."""
    masked_rewards = rewards * valid[:, :, None]
    returns = torch.zeros(
        (rewards.shape[0], rewards.shape[1] + 1, 1),
        device=rewards.device,
        dtype=rewards.dtype,
    )
    returns[:, :-1] = masked_rewards.flip(1).cumsum(1).flip(1)
    return returns
