"""Masked sequence losses and optimization for the image-history world model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn

from trans_wm.config import ACTION_HISTORY_LEN, OBS_HISTORY_LEN
from trans_wm.history import append_history, history_windows, previous_history_windows
from trans_wm.model import WorldModel
from utils.replay_buffer import EpisodeBatch, RolloutReplayBuffer


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 100.0
    observation_weight: float = 1.0
    reward_weight: float = 1.0
    vae_reconstruction_weight: float = 1.0
    vae_kl_weight: float = 1e-4
    planning_horizon: int = 16

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
                self.observation_weight,
                self.reward_weight,
                self.vae_reconstruction_weight,
                self.vae_kl_weight,
            )
        ):
            raise ValueError("Loss weights must be non-negative.")


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
    observations: torch.Tensor  # [B, 5 + P, C, H, W]
    obs_valid: torch.Tensor  # [B, 5 + P]
    action_history: torch.Tensor  # [B, 4, action_dim]
    action_valid: torch.Tensor  # [B, 4]
    actions: torch.Tensor  # [B, P, action_dim]
    rewards: torch.Tensor  # [B, P, 1]
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
    observation: torch.Tensor
    reward: torch.Tensor
    vae_reconstruction: torch.Tensor
    vae_kl: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": self.total.detach().item(),
            "observation": self.observation.detach().item(),
            "reward": self.reward.detach().item(),
            "vae_reconstruction": self.vae_reconstruction.detach().item(),
            "vae_kl": self.vae_kl.detach().item(),
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
    planning_horizon: int,
) -> TensorTransitionBatch:
    """Samples transitions with their exact image and action histories."""
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
        raise ValueError("Trans-WM training requires image observations.")
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
    """Returns one z per state plus the exact image windows and masks used."""
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
    return (
        latents.reshape(batch_size, states, model.config.latent_dim),
        obs_windows,
        obs_masks,
    )


def vae_kl_loss(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Standard-normal posterior KL, reduced over latent dimensions per sample."""
    if mean.shape != log_variance.shape or mean.ndim != 2:
        raise ValueError("mean and log_variance must have shape [batch, latent_dim].")
    return -0.5 * (
        1.0 + log_variance - mean.square() - log_variance.exp()
    ).mean(dim=-1)


def world_model_loss(
    model: WorldModel,
    batch: TensorEpisodeBatch,
    config: TrainingConfig,
) -> WorldModelLosses:
    """Computes task losses plus same-window VAE reconstruction and KL."""
    obs_windows, obs_masks = history_windows(
        batch.observations, batch.state_valid, OBS_HISTORY_LEN
    )
    batch_size, states = batch.observations.shape[:2]
    flat_windows = obs_windows.flatten(0, 1)
    flat_masks = obs_masks.flatten(0, 1)
    action_windows, action_masks = _state_action_windows(batch)
    posterior = model.posterior_online(
        flat_windows,
        flat_masks,
        action_windows.flatten(0, 1),
        action_masks.flatten(0, 1),
    )
    latents = posterior.mean.reshape(batch_size, states, model.config.latent_dim)
    vae_reconstruction = model.heads.observation_head(posterior.rsample())
    vae_reconstruction_per_state = (
        (vae_reconstruction - flat_windows).flatten(1).square().mean(dim=1)
    )
    vae_kl_per_state = vae_kl_loss(posterior.mean, posterior.log_variance)
    rollout_z = latents[:, :-1]
    observation_errors = []
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
            rollout_z.flatten(0, 1),
            step_actions.flatten(0, 1),
        ).reshape(*step_actions.shape[:2], model.config.latent_dim)
        predicted_observation = model.heads.observation_head(
            predicted_next_z.flatten(0, 1)
        ).reshape(*step_actions.shape[:2], *obs_windows.shape[2:])
        observation_errors.append(
            (predicted_observation - obs_windows[:, offset + 1 :])
            .flatten(start_dim=2)
            .square()
            .mean(dim=2)
        )
        reward_errors.append(
            (predicted_reward - batch.rewards[:, offset:]).square().squeeze(-1)
        )
        rollout_valid.append(step_valid)
        if offset + 1 < rollout_steps:
            rollout_z = predicted_next_z[:, :-1]

    valid = torch.cat([item.flatten() for item in rollout_valid]).to(dtype=latents.dtype)
    observation_loss = _masked_mean(
        torch.cat([item.flatten() for item in observation_errors]), valid
    )
    reward_loss = _masked_mean(
        torch.cat([item.flatten() for item in reward_errors]), valid
    )
    state_valid = batch.state_valid.flatten().to(dtype=latents.dtype)
    vae_reconstruction_loss = _masked_mean(vae_reconstruction_per_state, state_valid)
    vae_kl = _masked_mean(vae_kl_per_state, state_valid)
    total = (
        config.observation_weight * observation_loss
        + config.reward_weight * reward_loss
        + config.vae_reconstruction_weight * vae_reconstruction_loss
        + config.vae_kl_weight * vae_kl
    )
    return WorldModelLosses(
        total, observation_loss, reward_loss, vae_reconstruction_loss, vae_kl
    )


def transition_world_model_loss(
    model: WorldModel,
    batch: TensorTransitionBatch,
    config: TrainingConfig,
) -> WorldModelLosses:
    """Computes task and same-window VAE objectives on sampled transitions."""
    current_posterior = model.posterior_online(
        batch.current_observations,
        batch.current_obs_valid,
        batch.action_history,
        batch.action_valid,
    )
    batch_size, horizon = batch.actions.shape[:2]
    rollout_z = current_posterior.mean
    rollout_history = batch.action_history
    rollout_history_valid = batch.action_valid
    observation_errors = []
    reward_errors = []
    target_vae_errors = []
    target_vae_kls = []
    for offset in range(horizon):
        action = batch.actions[:, offset]
        target_observation = batch.observations[
            :, offset + 1 : offset + 1 + OBS_HISTORY_LEN
        ]
        target_obs_valid = batch.obs_valid[
            :, offset + 1 : offset + 1 + OBS_HISTORY_LEN
        ]
        rollout_history, rollout_history_valid = append_history(
            rollout_history, rollout_history_valid, action
        )
        target_posterior = model.posterior_online(
            target_observation,
            target_obs_valid,
            rollout_history,
            rollout_history_valid,
        )
        reward_errors.append(
            (model.heads.reward(rollout_z, action) - batch.rewards[:, offset])
            .square()
            .squeeze(-1)
        )
        rollout_z = model.predict_next_online(rollout_z, action)
        observation_errors.append(
            (model.heads.observation_head(rollout_z) - target_observation)
            .flatten(1)
            .square()
            .mean(dim=1)
        )
        target_vae_errors.append(
            (model.heads.observation_head(target_posterior.rsample()) - target_observation)
            .flatten(1)
            .square()
            .mean(dim=1)
        )
        target_vae_kls.append(
            vae_kl_loss(target_posterior.mean, target_posterior.log_variance)
        )
    valid = batch.transition_valid.flatten().to(dtype=rollout_z.dtype)
    observation_loss = _masked_mean(torch.stack(observation_errors, dim=1).flatten(), valid)
    reward_loss = _masked_mean(torch.stack(reward_errors, dim=1).flatten(), valid)
    current_vae_reconstruction = model.heads.observation_head(current_posterior.rsample())
    vae_reconstruction_per_state = torch.cat(
        (
            (current_vae_reconstruction - batch.current_observations)
            .flatten(1)
            .square()
            .mean(dim=1),
            torch.stack(target_vae_errors, dim=1).flatten(),
        )
    )
    state_valid = torch.cat(
        (
            torch.ones(batch_size, device=valid.device, dtype=valid.dtype),
            valid,
        )
    )
    vae_reconstruction_loss = _masked_mean(vae_reconstruction_per_state, state_valid)
    vae_kl = _masked_mean(
        torch.cat(
            (
                vae_kl_loss(current_posterior.mean, current_posterior.log_variance),
                torch.stack(target_vae_kls, dim=1).flatten(),
            )
        ),
        state_valid,
    )
    total = (
        config.observation_weight * observation_loss
        + config.reward_weight * reward_loss
        + config.vae_reconstruction_weight * vae_reconstruction_loss
        + config.vae_kl_weight * vae_kl
    )
    return WorldModelLosses(
        total, observation_loss, reward_loss, vae_reconstruction_loss, vae_kl
    )


class WorldModelTrainer:
    """Optimizes the world model without owning or assuming a policy."""

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
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()
        self.model.update_target()
        return losses.detached()

    def train_transitions(
        self,
        batch: EpisodeBatch,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Trains from sampled transitions without materializing full sequence windows."""
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
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                self.config.grad_clip_norm,
            )
        self.optimizer.step()
        self.model.update_target()
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
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in self.model.parameters() if parameter.requires_grad),
                    self.config.grad_clip_norm,
                )
            self.optimizer.step()
            self.model.update_target()
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
