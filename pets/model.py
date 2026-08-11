"""PETS ensemble dynamics model, model environment, and CEM planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as mbrl_common
import numpy as np
import torch
from omegaconf import OmegaConf

from utils.common import TrainingHistory
from utils.env import TerminationFn, no_termination, space_shapes


@dataclass
class PETSConfig:
    ensemble_size: int = 5
    hidden_size: int = 200
    num_layers: int = 3
    planning_horizon: int = 15
    cem_iterations: int = 4
    cem_population_size: int = 256
    cem_elite_ratio: float = 0.1
    cem_alpha: float = 0.1
    num_particles: int = 20


@dataclass
class ModelTrainingConfig:
    batch_size: int = 64
    validation_ratio: float = 0.05
    epochs: int = 25
    patience: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5


def _vector_shapes(env: gym.Env) -> tuple[tuple[int, ...], tuple[int, ...]]:
    obs_shape, action_shape = space_shapes(env)
    if len(obs_shape) != 1 or len(action_shape) != 1:
        raise ValueError(
            "The GaussianMLP PETS model currently supports vector observations and actions. "
            "CarRacing-v2 data can be collected, but training needs an image encoder / latent "
            "dynamics model before it is supported."
        )
    return obs_shape, action_shape


def build_dynamics_model(env: gym.Env, config: PETSConfig, device: str) -> models.OneDTransitionRewardModel:
    """Builds the probabilistic ensemble used by PETS."""
    obs_shape, action_shape = _vector_shapes(env)
    base_model = models.GaussianMLP(
        in_size=obs_shape[0] + action_shape[0],
        out_size=obs_shape[0] + 1,
        device=device,
        num_layers=config.num_layers,
        ensemble_size=config.ensemble_size,
        hid_size=config.hidden_size,
        deterministic=False,
        propagation_method="fixed_model",
        activation_fn_cfg={"_target_": "torch.nn.LeakyReLU", "negative_slope": 0.01},
    )
    return models.OneDTransitionRewardModel(
        base_model,
        target_is_delta=True,
        normalize=True,
        normalize_double_precision=True,
        learned_rewards=True,
        num_elites=config.ensemble_size,
    )


def build_model_env(
    env: gym.Env,
    dynamics_model: models.OneDTransitionRewardModel,
    *,
    device: str,
    seed: int,
    termination_fn: TerminationFn = no_termination,
) -> models.ModelEnv:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return models.ModelEnv(env, dynamics_model, termination_fn, reward_fn=None, generator=generator)


def build_pets_agent(env: gym.Env, model_env: models.ModelEnv, config: PETSConfig, device: str) -> planning.TrajectoryOptimizerAgent:
    """Creates a CEM MPC agent whose objective is evaluated in ``model_env``."""
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("PETS requires a continuous Box action space.")
    optimizer_cfg = OmegaConf.create(
        {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": config.cem_iterations,
            "elite_ratio": config.cem_elite_ratio,
            "population_size": config.cem_population_size,
            "alpha": config.cem_alpha,
            "device": device,
            "return_mean_elites": True,
            "clipped_normal": False,
        }
    )
    agent = planning.TrajectoryOptimizerAgent(
        optimizer_cfg=optimizer_cfg,
        action_lb=env.action_space.low,
        action_ub=env.action_space.high,
        planning_horizon=config.planning_horizon,
        replan_freq=1,
        verbose=False,
    )

    def evaluate_action_sequences(initial_state: np.ndarray, action_sequences: torch.Tensor) -> torch.Tensor:
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=config.num_particles
        )

    agent.set_trajectory_eval_fn(evaluate_action_sequences)
    return agent


def build_model_trainer(model: models.OneDTransitionRewardModel, config: ModelTrainingConfig) -> models.ModelTrainer:
    return models.ModelTrainer(model, optim_lr=config.learning_rate, weight_decay=config.weight_decay)


def load_dynamics_model(
    dynamics_model: models.OneDTransitionRewardModel, model_dir: str | Path, device: str
) -> None:
    """Loads an mbrl checkpoint onto ``device`` regardless of its training device."""
    model_dir = Path(model_dir)
    try:
        checkpoint = torch.load(model_dir / "model.pth", map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was added.
        checkpoint = torch.load(model_dir / "model.pth", map_location=device)
    dynamics_model.model.load_state_dict(checkpoint["state_dict"])
    dynamics_model.model.elite_models = checkpoint["elite_models"]
    if dynamics_model.input_normalizer is not None:
        dynamics_model.input_normalizer.load(model_dir)


def train_dynamics_model(
    model: models.OneDTransitionRewardModel,
    replay_buffer: Any,
    config: ModelTrainingConfig,
    *,
    trainer: models.ModelTrainer | None = None,
    history: TrainingHistory | None = None,
) -> tuple[models.ModelTrainer, TrainingHistory]:
    """Updates normalization statistics and fits the PETS ensemble on a replay buffer."""
    if replay_buffer.num_stored < 2:
        raise ValueError("At least two transitions are required to train a dynamics model.")
    history = history or TrainingHistory()
    trainer = trainer or build_model_trainer(model, config)
    model.update_normalizer(replay_buffer.get_all())
    train_iter, val_iter = mbrl_common.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=min(config.batch_size, replay_buffer.num_stored),
        val_ratio=config.validation_ratio,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=False,
    )
    trainer.train(
        train_iter,
        dataset_val=val_iter,
        num_epochs=config.epochs,
        patience=config.patience,
        callback=history.callback,
        silent=True,
    )
    return trainer, history
