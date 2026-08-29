"""Collect complete multi-environment episodes for world-model training."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from utils.common import configure_logging, seed_everything
from utils.data import ROLLOUT_FILE_TEMPLATE, save_dataset_metadata, save_episode_batch
from utils.env import make_env, observation_to_array, reset_env
from utils.replay_buffer import EpisodeBatch


IMAGE_SIZE = (128, 128)


def _render_image(env) -> np.ndarray:
    image = np.asarray(env.render())
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 HWC image from env.render(), got {image.shape} {image.dtype}.")
    return np.asarray(Image.fromarray(image).resize(IMAGE_SIZE, Image.Resampling.BILINEAR))


def _collect_random_rollout(
    envs,
    max_steps: int,
    seed: int,
    rollout_index: int,
    *,
    collect_images: bool,
) -> EpisodeBatch:
    observations = []
    observation_sequences: list[list[np.ndarray]] = []
    action_sequences: list[list[np.ndarray]] = []
    reward_sequences: list[list[float]] = []
    terminated_sequences: list[list[bool]] = []
    truncated_sequences: list[list[bool]] = []
    image_sequences: list[list[np.ndarray]] | None = [] if collect_images else None

    for env_index, env in enumerate(envs):
        rollout_seed = seed + rollout_index * len(envs) + env_index
        env.action_space.seed(rollout_seed)
        observation = reset_env(env, rollout_seed)
        observations.append(observation)
        observation_sequences.append([observation])
        action_sequences.append([])
        reward_sequences.append([])
        terminated_sequences.append([])
        truncated_sequences.append([])
        if image_sequences is not None:
            image_sequences.append([_render_image(env)])

    active = [True] * len(envs)
    for _ in range(max_steps):
        for env_index, env in enumerate(envs):
            if not active[env_index]:
                continue
            action = np.asarray(env.action_space.sample(), dtype=np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = observation_to_array(next_obs)
            action_sequences[env_index].append(action)
            reward_sequences[env_index].append(float(reward))
            terminated_sequences[env_index].append(terminated)
            truncated_sequences[env_index].append(truncated)
            observation_sequences[env_index].append(next_obs)
            observations[env_index] = next_obs
            if image_sequences is not None:
                image_sequences[env_index].append(_render_image(env))
            active[env_index] = not (terminated or truncated)
        if not any(active):
            break

    for env_index, is_active in enumerate(active):
        if is_active:
            truncated_sequences[env_index][-1] = True

    lengths = np.asarray([len(actions) for actions in action_sequences], dtype=np.int64)
    max_length = int(lengths.max())
    num_envs = len(envs)
    obs = np.zeros((num_envs, max_length + 1, *observations[0].shape), dtype=np.float32)
    action = np.zeros((num_envs, max_length, *action_sequences[0][0].shape), dtype=np.float32)
    reward = np.zeros((num_envs, max_length), dtype=np.float32)
    terminated = np.zeros((num_envs, max_length), dtype=bool)
    truncated = np.zeros((num_envs, max_length), dtype=bool)
    for env_index, length in enumerate(lengths):
        obs[env_index, : length + 1] = observation_sequences[env_index]
        action[env_index, :length] = action_sequences[env_index]
        reward[env_index, :length] = reward_sequences[env_index]
        terminated[env_index, :length] = terminated_sequences[env_index]
        truncated[env_index, :length] = truncated_sequences[env_index]

    images = None
    if image_sequences is not None:
        images = np.zeros((num_envs, max_length + 1, *image_sequences[0][0].shape), dtype=np.uint8)
        for env_index, length in enumerate(lengths):
            images[env_index, : length + 1] = image_sequences[env_index]
    return EpisodeBatch(obs, action, reward, terminated, truncated, lengths, images)


def iter_random_rollouts(
    env_id: str,
    rollouts: int,
    max_steps: int,
    seed: int,
    *,
    num_envs: int = 1,
    collect_images: bool = True,
) -> Iterator[tuple[int, EpisodeBatch]]:
    """Yields one complete episode batch for every parallel environment."""
    if rollouts <= 0 or max_steps <= 0 or num_envs <= 0:
        raise ValueError("rollouts, max_steps, and num_envs must all be positive.")
    envs = [make_env(env_id, render_mode="rgb_array" if collect_images else None) for _ in range(num_envs)]
    try:
        for rollout_index in range(rollouts):
            yield rollout_index, _collect_random_rollout(
                envs, max_steps, seed, rollout_index, collect_images=collect_images
            )
    finally:
        for env in envs:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="FetchPickAndPlace-v4")
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fetch-pick-and-place-random"))
    parser.add_argument("--no-images", dest="collect_images", action="store_false")
    parser.set_defaults(collect_images=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    rollout_files: list[str] = []
    num_transitions = 0
    for rollout_index, batch in tqdm(
        iter_random_rollouts(
            args.env_id, args.rollouts, args.max_steps, args.seed,
            num_envs=args.num_envs, collect_images=args.collect_images,
        ),
        total=args.rollouts,
        desc="Collecting rollouts",
        unit="rollout",
    ):
        path = save_episode_batch(batch, args.output_dir / ROLLOUT_FILE_TEMPLATE.format(rollout_index=rollout_index))
        rollout_files.append(path.name)
        num_transitions += batch.num_transitions
    save_dataset_metadata(
        args.output_dir,
        env_id=args.env_id,
        rollout_files=rollout_files,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        num_transitions=num_transitions,
        seed=args.seed,
        images_collected=args.collect_images,
    )
    logger.info("Saved %d transitions from %d rollouts to %s", num_transitions, args.rollouts, args.output_dir)


if __name__ == "__main__":
    main()
