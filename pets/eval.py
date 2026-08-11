"""Evaluate a saved PETS model in the real environment and record MP4 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.wrappers import RecordVideo

from pets.model import PETSConfig, build_dynamics_model, build_model_env, build_pets_agent, load_dynamics_model
from utils.common import configure_logging, resolve_device, seed_everything
from utils.env import make_env, reset_env, space_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("runs/pets-offline"))
    parser.add_argument("--env-id", default=None, help="Defaults to the environment stored with the model")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pets-eval"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--device", default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--no-video", action="store_true", help="Run evaluation without writing MP4 files")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_model_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "model_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "pets" not in metadata or "env_id" not in metadata:
        raise ValueError(f"Invalid PETS model metadata: {metadata_path}")
    return metadata


def action_from_agent(agent: Any, observation: np.ndarray, action_shape: tuple[int, ...]) -> np.ndarray:
    action = agent.act(observation)
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(action_shape)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.video_fps <= 0:
        raise ValueError("--episodes, --max-steps, and --video-fps must be positive.")
    if not (args.model_dir / "model.pth").is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_dir / 'model.pth'}")

    logger = configure_logging(args.verbose)
    seed_everything(args.seed)
    metadata = load_model_metadata(args.model_dir)
    env_id = args.env_id or metadata["env_id"]
    if args.env_id is not None and args.env_id != metadata["env_id"]:
        raise ValueError(f"Model was trained for {metadata['env_id']}, not --env-id {args.env_id}.")
    pets_config = PETSConfig(**metadata["pets"])
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_video:
        try:
            import moviepy  # noqa: F401
        except ModuleNotFoundError as error:
            raise RuntimeError("Video evaluation requires moviepy. Run ./venv.sh to install it.") from error

    env = make_env(env_id, render_mode=None if args.no_video else "rgb_array")
    if not args.no_video:
        env = RecordVideo(
            env,
            video_folder=str(args.output_dir / "videos"),
            episode_trigger=lambda _episode: True,
            name_prefix="pets-eval",
            fps=args.video_fps,
            disable_logger=True,
        )
    try:
        _, action_shape = space_shapes(env)
        dynamics_model = build_dynamics_model(env, pets_config, device)
        load_dynamics_model(dynamics_model, args.model_dir, device)
        model_env = build_model_env(env, dynamics_model, device=device, seed=args.seed)
        agent = build_pets_agent(env, model_env, pets_config, device)

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        for episode in range(args.episodes):
            observation = reset_env(env, args.seed + episode)
            agent.reset()
            total_reward = 0.0
            for step in range(args.max_steps):
                action = action_from_agent(agent, observation, action_shape)
                observation, reward, terminated, truncated, _ = env.step(action)
                observation = np.asarray(observation, dtype=np.float32)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            episode_rewards.append(total_reward)
            episode_lengths.append(step + 1)
            logger.info("episode=%d reward=%.2f steps=%d", episode, total_reward, step + 1)
    finally:
        env.close()

    summary = {
        "env_id": env_id,
        "model_dir": str(args.model_dir),
        "seed": args.seed,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": float(np.mean(episode_rewards)),
        "video_dir": None if args.no_video else str(args.output_dir / "videos"),
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("mean_reward=%.2f; results saved to %s", summary["mean_reward"], args.output_dir)


if __name__ == "__main__":
    main()
