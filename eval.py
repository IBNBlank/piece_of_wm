"""Evaluate world-model policies by interacting with a Gymnasium environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from utils.common import seed_everything
from utils.env import make_env, reset_env
from utils.particle_policy import ParticlePolicy


os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

OBS_HISTORY_LEN = 5
ACTION_HISTORY_LEN = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "trans_wm", "trans_wm_le"), default="all")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument(
        "--trans-wm-checkpoint",
        type=Path,
        default=Path("runs/trans_wm/checkpoint_best.pt"),
    )
    parser.add_argument(
        "--trans-wm-le-checkpoint",
        type=Path,
        default=Path("runs/trans_wm_le/checkpoint_best.pt"),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--particle-updates", type=int, default=5)
    parser.add_argument("--particle-sigma", type=float, default=0.1)
    parser.add_argument("--particle-temperature", type=float, default=2.0)
    parser.add_argument("--planning-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/results.json"))
    parser.add_argument("--visual-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.episodes,
        args.max_steps,
        args.num_particles,
        args.particle_updates,
        args.planning_horizon,
        args.fps,
    ) <= 0:
        raise ValueError(
            "Episodes, steps, particles, particle updates, horizon, and FPS must be positive."
        )
    if args.particle_sigma < 0.0:
        raise ValueError("--particle-sigma must be non-negative.")
    if args.particle_temperature <= 0.0:
        raise ValueError("--particle-temperature must be positive.")

    seed_everything(args.seed)
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoints = {
        "trans_wm": args.trans_wm_checkpoint,
        "trans_wm_le": args.trans_wm_le_checkpoint,
    }
    model_names = tuple(checkpoints) if args.model == "all" else (args.model,)
    baseline_returns = evaluate_random_policy(
        args.env_id, args.episodes, args.max_steps, args.seed
    )
    results: dict[str, Any] = {
        "mode": "online_environment",
        "env_id": args.env_id,
        "device": str(device),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "num_particles": args.num_particles,
        "particle_updates": args.particle_updates,
        "particle_sigma": args.particle_sigma,
        "particle_temperature": args.particle_temperature,
        "planning_horizon": args.planning_horizon,
        "seed": args.seed,
        "random_baseline": _return_summary(baseline_returns),
        "models": {},
    }

    for model_name in model_names:
        model, checkpoint_rollout = load_model(model_name, checkpoints[model_name], device)
        model_result = evaluate_online(
            model_name,
            model,
            args.env_id,
            args.episodes,
            args.max_steps,
            args.num_particles,
            args.particle_updates,
            args.particle_sigma,
            args.particle_temperature,
            args.planning_horizon,
            args.seed,
            args.visual_dir / model_name,
            args.fps,
            record_video=not args.no_video,
        )
        model_result["checkpoint"] = str(checkpoints[model_name])
        model_result["checkpoint_rollout"] = checkpoint_rollout
        results["models"][model_name] = model_result
        print(
            f"[{model_name}] online_return={model_result['mean_return']:.3f} "
            f"std={model_result['std_return']:.3f} "
            f"random_baseline={results['random_baseline']['mean_return']:.3f}",
            flush=True,
        )
        for artifact in model_result["artifacts"].values():
            print(f"[{model_name}] artifact: {artifact}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Online evaluation saved to {args.output}", flush=True)


def load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> tuple[Any, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint_path}")
    package = importlib.import_module(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_version = {"trans_wm": 7, "trans_wm_le": 7}[model_name]
    if checkpoint.get("architecture_version") != expected_version:
        raise ValueError(
            f"{checkpoint_path} is incompatible with the current model architecture; "
            "retrain from scratch."
        )
    model = package.WorldModel(package.WorldModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint["rollout"])


def evaluate_online(
    model_name: str,
    model: Any,
    env_id: str,
    episodes: int,
    max_steps: int,
    num_particles: int,
    particle_updates: int,
    particle_sigma: float,
    particle_temperature: float,
    planning_horizon: int,
    seed: int,
    output_dir: Path,
    fps: int,
    *,
    record_video: bool,
) -> dict[str, Any]:
    _validate_policy_environment(env_id, model)
    policy = ParticlePolicy(num_particles=num_particles, horizon=planning_horizon)
    parameter = next(model.parameters())
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    episode_records = []
    env = make_env(env_id, render_mode="rgb_array")
    try:
        for episode in range(episodes):
            record = run_online_episode(
                model_name,
                model,
                env,
                policy,
                max_steps,
                particle_updates,
                particle_sigma,
                particle_temperature,
                seed + episode,
                generator,
                record_frames=record_video,
            )
            episode_records.append(record)
            print(
                f"[{model_name}] episode={episode + 1}/{episodes} "
                f"return={record['return']:.3f} length={record['length']}",
                flush=True,
            )
    finally:
        env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    returns = [record["return"] for record in episode_records]
    metrics_path = output_dir / "online_returns.png"
    _returns_image(returns).save(metrics_path)
    artifacts = {"returns": str(metrics_path)}
    if record_video:
        for episode, record in enumerate(episode_records, start=1):
            video_path = output_dir / f"online_episode_{episode:03d}.gif"
            frames = record["frames"]
            frames[0].save(
                video_path,
                save_all=True,
                append_images=frames[1:],
                duration=max(1, round(1000 / fps)),
                loop=0,
                disposal=2,
            )
            artifacts[f"video_episode_{episode}"] = str(video_path)
    return {
        **_return_summary(returns),
        "episode_lengths": [record["length"] for record in episode_records],
        "artifacts": artifacts,
    }


def run_online_episode(
    model_name: str,
    model: Any,
    env: Any,
    policy: ParticlePolicy,
    max_steps: int,
    particle_updates: int,
    particle_sigma: float,
    particle_temperature: float,
    seed: int,
    generator: torch.Generator,
    *,
    record_frames: bool,
) -> dict[str, Any]:
    reset_env(env, seed)
    images = [_render_model_image(env, model.config.observation_shape)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    frames: list[Image.Image] = []
    predicted_rewards: list[float] = []

    for timestep in range(max_steps):
        observation_history, observation_valid = _observation_history(images, model)
        action_history, action_valid = _action_history(actions, model)
        with torch.inference_mode():
            latent = model.encode_ema(
                observation_history, observation_valid, action_history, action_valid
            )
            action, predicted_reward = select_particle_action(
                model,
                latent,
                policy,
                particle_updates,
                particle_sigma,
                particle_temperature,
                generator,
            )
        action_array = action.detach().cpu().numpy().reshape(model.config.action_shape)
        _, reward, terminated, truncated, _ = env.step(action_array)
        actions.append(action_array.astype(np.float32, copy=False))
        rewards.append(float(reward))
        predicted_rewards.append(float(predicted_reward.item()))
        images.append(_render_model_image(env, model.config.observation_shape))
        if record_frames:
            frames.append(
                _online_video_frame(
                    env,
                    timestep,
                    action_array,
                    rewards,
                    predicted_rewards,
                )
            )
        if terminated or truncated:
            break

    record = {
        "return": float(sum(rewards)),
        "length": len(rewards),
        "frames": frames,
    }
    return record


def select_particle_action(
    model: Any,
    latent: torch.Tensor,
    policy: ParticlePolicy,
    particle_updates: int,
    particle_sigma: float,
    particle_temperature: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = next(model.parameters())
    particles = policy.init_particles(
        latent.shape[0], device=parameter.device, dtype=parameter.dtype, generator=generator
    )
    for update_index in range(particle_updates):
        scores, _ = score_particles(
            model,
            latent,
            particles,
        )
        particles = policy.update_particles(
            particles,
            scores,
            sigma=_particle_sigma(particle_sigma, policy.horizon, update_index),
            temperature=particle_temperature,
            generator=generator,
        )
    scores, rewards = score_particles(
        model,
        latent,
        particles,
    )
    best = scores.argmax(dim=1)
    batch_indices = torch.arange(len(best), device=best.device)
    action = particles[batch_indices, best, 0]
    return action, rewards[batch_indices, best]


def _particle_sigma(initial_sigma: float, horizon: int, update_index: int) -> float:
    return max(1.0 / horizon, initial_sigma - update_index / horizon)


def score_particles(
    model: Any,
    latent: torch.Tensor,
    particles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if particles.ndim != 4:
        raise ValueError("particles must have shape [batch, particles, horizon, action_dim].")
    batch_size, num_particles, horizon, action_dim = particles.shape
    if horizon == 0:
        raise ValueError("particle sequence horizon must be positive.")
    repeated_latent = latent[:, None].expand(-1, num_particles, -1).reshape(
        batch_size * num_particles, -1
    )
    scores = torch.zeros(batch_size, num_particles, device=latent.device, dtype=latent.dtype)
    first_rewards = None
    for timestep in range(horizon):
        flat_actions = particles[:, :, timestep].reshape(
            batch_size * num_particles, action_dim
        )
        rewards = model.ema_heads.reward(repeated_latent, flat_actions).reshape(
            batch_size, num_particles
        )
        if first_rewards is None:
            first_rewards = rewards
        scores = scores + rewards
        repeated_latent = model.predict_next_ema(repeated_latent, flat_actions)
    return scores, first_rewards


def evaluate_random_policy(env_id: str, episodes: int, max_steps: int, seed: int) -> list[float]:
    returns = []
    env = make_env(env_id)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            reset_env(env, episode_seed)
            env.action_space.seed(episode_seed)
            total = 0.0
            for _ in range(max_steps):
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    return returns


def _validate_policy_environment(env_id: str, model: Any) -> None:
    if env_id != "Pendulum-v1":
        raise ValueError("ParticlePolicy currently supports only Pendulum-v1.")
    if model.config.action_shape != (1,):
        raise ValueError("Pendulum online evaluation requires checkpoint action_shape=(1,).")


def _render_model_image(env: Any, observation_shape: tuple[int, int, int]) -> np.ndarray:
    channels, height, width = observation_shape
    image = np.asarray(env.render())
    if image.ndim != 3 or image.shape[2] != channels or image.dtype != np.uint8:
        raise ValueError("Environment rendering does not match checkpoint image channels.")
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
    )


def _observation_history(images: list[np.ndarray], model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = next(model.parameters())
    selected = images[-OBS_HISTORY_LEN:]
    padded = np.zeros((OBS_HISTORY_LEN, *selected[0].shape), dtype=np.uint8)
    padded[-len(selected) :] = selected
    tensor = torch.as_tensor(padded, device=parameter.device, dtype=parameter.dtype)
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    valid = torch.zeros((1, OBS_HISTORY_LEN), dtype=torch.bool, device=parameter.device)
    valid[:, -len(selected) :] = True
    return tensor, valid


def _action_history(actions: list[np.ndarray], model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    parameter = next(model.parameters())
    selected = actions[-ACTION_HISTORY_LEN:]
    history = torch.zeros(
        (1, ACTION_HISTORY_LEN, model.config.action_dim),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    valid = torch.zeros((1, ACTION_HISTORY_LEN), dtype=torch.bool, device=parameter.device)
    if selected:
        history[0, -len(selected) :] = torch.as_tensor(
            np.asarray(selected).reshape(-1, model.config.action_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        valid[:, -len(selected) :] = True
    return history, valid


def _online_video_frame(
    env: Any,
    timestep: int,
    action: np.ndarray,
    rewards: list[float],
    predicted_rewards: list[float],
) -> Image.Image:
    rendered = Image.fromarray(np.asarray(env.render())).convert("RGB").resize((500, 500))
    canvas = Image.new("RGB", (760, 500), "white")
    canvas.paste(rendered, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((520, 30), "ONLINE ENVIRONMENT", fill=(0, 100, 60))
    draw.text((520, 70), f"step: {timestep}", fill="black")
    draw.text((520, 100), f"action: {float(action.item()):.3f}", fill="black")
    draw.text((520, 130), f"reward: {rewards[-1]:.3f}", fill="black")
    draw.text((520, 160), f"pred reward: {predicted_rewards[-1]:.3f}", fill="black")
    draw.text((520, 220), f"return: {sum(rewards):.3f}", fill="black")
    _draw_series(draw, (520, 280, 740, 460), rewards, (30, 100, 210))
    draw.text((520, 465), "online reward", fill="black")
    return canvas


def _returns_image(returns: list[float]) -> Image.Image:
    canvas = Image.new("RGB", (800, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "Online episode returns", fill="black")
    _draw_series(draw, (60, 50, 780, 370), returns, (20, 120, 80))
    draw.text((60, 385), f"mean={np.mean(returns):.3f} std={np.std(returns):.3f}", fill="black")
    return canvas


def _draw_series(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    values: list[float],
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = bounds
    low, high = min(values), max(values)
    if low == high:
        high = low + 1.0
    draw.rectangle(bounds, outline=(170, 170, 170))
    points = []
    for index, value in enumerate(values):
        x = left + index * (right - left) / max(1, len(values) - 1)
        y = bottom - (value - low) * (bottom - top) / (high - low)
        points.append((round(x), round(y)))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    elif points:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((left + 2, top + 2), f"{high:.3g}", fill=(80, 80, 80))
    draw.text((left + 2, bottom - 12), f"{low:.3g}", fill=(80, 80, 80))


def _return_summary(returns: list[float]) -> dict[str, Any]:
    return {
        "returns": returns,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


if __name__ == "__main__":
    main()
