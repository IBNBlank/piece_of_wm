"""Train Dreamer v1 with three-frame observations."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from dreamer_like import DreamerV1, WorldModelConfig
from utils.replay_buffer import OfflineRolloutDataset

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, default=Path('runs/dreamer_like/checkpoint.pt'))
    a = p.parse_args()
    data = OfflineRolloutDataset(a.data_dir)
    sample = data.sample(1)
    if sample.images is None:
        raise ValueError("Dreamer requires image rollouts.")
    config = WorldModelConfig(
        observation_shape=(sample.images.shape[-1], sample.images.shape[-3], sample.images.shape[-2]),
        action_shape=tuple(sample.action.shape[2:]),
    )
    model = DreamerV1(config).to(a.device); optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for _ in range(a.steps):
        batch = data.sample(a.batch_size); horizon = min(a.horizon, batch.action.shape[1])
        images = torch.as_tensor(batch.images[:, :horizon + 1], device=a.device, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0
        histories, history_masks = [], []
        for t in range(horizon + 1):
            padding = max(0, 3 - t)
            histories.append(torch.cat((torch.zeros_like(images[:, :padding]), images[:, max(0, t - 2):t + 1]), dim=1))
            history_masks.append(torch.tensor([False] * padding + [True] * (3 - padding), device=a.device))
        frames = torch.stack(histories[:horizon], dim=1)
        masks = torch.stack(history_masks[:horizon]).expand(frames.shape[0], -1, -1)
        actions = torch.as_tensor(batch.action[:, :horizon], device=a.device, dtype=torch.float32).flatten(start_dim=2)
        rewards = torch.as_tensor(batch.reward[:, :horizon, None], device=a.device, dtype=torch.float32)
        loss = model.loss(frames, masks, actions, rewards).total
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    a.output.parent.mkdir(parents=True, exist_ok=True); torch.save({'model': model.state_dict(), 'model_config': config.__dict__}, a.output)

if __name__ == '__main__': main()
