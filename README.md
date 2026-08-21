# piece_of_wm

World-model data collection utilities with a sequence-preserving replay buffer.

```bash
./venv.sh
source .venv/bin/activate

# Each rollout file contains NUM_ENVS complete episodes and optional 128x128 images.
NUM_ENVS=4 ROLLOUTS=100 ./run_collect_data.sh --env-id Pendulum-v1 --output-dir dataset/pendulum-random
```

`collect_data.py` writes complete episode batches. `utils/replay_buffer.py`
keeps a fixed number of those batches in RAM and mirrors them under
`runs/.../replay_buffer` for restart recovery.

Each rollout stores `obs` with shape `(NUM_ENVS, T + 1, ...)`, transition arrays
with shape `(NUM_ENVS, T, ...)`, and `lengths` to mask padded time steps.

## Train Trans-WM

After collecting image rollouts, train the CNN/Transformer world model with:

```bash
./run_train_trans_wm.sh
```

The default input is `dataset/pendulum-random`, and checkpoints and metrics are
written to `runs/trans_wm`. Common overrides are supplied through environment
variables:

```bash
DATA_DIR=dataset/pendulum-random \
OUTPUT_DIR=runs/trans_wm-pendulum \
ROLLOUTS=500 NUM_ENVS=10 MAX_STEPS=200 \
BATCH_SIZE=16 DEVICE=cuda \
./run_train_trans_wm.sh
```

`NUM_ENVS` and `MAX_STEPS` are checked against `dataset.json` before training,
so the training configuration cannot silently disagree with the collected
rollout layout.

The script samples bounded transition batches from the sequence rollouts, so it
does not materialize all ten-frame windows for an entire dataset in memory. To
continue from a checkpoint, set `RESUME` to a checkpoint file or the run
directory and keep the matching model/training arguments in `EXTRA_ARGS`. A run
directory automatically selects its newest numbered checkpoint:

```bash
RESUME=runs/trans_wm ROLLOUTS=1000 \
./run_train_trans_wm.sh
```

At startup, training imports the rollout files into a RAM-resident
`RolloutReplayBuffer` and mirrors them under `OUTPUT_DIR/replay_buffer`. Each
training unit combines two randomly selected complete rollout batches by
default (`SAMPLE_ROLLOUTS=2`) before sampling transitions. Periodic evaluation
uses a fixed transition sample.

Replay updates train the encoder, dynamics, and action-conditioned reward model
`R(z_t, a_t)`. They do not update the value head. Each training unit also runs
the current particle policy in the real Gymnasium environment (two complete
episodes by default, `VALUE_ROLLOUTS=2`) and trains only `V(z_t)` against the
Monte Carlo return-to-go with `gamma=0.95`. Every 10 training rollouts, the
policy is evaluated over 10 separate episodes. The two newest numbered
checkpoints are retained for resuming, while `checkpoint_best.pt` retains the
highest evaluation return.

The action-conditioned reward head is an architecture change. Older checkpoints
are rejected explicitly and must be retrained from scratch.

## Online evaluation

`run_eval.sh` evaluates the checkpoint by resetting and stepping the real
Gymnasium environment. Candidate actions are selected with the particle policy;
the rollout dataset is not used by evaluation.

```bash
MODEL=trans_wm_le \
TRANS_WM_LE_CHECKPOINT=runs/trans_wm_le/checkpoint_best.pt \
EPISODES=5 DEVICE=cuda \
./run_eval.sh
```

Results include per-episode online returns, a same-seed random-policy baseline,
an online return plot, and a GIF recorded from the first environment episode.
The default planning horizon is one model step. Longer model planning must be
requested explicitly with `PLANNING_HORIZON` and should only be used after
multi-step prediction quality has been validated.
