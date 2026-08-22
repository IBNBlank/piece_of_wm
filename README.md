# piece_of_wm

World-model data collection utilities with a sequence-preserving replay buffer.

```bash
./venv.sh
source .venv/bin/activate

# Each rollout file contains NUM_ENVS complete episodes and optional 128x128 images.
NUM_ENVS=4 ROLLOUTS=100 ./run_collect_data.sh --env-id Pendulum-v1 --output-dir dataset/pendulum-random
```

`collect_data.py` writes complete episode batches. Offline pretraining reads
those files directly into RAM without copying them into the run directory.
Formal online training uses the persistent replay buffer under
`runs/.../replay_buffer` for restart recovery.

Each rollout stores `obs` with shape `(NUM_ENVS, T + 1, ...)`, transition arrays
with shape `(NUM_ENVS, T, ...)`, and `lengths` to mask padded time steps.

## Train Trans-WM

Pretrain the world-model components before enabling online critic training:

```bash
./run_pretrain_trans_wm.sh
# Or: ./run_pretrain_trans_wm_le.sh
```

Run pretraining and formal training sequentially with one command:

```bash
DEVICE=cuda PRETRAIN_EPOCHS=100 TRAIN_ROLLOUTS=500 NUM_CRITICS=5 \
./run_integrate_trans_wm.sh

# Trans-WM-LE:
DEVICE=cuda PRETRAIN_EPOCHS=100 TRAIN_ROLLOUTS=500 NUM_CRITICS=5 \
./run_integrate_trans_wm_le.sh
```

Pretraining uses replay updates only. It does not create a Gym environment,
run the particle policy, or update the critic ensemble. Validation loss selects
`checkpoint_best.pt`; the two newest numbered checkpoints are retained for
`RESUME`-based pretraining continuation. `EPOCHS` controls complete passes over
all valid offline transitions; transitions are shuffled and visited exactly
once per epoch, including the final partial minibatch. `CHECKPOINT_EPOCHS`
controls validation and checkpoint frequency.

Start formal training from the best pretraining checkpoint with:

```bash
PRETRAINED_CHECKPOINT=runs/trans_wm_pretrain/checkpoint_best.pt \
./run_train_trans_wm.sh

# Trans-WM-LE:
PRETRAINED_CHECKPOINT=runs/trans_wm_le_pretrain/checkpoint_best.pt \
./run_train_trans_wm_le.sh
```

`PRETRAINED_CHECKPOINT` restores the model and world optimizer, but starts the
formal rollout counter, critic optimizer, RNG streams, and best-return tracking
fresh. `RESUME` instead restores the complete state of the same phase. The two
options are mutually exclusive, and model/training configuration must match
the checkpoint.

To train the CNN/Transformer world model and critic together without a
pretraining phase, run:

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

At startup, formal online training imports the rollout files into a RAM-resident
`RolloutReplayBuffer` and mirrors them under `OUTPUT_DIR/replay_buffer`. Each
training unit combines two randomly selected complete rollout batches by
default (`SAMPLE_ROLLOUTS=2`) before sampling transitions. Periodic evaluation
uses a fixed transition sample.

Replay updates train the encoder, dynamics, and action-conditioned reward model
`R(z_t, a_t)`. They do not update the value head. Each training unit also runs
the current particle policy in the real Gymnasium environment (two complete
episodes by default, `VALUE_ROLLOUTS=2`) and trains only `V(z_t)` against the
TD(lambda) return. Its one-step bootstrap latent is predicted by the EMA world
model and evaluated by the worst EMA critic. Critic training also uses the
ensemble minimum. `GAMMA` and `LAMBDA_RETURN` both default to `0.95`. Planning
uses `GAMMA`, the ensemble mean, and the
shell-configured `NUM_PARTICLES` and `PLANNING_HORIZON`. Every 10 training rollouts, the
policy is evaluated over 10 separate episodes. The two newest numbered
checkpoints are retained for resuming, while `checkpoint_best.pt` retains the
highest evaluation return.

The ensemble critic is an architecture change. Older single-critic checkpoints
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
