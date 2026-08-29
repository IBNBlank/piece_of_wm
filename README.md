# piece_of_wm

World-model data collection utilities with a sequence-preserving replay buffer.

```bash
./venv.sh
source .venv/bin/activate

# Each rollout file contains NUM_ENVS complete episodes and optional 128x128 images.
NUM_ENVS=4 ROLLOUTS=100 ./run_collect_data.sh --env-id FetchPickAndPlace-v4 --output-dir dataset/fetch-pick-and-place-random
```

`collect_data.py` writes complete episode batches. Offline pretraining reads
those files directly into RAM without copying them into the run directory.
Formal online training uses the persistent replay buffer under
`runs/.../replay_buffer` for restart recovery.

Each rollout stores `obs` with shape `(NUM_ENVS, T + 1, ...)`, transition arrays
with shape `(NUM_ENVS, T, ...)`, and `lengths` to mask padded time steps.

## Train Dreamer-like

Pretrain the world-model components before multi-step reward planning:

```bash
./run_pretrain_dreamer_like.sh
# Or: ./run_pretrain_tdmpc_like.sh
```

Run pretraining, formal training, and final online evaluation sequentially with
one command:

```bash
DEVICE=cuda PRETRAIN_EPOCHS=100 TRAIN_ROLLOUTS=500 \
./run_integrate_dreamer_like.sh

# TD-MPC-like:
DEVICE=cuda PRETRAIN_EPOCHS=100 TRAIN_ROLLOUTS=500 \
./run_integrate_tdmpc_like.sh
```

The integrated runners start each stage from scratch by default. Set
`PRETRAIN_RESUME` or `TRAIN_RESUME` explicitly to resume a checkpoint, and use a
new output directory to keep a fresh run separate from previous artifacts.

Pretraining uses replay updates only. It does not create a Gym environment or
run the particle policy. Validation loss selects
`checkpoint_best.pt`; the two newest numbered checkpoints are retained for
`RESUME`-based pretraining continuation. `EPOCHS` controls complete passes over
all valid offline transitions; transitions are shuffled and visited exactly
once per epoch, including the final partial minibatch. `CHECKPOINT_EPOCHS`
controls validation and checkpoint frequency.

Start formal training from the best pretraining checkpoint with:

```bash
PRETRAINED_CHECKPOINT=runs/dreamer_like_pretrain/checkpoint_best.pt \
./run_train_dreamer_like.sh

# TD-MPC-like:
PRETRAINED_CHECKPOINT=runs/tdmpc_like_pretrain/checkpoint_best.pt \
./run_train_tdmpc_like.sh
```

`PRETRAINED_CHECKPOINT` restores only model weights; the
formal optimizer, rollout counter, RNG streams, and best-return tracking start
fresh. `RESUME` instead restores the complete state of the same phase. The two
options are mutually exclusive, and the model configuration must match the checkpoint.

To train the CNN/Transformer world model without a
pretraining phase, run:

```bash
./run_train_dreamer_like.sh
```

The default input is `dataset/fetch-pick-and-place-random`, and checkpoints and metrics are
written to `runs/dreamer_like`. Common overrides are supplied through environment
variables:

```bash
DATA_DIR=dataset/fetch-pick-and-place-random \
OUTPUT_DIR=runs/dreamer_like-fetch \
ROLLOUTS=500 NUM_ENVS=10 MAX_STEPS=200 \
BATCH_SIZE=16 DEVICE=cuda \
./run_train_dreamer_like.sh
```

`NUM_ENVS` and `MAX_STEPS` are checked against `dataset.json` before training,
so the training configuration cannot silently disagree with the collected
rollout layout.

The script samples bounded transition batches from the sequence rollouts, so it
does not materialize all three-frame windows for an entire dataset in memory. To
continue from a checkpoint, set `RESUME` to a checkpoint file or the run
directory and keep the matching model/training arguments in `EXTRA_ARGS`. A run
directory automatically selects its newest numbered checkpoint:

```bash
RESUME=runs/dreamer_like ROLLOUTS=1000 \
./run_train_dreamer_like.sh
```

At startup, formal online training imports the rollout files into a RAM-resident
`RolloutReplayBuffer` and mirrors them under `OUTPUT_DIR/replay_buffer`. Each
training unit combines two randomly selected complete rollout batches by
default (`SAMPLE_ROLLOUTS=2`) before sampling transitions. Periodic evaluation
uses a fixed transition sample.

Replay updates train the encoder, dynamics, and action-conditioned reward model
`R(z_t, a_t)` for both models. Particle planning scores are the sums of multi-step
predicted rewards, without a critic or terminal value bootstrap. Every 10 training rollouts, the
policy is evaluated over 10 separate episodes. The two newest numbered
checkpoints are retained for resuming, while `checkpoint_best.pt` retains the
highest evaluation return.

World-model updates use the same `PLANNING_HORIZON` as policy evaluation. Its
default is 8. From
each sampled starting state, dynamics is recursively unrolled over consecutive
dataset actions. Every valid prediction step contributes equally to the
normalized loss; episode-tail padding is masked and no discount is applied.

## Online evaluation

`run_eval.sh` evaluates the checkpoint by resetting and stepping the real
Gymnasium environment. Candidate actions are selected with the particle policy;
the rollout dataset is not used by evaluation.

```bash
MODEL=tdmpc_like \
TDMPC_LIKE_CHECKPOINT=runs/tdmpc_like/checkpoint_best.pt \
EPISODES=5 DEVICE=cuda \
./run_eval.sh
```

Results include per-episode online returns, a same-seed random-policy baseline,
an online return plot, and a GIF recorded from the first environment episode.
The default `PLANNING_HORIZON` is eight model steps and is shared by recursive
world-model training and particle planning.
