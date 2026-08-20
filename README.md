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
continue from a checkpoint, set `RESUME` and keep the matching model/training
arguments in `EXTRA_ARGS`:

```bash
RESUME=runs/trans_wm/checkpoint.pt ROLLOUTS=1000 \
./run_train_trans_wm.sh
```
