# piece_of_wm

World-model data collection utilities with sequence-preserving replay.

```bash
./venv.sh
source .venv/bin/activate
NUM_ENVS=4 ROLLOUTS=100 ./run_collect_data.sh \
  --env-id FetchPickAndPlace-v4 \
  --output-dir dataset/fetch-pick-and-place-random
```

## Dreamer v1

`dreamer_like` contains only the multi-frame Dreamer v1 learner. Train it with:

```bash
DATA_DIR=dataset/fetch-pick-and-place-random DEVICE=cuda \
./run_train_dreamer_like.sh
```

See [dreamer_like/README.md](dreamer_like/README.md) for tensor layouts and
the RSSM training semantics.

## TD-MPC-like

The separate `tdmpc_like` implementation retains its own training and
evaluation scripts:

```bash
./run_integrate_tdmpc_like.sh
```
