# piece_of_wm

PETS (Probabilistic Ensembles with Trajectory Sampling) world-model example,
split from `01_pets_pendulum.ipynb` into command-line modules.

```bash
./venv.sh
source .venv/bin/activate

# Save random real-environment transitions.
./run_collect_data.sh --env-id Pendulum-v1 --episodes 10 --output-dir dataset/pendulum-random

# Train only the dynamics ensemble from saved data.
./pets/run_train_offline.sh --data-dir dataset/pendulum-random --output-dir runs/pets-offline

# Alternate model fitting, CEM/MPC control, and real data collection.
./pets/run_train_online.sh --initial-episodes 1 --trials 6 --output-dir runs/pets-online

# Evaluate a checkpoint and save an MP4 under runs/pets-eval/videos/.
./pets/run_eval.sh --model-dir runs/pets-offline --output-dir runs/pets-eval
```

`utils/env.py` owns Gymnasium setup and space validation. `utils/data.py` owns
the mbrl replay buffer and portable dataset files. `utils/common.py` owns
seeding, logging, plots, and checkpoints. `pets/model.py` owns the PETS model,
model environment, CEM planner, and model-training loop.

The current `GaussianMLP` PETS model is intentionally limited to vector
observations, so it runs on `Pendulum-v1`. `collect_data.py` can also store
continuous-control image observations such as `CarRacing-v2`, but PETS training
will report that an image encoder/latent dynamics model is required rather than
flattening pixels silently.
