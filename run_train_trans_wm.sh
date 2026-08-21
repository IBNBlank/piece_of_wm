#!/usr/bin/env bash
###############################################################################
# Train Trans-WM from sequence-preserving image rollouts.
#
# Usage:
#   ./run_train_trans_wm.sh
#   ROLLOUTS=500 BATCH_SIZE=16 DEVICE=cuda ./run_train_trans_wm.sh
#
# Tunables (env vars):
#   PYTHON              : Python interpreter
#   DATA_DIR            : rollout dataset directory (default: dataset/pendulum-random)
#   OUTPUT_DIR          : checkpoints and metrics directory (default: runs/trans_wm)
#   NUM_ENVS            : environments per collected rollout (default: 10)
#   MAX_STEPS           : maximum steps per environment episode (default: 200)
#   ROLLOUTS            : number of rollout training units (default: 500)
#   BATCH_SIZE          : sampled transitions per optimization batch (default: 32)
#   REPLAY_CAPACITY     : rollout files held in RAM (default: complete dataset)
#   SAMPLE_ROLLOUTS     : rollout batches combined per update (default: 2)
#   VALUE_ROLLOUTS      : current-policy online episodes per update (default: 2)
#   VALUE_EPOCHS        : value-only epochs per online episode (default: 1)
#   PARTICLE_UPDATES    : particle resampling iterations (default: 4)
#   PARTICLE_SIGMA      : particle perturbation standard deviation (default: 0.1)
#   PLANNING_HORIZON    : model planning horizon (default: 1)
#   EVALUATION_ROLLOUTS : online evaluation episodes per checkpoint (default: 10)
#   VALIDATION_BATCH_SIZE: fixed validation transition sample size (default: 256)
#   EPOCHS_PER_ROLLOUT  : optimization epochs for each rollout (default: 10)
#   CHECKPOINT_ROLLOUTS : checkpoint frequency in rollouts (default: 10)
#   EARLY_STOP_PATIENCE : validation checks without improvement (default: 4)
#   SEED                 : random seed (default: 0)
#   DEVICE               : torch device; empty selects CUDA when available
#   RESUME               : checkpoint path to resume (default: empty)
#   LEARNING_RATE        : AdamW learning rate (default: 1e-4)
#   GRAD_CLIP_NORM       : gradient clipping norm (default: 10.0)
#   EXTRA_ARGS           : additional arguments forwarded to trans_wm.train
###############################################################################
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -n "${PYTHON:-}" ]; then
	:
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PYTHON="${VIRTUAL_ENV}/bin/python"
else
	PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi

if [ ! -x "${PYTHON}" ]; then
	echo "[run_train_trans_wm] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_train_trans_wm] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/trans_wm}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
ROLLOUTS="${ROLLOUTS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-}"
SAMPLE_ROLLOUTS="${SAMPLE_ROLLOUTS:-2}"
VALUE_ROLLOUTS="${VALUE_ROLLOUTS:-2}"
VALUE_EPOCHS="${VALUE_EPOCHS:-1}"
PARTICLE_UPDATES="${PARTICLE_UPDATES:-4}"
PARTICLE_SIGMA="${PARTICLE_SIGMA:-0.1}"
PLANNING_HORIZON="${PLANNING_HORIZON:-1}"
EVALUATION_ROLLOUTS="${EVALUATION_ROLLOUTS:-10}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-256}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-10}"
CHECKPOINT_ROLLOUTS="${CHECKPOINT_ROLLOUTS:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
RESUME="${RESUME:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
	--data-dir "${DATA_DIR}"
	--output-dir "${OUTPUT_DIR}"
	--num-envs "${NUM_ENVS}"
	--max-steps "${MAX_STEPS}"
	--rollouts "${ROLLOUTS}"
	--batch-size "${BATCH_SIZE}"
	--sample-rollouts "${SAMPLE_ROLLOUTS}"
	--value-rollouts "${VALUE_ROLLOUTS}"
	--value-epochs "${VALUE_EPOCHS}"
	--particle-updates "${PARTICLE_UPDATES}"
	--particle-sigma "${PARTICLE_SIGMA}"
	--planning-horizon "${PLANNING_HORIZON}"
	--evaluation-rollouts "${EVALUATION_ROLLOUTS}"
	--validation-batch-size "${VALIDATION_BATCH_SIZE}"
	--epochs-per-rollout "${EPOCHS_PER_ROLLOUT}"
	--checkpoint-rollouts "${CHECKPOINT_ROLLOUTS}"
	--early-stop-patience "${EARLY_STOP_PATIENCE}"
	--seed "${SEED}"
	--learning-rate "${LEARNING_RATE}"
	--grad-clip-norm "${GRAD_CLIP_NORM}"
)
if [ -n "${REPLAY_CAPACITY}" ]; then
	args+=(--replay-capacity "${REPLAY_CAPACITY}")
fi
if [ -n "${DEVICE}" ]; then
	args+=(--device "${DEVICE}")
fi
if [ -n "${RESUME}" ]; then
	args+=(--resume "${RESUME}")
fi

echo "[run_train_trans_wm] data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"
echo "[run_train_trans_wm] rollouts=${ROLLOUTS} num_envs=${NUM_ENVS} max_steps=${MAX_STEPS} epochs_per_rollout=${EPOCHS_PER_ROLLOUT} batch_size=${BATCH_SIZE} sample_rollouts=${SAMPLE_ROLLOUTS} value_rollouts=${VALUE_ROLLOUTS} evaluation_rollouts=${EVALUATION_ROLLOUTS} replay_capacity=${REPLAY_CAPACITY:-all} device=${DEVICE:-auto} seed=${SEED}"

# shellcheck disable=SC2086
"${PYTHON}" -m trans_wm.train "${args[@]}" ${EXTRA_ARGS} "$@"
