#!/usr/bin/env bash
###############################################################################
# Train Trans-WM-LE from sequence-preserving image rollouts.
#
# Usage:
#   ./run_train_trans_wm_le.sh
#   ROLLOUTS=500 BATCH_SIZE=16 DEVICE=cuda ./run_train_trans_wm_le.sh
#   JEPA_WEIGHT=1.0 SIGREG_WEIGHT=0.1 ./run_train_trans_wm_le.sh
#
# Tunables (env vars):
#   PYTHON                : Python interpreter
#   DATA_DIR              : rollout dataset directory (default: dataset/pendulum-random)
#   OUTPUT_DIR            : checkpoints and metrics directory (default: runs/trans_wm_le)
#   NUM_ENVS              : environments per collected rollout (default: 10)
#   MAX_STEPS             : maximum steps per environment episode (default: 200)
#   ROLLOUTS              : number of rollout training units (default: 100)
#   BATCH_SIZE            : sampled transitions per optimization batch (default: 8)
#   EPOCHS_PER_ROLLOUT    : optimization epochs for each rollout (default: 10)
#   CHECKPOINT_ROLLOUTS   : checkpoint frequency in rollouts (default: 10)
#   SEED                  : random seed (default: 0)
#   DEVICE                : torch device; empty selects CUDA when available
#   RESUME                : checkpoint path to resume (default: empty)
#   OBSERVATION_DIM       : CNN observation representation size (default: 128)
#   MODEL_DIM             : hidden size of latent and dynamics MLPs (default: 256)
#   CNN_CHANNELS          : comma-separated CNN channels (default: 32,64,128)
#   GAMMA                 : value discount factor (default: 0.99)
#   TARGET_EMA            : EMA target decay (default: 0.99)
#   LEARNING_RATE         : AdamW learning rate (default: 3e-4)
#   WEIGHT_DECAY          : AdamW weight decay (default: 1e-5)
#   GRAD_CLIP_NORM        : gradient clipping norm (default: 100.0)
#   JEPA_WEIGHT           : JEPA latent prediction loss weight (default: 1.0)
#   SIGREG_WEIGHT         : SIGReg regularization weight (default: 1.0)
#   SIGREG_PROJECTIONS    : random SIGReg projections (default: 256)
#   SIGREG_FREQUENCIES    : frequencies per SIGReg projection (default: 17)
#   SIGREG_MAX_FREQUENCY  : maximum SIGReg frequency (default: 5.0)
#   EXTRA_ARGS            : additional arguments forwarded to trans_wm_le.train
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
	echo "[run_train_trans_wm_le] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_train_trans_wm_le] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/trans_wm_le}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
ROLLOUTS="${ROLLOUTS:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-10}"
CHECKPOINT_ROLLOUTS="${CHECKPOINT_ROLLOUTS:-10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
RESUME="${RESUME:-}"
OBSERVATION_DIM="${OBSERVATION_DIM:-128}"
MODEL_DIM="${MODEL_DIM:-256}"
CNN_CHANNELS="${CNN_CHANNELS:-32,64,128}"
GAMMA="${GAMMA:-0.99}"
TARGET_EMA="${TARGET_EMA:-0.99}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100.0}"
JEPA_WEIGHT="${JEPA_WEIGHT:-1.0}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-1.0}"
SIGREG_PROJECTIONS="${SIGREG_PROJECTIONS:-256}"
SIGREG_FREQUENCIES="${SIGREG_FREQUENCIES:-17}"
SIGREG_MAX_FREQUENCY="${SIGREG_MAX_FREQUENCY:-5.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
	--data-dir "${DATA_DIR}"
	--output-dir "${OUTPUT_DIR}"
	--num-envs "${NUM_ENVS}"
	--max-steps "${MAX_STEPS}"
	--rollouts "${ROLLOUTS}"
	--batch-size "${BATCH_SIZE}"
	--epochs-per-rollout "${EPOCHS_PER_ROLLOUT}"
	--checkpoint-rollouts "${CHECKPOINT_ROLLOUTS}"
	--seed "${SEED}"
	--observation-dim "${OBSERVATION_DIM}"
	--model-dim "${MODEL_DIM}"
	--cnn-channels "${CNN_CHANNELS}"
	--gamma "${GAMMA}"
	--target-ema "${TARGET_EMA}"
	--learning-rate "${LEARNING_RATE}"
	--weight-decay "${WEIGHT_DECAY}"
	--grad-clip-norm "${GRAD_CLIP_NORM}"
	--jepa-weight "${JEPA_WEIGHT}"
	--sigreg-weight "${SIGREG_WEIGHT}"
	--sigreg-projections "${SIGREG_PROJECTIONS}"
	--sigreg-frequencies "${SIGREG_FREQUENCIES}"
	--sigreg-max-frequency "${SIGREG_MAX_FREQUENCY}"
)
if [ -n "${DEVICE}" ]; then
	args+=(--device "${DEVICE}")
fi
if [ -n "${RESUME}" ]; then
	args+=(--resume "${RESUME}")
fi

echo "[run_train_trans_wm_le] data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"
echo "[run_train_trans_wm_le] rollouts=${ROLLOUTS} num_envs=${NUM_ENVS} max_steps=${MAX_STEPS} epochs_per_rollout=${EPOCHS_PER_ROLLOUT} batch_size=${BATCH_SIZE} device=${DEVICE:-auto} seed=${SEED}"
echo "[run_train_trans_wm_le] observation_dim=${OBSERVATION_DIM} model_dim=${MODEL_DIM} jepa_weight=${JEPA_WEIGHT} sigreg_weight=${SIGREG_WEIGHT}"

# shellcheck disable=SC2086
"${PYTHON}" -m trans_wm_le.train "${args[@]}" ${EXTRA_ARGS} "$@"
