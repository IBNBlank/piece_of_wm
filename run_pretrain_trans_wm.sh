#!/usr/bin/env bash
###############################################################################
# Pretrain Trans-WM without environment rollouts or critic updates.
#
# Usage:
#   ./run_pretrain_trans_wm.sh
#   DEVICE=cuda EPOCHS=10 ./run_pretrain_trans_wm.sh
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
	echo "[run_pretrain_trans_wm] error: Python executable not found: ${PYTHON}" >&2
	exit 1
fi

DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/trans_wm_pretrain}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
RESUME="${RESUME:-}"
LATENT_DIM="${LATENT_DIM:-128}"
MODEL_DIM="${MODEL_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-4}"
FEEDFORWARD_DIM="${FEEDFORWARD_DIM:-512}"
CNN_CHANNELS="${CNN_CHANNELS:-32,64,128}"
DROPOUT="${DROPOUT:-0.0}"
NUM_CRITICS="${NUM_CRITICS:-2}"
GAMMA="${GAMMA:-0.95}"
TARGET_EMA="${TARGET_EMA:-0.99}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
VAE_RECONSTRUCTION_WEIGHT="${VAE_RECONSTRUCTION_WEIGHT:-1.0}"
VAE_KL_WEIGHT="${VAE_KL_WEIGHT:-1e-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
args=(
	--pretrain
	--data-dir "${DATA_DIR}"
	--output-dir "${OUTPUT_DIR}"
	--num-envs "${NUM_ENVS}"
	--max-steps "${MAX_STEPS}"
	--epochs "${EPOCHS}"
	--batch-size "${BATCH_SIZE}"
	--checkpoint-epochs "${CHECKPOINT_EPOCHS}"
	--seed "${SEED}"
	--latent-dim "${LATENT_DIM}"
	--model-dim "${MODEL_DIM}"
	--num-layers "${NUM_LAYERS}"
	--num-heads "${NUM_HEADS}"
	--feedforward-dim "${FEEDFORWARD_DIM}"
	--cnn-channels "${CNN_CHANNELS}"
	--dropout "${DROPOUT}"
	--num-critics "${NUM_CRITICS}"
	--gamma "${GAMMA}"
	--target-ema "${TARGET_EMA}"
	--learning-rate "${LEARNING_RATE}"
	--weight-decay "${WEIGHT_DECAY}"
	--grad-clip-norm "${GRAD_CLIP_NORM}"
	--vae-reconstruction-weight "${VAE_RECONSTRUCTION_WEIGHT}"
	--vae-kl-weight "${VAE_KL_WEIGHT}"
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

echo "[run_pretrain_trans_wm] data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"
echo "[run_pretrain_trans_wm] epochs=${EPOCHS} batch_size=${BATCH_SIZE} device=${DEVICE:-auto}"
# shellcheck disable=SC2086
"${PYTHON}" -m trans_wm.train "${args[@]}" ${EXTRA_ARGS} "$@"
