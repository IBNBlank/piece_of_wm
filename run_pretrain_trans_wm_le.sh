#!/usr/bin/env bash
###############################################################################
# Pretrain Trans-WM-LE from offline replay data.
#
# Usage:
#   ./run_pretrain_trans_wm_le.sh
#   DEVICE=cuda EPOCHS=10 ./run_pretrain_trans_wm_le.sh
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
	echo "[run_pretrain_trans_wm_le] error: Python executable not found: ${PYTHON}" >&2
	exit 1
fi

DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/trans_wm_le_pretrain}"
NUM_ENVS="${NUM_ENVS:-20}"
MAX_STEPS="${MAX_STEPS:-200}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
PLANNING_HORIZON="${PLANNING_HORIZON:-16}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
RESUME="${RESUME:-}"
TARGET_EMA="${TARGET_EMA:-0.99}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
JEPA_WEIGHT="${JEPA_WEIGHT:-1.0}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.2}"
SIGREG_PROJECTIONS="${SIGREG_PROJECTIONS:-256}"
SIGREG_FREQUENCIES="${SIGREG_FREQUENCIES:-17}"
SIGREG_MAX_FREQUENCY="${SIGREG_MAX_FREQUENCY:-5.0}"
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
	--planning-horizon "${PLANNING_HORIZON}"
	--seed "${SEED}"
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
if [ -n "${REPLAY_CAPACITY}" ]; then
	args+=(--replay-capacity "${REPLAY_CAPACITY}")
fi
if [ -n "${DEVICE}" ]; then
	args+=(--device "${DEVICE}")
fi
if [ -n "${RESUME}" ]; then
	args+=(--resume "${RESUME}")
fi

echo "[run_pretrain_trans_wm_le] data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"
echo "[run_pretrain_trans_wm_le] epochs=${EPOCHS} batch_size=${BATCH_SIZE} planning_horizon=${PLANNING_HORIZON} device=${DEVICE:-auto}"
# shellcheck disable=SC2086
"${PYTHON}" -m trans_wm_le.train "${args[@]}" ${EXTRA_ARGS} "$@"
