#!/usr/bin/env bash
###############################################################################
# Offline PETS dynamics-model training from a saved replay buffer.
#
# Usage:
#   ./run_train_offline.sh
#   DATA_DIR=data/my-data EPOCHS=100 SEED=42 ./run_train_offline.sh
#
# Tunables (env vars):
#   PYTHON, ENV_ID, DATA_DIR, OUTPUT_DIR, SEED, DEVICE
#   EPOCHS, BATCH_SIZE, ENSEMBLE_SIZE, HIDDEN_SIZE
#   EXTRA_ARGS  : extra CLI arguments forwarded to pets/train_offline.py
###############################################################################
set -u

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}" || exit 1

if [ -n "${PYTHON:-}" ]; then
	:
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PYTHON="${VIRTUAL_ENV}/bin/python"
else
	PYTHON="${REPO_DIR}/.venv/bin/python"
fi

if [ ! -x "${PYTHON}" ]; then
	echo "[run_train_offline] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_train_offline] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

ENV_ID="${ENV_ID:-Pendulum-v1}"
DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pets-offline}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"
HIDDEN_SIZE="${HIDDEN_SIZE:-200}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${DATA_DIR}/replay_buffer.npz" ]; then
	echo "[run_train_offline] error: missing ${DATA_DIR}/replay_buffer.npz" >&2
	echo "[run_train_offline] collect data first with ./run_collect_data.sh" >&2
	exit 1
fi

DEVICE_ARGS=()
if [ -n "${DEVICE}" ]; then
	DEVICE_ARGS=(--device "${DEVICE}")
fi
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_train_offline] env_id=${ENV_ID} data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"
echo "[run_train_offline] seed=${SEED} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
echo "[run_train_offline] ensemble_size=${ENSEMBLE_SIZE} hidden_size=${HIDDEN_SIZE} device=${DEVICE:-auto}"

# shellcheck disable=SC2086
"${PYTHON}" "${SCRIPT_DIR}/train_offline.py" \
	--env-id "${ENV_ID}" \
	--data-dir "${DATA_DIR}" \
	--output-dir "${OUTPUT_DIR}" \
	--seed "${SEED}" \
	--epochs "${EPOCHS}" \
	--batch-size "${BATCH_SIZE}" \
	--ensemble-size "${ENSEMBLE_SIZE}" \
	--hidden-size "${HIDDEN_SIZE}" \
	"${DEVICE_ARGS[@]}" \
	${EXTRA_ARGS} \
	"$@"
