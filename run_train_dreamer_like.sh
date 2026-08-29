#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
if [ -n "${PYTHON:-}" ]; then :
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then PYTHON="${VIRTUAL_ENV}/bin/python"
else PYTHON="${SCRIPT_DIR}/.venv/bin/python"; fi
if [ ! -x "${PYTHON}" ]; then
  echo "[run_train_dreamer_like] error: Python executable not found: ${PYTHON}" >&2
  exit 1
fi

DATA_DIR="${DATA_DIR:-dataset/fetch-pick-and-place-random}"
OUTPUT="${OUTPUT:-runs/dreamer_like/checkpoint.pt}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HORIZON="${HORIZON:-16}"
DEVICE="${DEVICE:-cpu}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_train_dreamer_like] data_dir=${DATA_DIR} steps=${STEPS} batch_size=${BATCH_SIZE} horizon=${HORIZON} device=${DEVICE}"
"${PYTHON}" -m dreamer_like.train \
  --data-dir "${DATA_DIR}" --steps "${STEPS}" --batch-size "${BATCH_SIZE}" \
  --horizon "${HORIZON}" --device "${DEVICE}" --output "${OUTPUT}" "$@"
