#!/usr/bin/env bash
###############################################################################
# Random real-environment data collection for PETS world-model training.
#
# Usage:
#   ./run_collect_data.sh
#   ENV_ID=Pendulum-v1 EPISODES=20 SEED=42 ./run_collect_data.sh
#
# Tunables (env vars):
#   PYTHON      : Python interpreter (takes precedence over virtualenv choices)
#   ENV_ID      : Gymnasium environment (default: Pendulum-v1)
#   EPISODES    : number of random-policy episodes (default: 10)
#   MAX_STEPS   : maximum steps per episode (default: 200)
#   SEED        : random seed (default: 0)
#   OUTPUT_DIR  : directory for replay_buffer.npz (default: data/pendulum-random)
#   EXTRA_ARGS  : extra CLI arguments forwarded to collect_data.py
###############################################################################
set -u

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
cd "${REPO_DIR}" || exit 1

if [ -n "${PYTHON:-}" ]; then
	:
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
	PYTHON="${VIRTUAL_ENV}/bin/python"
else
	PYTHON="${REPO_DIR}/.venv/bin/python"
fi

if [ ! -x "${PYTHON}" ]; then
	echo "[run_collect_data] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_collect_data] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

ENV_ID="${ENV_ID:-Pendulum-v1}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset/pendulum-random}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_collect_data] env_id=${ENV_ID} episodes=${EPISODES} max_steps=${MAX_STEPS} seed=${SEED}"
echo "[run_collect_data] output_dir=${OUTPUT_DIR}"

# shellcheck disable=SC2086
"${PYTHON}" "${SCRIPT_DIR}/collect_data.py" \
	--env-id "${ENV_ID}" \
	--episodes "${EPISODES}" \
	--max-steps "${MAX_STEPS}" \
	--seed "${SEED}" \
	--output-dir "${OUTPUT_DIR}" \
	${EXTRA_ARGS} \
	"$@"
