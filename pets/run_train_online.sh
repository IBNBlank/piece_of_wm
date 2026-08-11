#!/usr/bin/env bash
###############################################################################
# Online PETS training: initial random data -> model fitting -> CEM/MPC rollout.
#
# Usage:
#   ./run_train_online.sh
#   SEED=42 TRIALS=10 EPOCHS=50 ./run_train_online.sh
#
# Tunables (env vars):
#   PYTHON, ENV_ID, DATA_DIR, OUTPUT_DIR, SEED, DEVICE
#   INITIAL_EPISODES, TRIALS, MAX_STEPS, EPOCHS, BATCH_SIZE
#   ENSEMBLE_SIZE, HIDDEN_SIZE, PLANNING_HORIZON
#   CEM_ITERATIONS, CEM_POPULATION_SIZE, NUM_PARTICLES
#   EXTRA_ARGS  : extra CLI arguments forwarded to pets/train_online.py
#
# Set DATA_DIR to reuse a prior random/offline replay buffer. Leave it empty to
# collect INITIAL_EPISODES random episodes before online PETS training.
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
	echo "[run_train_online] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_train_online] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

ENV_ID="${ENV_ID:-Pendulum-v1}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pets-online}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
INITIAL_EPISODES="${INITIAL_EPISODES:-1}"
TRIALS="${TRIALS:-6}"
MAX_STEPS="${MAX_STEPS:-200}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"
HIDDEN_SIZE="${HIDDEN_SIZE:-200}"
PLANNING_HORIZON="${PLANNING_HORIZON:-15}"
CEM_ITERATIONS="${CEM_ITERATIONS:-4}"
CEM_POPULATION_SIZE="${CEM_POPULATION_SIZE:-256}"
NUM_PARTICLES="${NUM_PARTICLES:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "${DATA_DIR}" ] && [ ! -f "${DATA_DIR}/replay_buffer.npz" ]; then
	echo "[run_train_online] error: missing ${DATA_DIR}/replay_buffer.npz" >&2
	exit 1
fi

DATA_ARGS=()
if [ -n "${DATA_DIR}" ]; then
	DATA_ARGS=(--data-dir "${DATA_DIR}")
fi
DEVICE_ARGS=()
if [ -n "${DEVICE}" ]; then
	DEVICE_ARGS=(--device "${DEVICE}")
fi
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_train_online] env_id=${ENV_ID} data_dir=${DATA_DIR:-new random data} output_dir=${OUTPUT_DIR}"
echo "[run_train_online] seed=${SEED} initial_episodes=${INITIAL_EPISODES} trials=${TRIALS} max_steps=${MAX_STEPS}"
echo "[run_train_online] epochs=${EPOCHS} batch_size=${BATCH_SIZE} ensemble_size=${ENSEMBLE_SIZE}"
echo "[run_train_online] horizon=${PLANNING_HORIZON} cem=${CEM_ITERATIONS}x${CEM_POPULATION_SIZE} particles=${NUM_PARTICLES}"

# shellcheck disable=SC2086
"${PYTHON}" "${SCRIPT_DIR}/train_online.py" \
	--env-id "${ENV_ID}" \
	--output-dir "${OUTPUT_DIR}" \
	--initial-episodes "${INITIAL_EPISODES}" \
	--trials "${TRIALS}" \
	--max-steps "${MAX_STEPS}" \
	--seed "${SEED}" \
	--epochs "${EPOCHS}" \
	--batch-size "${BATCH_SIZE}" \
	--ensemble-size "${ENSEMBLE_SIZE}" \
	--hidden-size "${HIDDEN_SIZE}" \
	--planning-horizon "${PLANNING_HORIZON}" \
	--cem-iterations "${CEM_ITERATIONS}" \
	--cem-population-size "${CEM_POPULATION_SIZE}" \
	--num-particles "${NUM_PARTICLES}" \
	"${DATA_ARGS[@]}" \
	"${DEVICE_ARGS[@]}" \
	${EXTRA_ARGS} \
	"$@"
