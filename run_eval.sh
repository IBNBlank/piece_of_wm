#!/usr/bin/env bash
###############################################################################
# Online environment evaluation for Trans-WM and Trans-WM-LE policies.
#
# Usage:
#   ./run_eval.sh
#   MODEL=trans_wm_le EPISODES=10 DEVICE=cuda ./run_eval.sh
#
# Tunables (env vars):
#   PYTHON                  : Python interpreter
#   MODEL                   : all, trans_wm, or trans_wm_le (default: all)
#   ENV_ID                  : online Gymnasium environment (default: Pendulum-v1)
#   TRANS_WM_CHECKPOINT     : Trans-WM checkpoint path
#   TRANS_WM_LE_CHECKPOINT  : Trans-WM-LE checkpoint path
#   EPISODES                : online evaluation episodes (default: 5)
#   MAX_STEPS               : maximum environment steps per episode (default: 200)
#   NUM_PARTICLES           : action particles per policy update (default: 1000)
#   PARTICLE_UPDATES        : particle resampling iterations per action (default: 5)
#   PARTICLE_SIGMA          : particle perturbation standard deviation (default: 0.1)
#   PARTICLE_TEMPERATURE    : softmax resampling temperature (default: 2.0)
#   PLANNING_HORIZON        : model steps used for training and planning (default: 20)
#   SEED                    : environment and policy seed
#   DEVICE                  : torch device; empty selects CUDA when available
#   OUTPUT                  : JSON results path
#   VISUAL_DIR              : online return plot and per-episode GIF directory
#   FPS                     : GIF playback frame rate
#   EXTRA_ARGS              : additional arguments forwarded to eval.py
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
	echo "[run_eval] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_eval] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

MODEL="${MODEL:-trans_wm_le}"
ENV_ID="${ENV_ID:-Pendulum-v1}"
TRANS_WM_CHECKPOINT="${TRANS_WM_CHECKPOINT:-runs/trans_wm/checkpoint_best.pt}"
TRANS_WM_LE_CHECKPOINT="${TRANS_WM_LE_CHECKPOINT:-runs/trans_wm_le/checkpoint_best.pt}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-200}"
PARTICLE_UPDATES="${PARTICLE_UPDATES:-5}"
NUM_PARTICLES="${NUM_PARTICLES:-1000}"
PARTICLE_SIGMA="${PARTICLE_SIGMA:-0.1}"
PARTICLE_TEMPERATURE="${PARTICLE_TEMPERATURE:-2.0}"
PLANNING_HORIZON="${PLANNING_HORIZON:-20}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
OUTPUT="${OUTPUT:-runs/eval/results.json}"
VISUAL_DIR="${VISUAL_DIR:-runs/eval}"
FPS="${FPS:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
	--model "${MODEL}"
	--env-id "${ENV_ID}"
	--trans-wm-checkpoint "${TRANS_WM_CHECKPOINT}"
	--trans-wm-le-checkpoint "${TRANS_WM_LE_CHECKPOINT}"
	--episodes "${EPISODES}"
	--max-steps "${MAX_STEPS}"
	--particle-updates "${PARTICLE_UPDATES}"
	--num-particles "${NUM_PARTICLES}"
	--particle-sigma "${PARTICLE_SIGMA}"
	--particle-temperature "${PARTICLE_TEMPERATURE}"
	--planning-horizon "${PLANNING_HORIZON}"
	--seed "${SEED}"
	--output "${OUTPUT}"
	--visual-dir "${VISUAL_DIR}"
	--fps "${FPS}"
)
if [ -n "${DEVICE}" ]; then
	args+=(--device "${DEVICE}")
fi

echo "[run_eval] ONLINE model=${MODEL} env_id=${ENV_ID} episodes=${EPISODES} max_steps=${MAX_STEPS}"
echo "[run_eval] particles=${NUM_PARTICLES} particle_updates=${PARTICLE_UPDATES} sigma=${PARTICLE_SIGMA} temperature=${PARTICLE_TEMPERATURE} planning_horizon=${PLANNING_HORIZON} device=${DEVICE:-auto} seed=${SEED}"

# shellcheck disable=SC2086
"${PYTHON}" "${SCRIPT_DIR}/eval.py" "${args[@]}" ${EXTRA_ARGS} "$@"
