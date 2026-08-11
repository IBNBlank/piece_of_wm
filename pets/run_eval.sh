#!/usr/bin/env bash
###############################################################################
# Evaluate a saved PETS model in the real environment and record MP4 videos.
#
# Usage:
#   ./pets/run_eval.sh
#   MODEL_DIR=runs/pets-online/model EPISODES=3 ./pets/run_eval.sh
#
# Tunables (env vars):
#   PYTHON, MODEL_DIR, ENV_ID, OUTPUT_DIR, EPISODES, MAX_STEPS, SEED, DEVICE
#   VIDEO_FPS, NO_VIDEO, EXTRA_ARGS
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
	echo "[run_eval] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_eval] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

MODEL_DIR="${MODEL_DIR:-runs/pets-offline}"
ENV_ID="${ENV_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pets-eval}"
EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
SEED="${SEED:-2048}"
DEVICE="${DEVICE:-}"
VIDEO_FPS="${VIDEO_FPS:-30}"
NO_VIDEO="${NO_VIDEO:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "${MODEL_DIR}/model.pth" ] || [ ! -f "${MODEL_DIR}/model_config.json" ]; then
	echo "[run_eval] error: missing model.pth or model_config.json in ${MODEL_DIR}" >&2
	exit 1
fi

ENV_ARGS=()
if [ -n "${ENV_ID}" ]; then
	ENV_ARGS=(--env-id "${ENV_ID}")
fi
DEVICE_ARGS=()
if [ -n "${DEVICE}" ]; then
	DEVICE_ARGS=(--device "${DEVICE}")
fi
VIDEO_ARGS=()
if [ "${NO_VIDEO}" = "1" ]; then
	VIDEO_ARGS=(--no-video)
fi
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[run_eval] model_dir=${MODEL_DIR} output_dir=${OUTPUT_DIR} episodes=${EPISODES}"
echo "[run_eval] max_steps=${MAX_STEPS} seed=${SEED} video_fps=${VIDEO_FPS} no_video=${NO_VIDEO}"

# shellcheck disable=SC2086
"${PYTHON}" "${SCRIPT_DIR}/eval.py" \
	--model-dir "${MODEL_DIR}" \
	--output-dir "${OUTPUT_DIR}" \
	--episodes "${EPISODES}" \
	--max-steps "${MAX_STEPS}" \
	--seed "${SEED}" \
	--video-fps "${VIDEO_FPS}" \
	"${ENV_ARGS[@]}" \
	"${DEVICE_ARGS[@]}" \
	"${VIDEO_ARGS[@]}" \
	${EXTRA_ARGS} \
	"$@"
