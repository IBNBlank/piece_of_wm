#!/usr/bin/env bash
###############################################################################
# Run Trans-WM-LE pretraining, formal training, and final online evaluation.
#
# Usage:
#   ./run_integrate_trans_wm_le.sh
#   DEVICE=cuda PRETRAIN_EPOCHS=100 TRAIN_ROLLOUTS=500 ./run_integrate_trans_wm_le.sh
#
# Shared:
#   DATA_DIR              : offline dataset (default: dataset/pendulum-random)
#   NUM_ENVS              : environments per collected rollout (default: 20)
#   MAX_STEPS             : maximum episode steps (default: 200)
#   DEVICE                : torch device; empty selects CUDA when available
#   SEED                  : random seed (default: 0)
#   PLANNING_HORIZON      : shared WM-training/planning horizon (default: 10)
#   NUM_PARTICLES         : planning particles (default: 1000)
#   PARTICLE_UPDATES      : particle update iterations (default: 5)
#   PARTICLE_SIGMA        : initial particle noise (default: 0.1)
#   PARTICLE_TEMPERATURE  : resampling temperature (default: 2.0)
#
# Pretraining:
#   PRETRAIN_OUTPUT_DIR   : pretraining run directory
#                          (default: runs/trans_wm_le_pretrain)
#   PRETRAIN_EPOCHS       : complete offline dataset passes (default: 100)
#   PRETRAIN_BATCH_SIZE   : offline minibatch size (default: 256)
#   PRETRAIN_RESUME       : pretraining checkpoint or run directory
#   PRETRAIN_EXTRA_ARGS   : extra arguments for the pretraining script
#
# Formal training:
#   TRAIN_OUTPUT_DIR      : formal training run directory
#                          (default: runs/trans_wm_le)
#   TRAIN_ROLLOUTS        : formal training units (default: 500)
#   TRAIN_BATCH_SIZE      : replay minibatch size (default: 128)
#   TRAIN_RESUME          : formal checkpoint or run directory
#   TRAIN_EXTRA_ARGS      : extra arguments for the formal training script
#
# Evaluation:
#   EVAL_EPISODES         : final online evaluation episodes (default: 5)
#   EVAL_OUTPUT           : evaluation JSON path
#   EVAL_VISUAL_DIR       : evaluation plots/GIF directory
#   EVAL_EXTRA_ARGS       : extra arguments for run_eval.sh
###############################################################################
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
NUM_ENVS="${NUM_ENVS:-20}"
MAX_STEPS="${MAX_STEPS:-200}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-0}"
PLANNING_HORIZON="${PLANNING_HORIZON:-10}"
NUM_PARTICLES="${NUM_PARTICLES:-1000}"
PARTICLE_UPDATES="${PARTICLE_UPDATES:-5}"
PARTICLE_SIGMA="${PARTICLE_SIGMA:-0.1}"
PARTICLE_TEMPERATURE="${PARTICLE_TEMPERATURE:-2.0}"

PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-runs/trans_wm_le_pretrain}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_RESUME="${PRETRAIN_RESUME:-}"
PRETRAIN_EXTRA_ARGS="${PRETRAIN_EXTRA_ARGS:-}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-runs/trans_wm_le}"
TRAIN_ROLLOUTS="${TRAIN_ROLLOUTS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
TRAIN_RESUME="${TRAIN_RESUME:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${TRAIN_OUTPUT_DIR}/eval/results.json}"
EVAL_VISUAL_DIR="${EVAL_VISUAL_DIR:-${TRAIN_OUTPUT_DIR}/eval}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

if [ "${PRETRAIN_OUTPUT_DIR}" = "${TRAIN_OUTPUT_DIR}" ]; then
	echo "[run_integrate_trans_wm_le] error: pretraining and training output directories must differ" >&2
	exit 1
fi

echo "[run_integrate_trans_wm_le] stage=pretrain output_dir=${PRETRAIN_OUTPUT_DIR}"
env \
	DATA_DIR="${DATA_DIR}" \
	OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}" \
	NUM_ENVS="${NUM_ENVS}" \
	MAX_STEPS="${MAX_STEPS}" \
	EPOCHS="${PRETRAIN_EPOCHS}" \
	BATCH_SIZE="${PRETRAIN_BATCH_SIZE}" \
	PLANNING_HORIZON="${PLANNING_HORIZON}" \
	DEVICE="${DEVICE}" \
	SEED="${SEED}" \
	RESUME="${PRETRAIN_RESUME}" \
	EXTRA_ARGS="${PRETRAIN_EXTRA_ARGS}" \
	"${SCRIPT_DIR}/run_pretrain_trans_wm_le.sh"

pretrained_checkpoint="${PRETRAIN_OUTPUT_DIR}/checkpoint_best.pt"
if [ ! -f "${pretrained_checkpoint}" ]; then
	echo "[run_integrate_trans_wm_le] error: pretraining did not produce ${pretrained_checkpoint}" >&2
	exit 1
fi
if [ -n "${TRAIN_RESUME}" ]; then
	train_pretrained_checkpoint=""
else
	train_pretrained_checkpoint="${pretrained_checkpoint}"
fi

echo "[run_integrate_trans_wm_le] stage=train output_dir=${TRAIN_OUTPUT_DIR}"
env \
	DATA_DIR="${DATA_DIR}" \
	OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
	NUM_ENVS="${NUM_ENVS}" \
	MAX_STEPS="${MAX_STEPS}" \
	ROLLOUTS="${TRAIN_ROLLOUTS}" \
	BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
	PLANNING_HORIZON="${PLANNING_HORIZON}" \
	NUM_PARTICLES="${NUM_PARTICLES}" \
	PARTICLE_UPDATES="${PARTICLE_UPDATES}" \
	PARTICLE_SIGMA="${PARTICLE_SIGMA}" \
	PARTICLE_TEMPERATURE="${PARTICLE_TEMPERATURE}" \
	DEVICE="${DEVICE}" \
	SEED="${SEED}" \
	RESUME="${TRAIN_RESUME}" \
	PRETRAINED_CHECKPOINT="${train_pretrained_checkpoint}" \
	EXTRA_ARGS="${TRAIN_EXTRA_ARGS}" \
	"${SCRIPT_DIR}/run_train_trans_wm_le.sh"

trained_checkpoint="${TRAIN_OUTPUT_DIR}/checkpoint_best.pt"
if [ ! -f "${trained_checkpoint}" ]; then
	echo "[run_integrate_trans_wm_le] error: training did not produce ${trained_checkpoint}" >&2
	exit 1
fi

echo "[run_integrate_trans_wm_le] stage=eval checkpoint=${trained_checkpoint}"
env \
	MODEL="trans_wm_le" \
	TRANS_WM_LE_CHECKPOINT="${trained_checkpoint}" \
	EPISODES="${EVAL_EPISODES}" \
	MAX_STEPS="${MAX_STEPS}" \
	PLANNING_HORIZON="${PLANNING_HORIZON}" \
	NUM_PARTICLES="${NUM_PARTICLES}" \
	PARTICLE_UPDATES="${PARTICLE_UPDATES}" \
	PARTICLE_SIGMA="${PARTICLE_SIGMA}" \
	PARTICLE_TEMPERATURE="${PARTICLE_TEMPERATURE}" \
	DEVICE="${DEVICE}" \
	SEED="${SEED}" \
	OUTPUT="${EVAL_OUTPUT}" \
	VISUAL_DIR="${EVAL_VISUAL_DIR}" \
	EXTRA_ARGS="${EVAL_EXTRA_ARGS}" \
	"${SCRIPT_DIR}/run_eval.sh"
