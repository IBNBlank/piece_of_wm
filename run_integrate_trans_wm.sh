#!/usr/bin/env bash
###############################################################################
# Run offline Trans-WM pretraining followed by formal online training.
#
# Usage:
#   ./run_integrate_trans_wm.sh
#   DEVICE=cuda PRETRAIN_EPOCHS=100 TRAIN_ROLLOUTS=500 \
#     NUM_CRITICS=5 ./run_integrate_trans_wm.sh
#
# Shared:
#   DATA_DIR              : offline dataset (default: dataset/pendulum-random)
#   NUM_ENVS              : environments per collected rollout (default: 10)
#   MAX_STEPS             : maximum episode steps (default: 200)
#   DEVICE                : torch device; empty selects CUDA when available
#   SEED                  : random seed (default: 0)
#   NUM_CRITICS           : critic count stored by pretraining (default: 2)
#
# Pretraining:
#   PRETRAIN_OUTPUT_DIR   : pretraining run directory
#                          (default: runs/trans_wm_pretrain)
#   PRETRAIN_EPOCHS       : complete offline dataset passes (default: 100)
#   PRETRAIN_BATCH_SIZE   : offline minibatch size (default: 512)
#   PRETRAIN_RESUME       : pretraining checkpoint or run directory
#   PRETRAIN_EXTRA_ARGS   : extra arguments for the pretraining script
#
# Formal training:
#   TRAIN_OUTPUT_DIR      : formal training run directory
#                          (default: runs/trans_wm)
#   TRAIN_ROLLOUTS        : formal training units (default: 500)
#   TRAIN_BATCH_SIZE      : replay minibatch size (default: 512)
#   TRAIN_RESUME          : formal checkpoint or run directory
#   TRAIN_EXTRA_ARGS      : extra arguments for the formal training script
###############################################################################
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-dataset/pendulum-random}"
NUM_ENVS="${NUM_ENVS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-0}"
NUM_CRITICS="${NUM_CRITICS:-2}"

PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-runs/trans_wm_pretrain}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-512}"
PRETRAIN_RESUME="${PRETRAIN_RESUME:-}"
PRETRAIN_EXTRA_ARGS="${PRETRAIN_EXTRA_ARGS:-}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-runs/trans_wm}"
TRAIN_ROLLOUTS="${TRAIN_ROLLOUTS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_RESUME="${TRAIN_RESUME:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

if [ "${PRETRAIN_OUTPUT_DIR}" = "${TRAIN_OUTPUT_DIR}" ]; then
	echo "[run_integrate_trans_wm] error: pretraining and training output directories must differ" >&2
	exit 1
fi

echo "[run_integrate_trans_wm] stage=pretrain output_dir=${PRETRAIN_OUTPUT_DIR}"
env \
	DATA_DIR="${DATA_DIR}" \
	OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR}" \
	NUM_ENVS="${NUM_ENVS}" \
	MAX_STEPS="${MAX_STEPS}" \
	EPOCHS="${PRETRAIN_EPOCHS}" \
	BATCH_SIZE="${PRETRAIN_BATCH_SIZE}" \
	NUM_CRITICS="${NUM_CRITICS}" \
	DEVICE="${DEVICE}" \
	SEED="${SEED}" \
	RESUME="${PRETRAIN_RESUME}" \
	EXTRA_ARGS="${PRETRAIN_EXTRA_ARGS}" \
	"${SCRIPT_DIR}/run_pretrain_trans_wm.sh"

pretrained_checkpoint="${PRETRAIN_OUTPUT_DIR}/checkpoint_best.pt"
if [ ! -f "${pretrained_checkpoint}" ]; then
	echo "[run_integrate_trans_wm] error: pretraining did not produce ${pretrained_checkpoint}" >&2
	exit 1
fi
if [ -n "${TRAIN_RESUME}" ]; then
	train_pretrained_checkpoint=""
else
	train_pretrained_checkpoint="${pretrained_checkpoint}"
fi

echo "[run_integrate_trans_wm] stage=train output_dir=${TRAIN_OUTPUT_DIR}"
env \
	DATA_DIR="${DATA_DIR}" \
	OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
	NUM_ENVS="${NUM_ENVS}" \
	MAX_STEPS="${MAX_STEPS}" \
	ROLLOUTS="${TRAIN_ROLLOUTS}" \
	BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
	DEVICE="${DEVICE}" \
	SEED="${SEED}" \
	RESUME="${TRAIN_RESUME}" \
	PRETRAINED_CHECKPOINT="${train_pretrained_checkpoint}" \
	EXTRA_ARGS="${TRAIN_EXTRA_ARGS}" \
	"${SCRIPT_DIR}/run_train_trans_wm.sh"
