#!/usr/bin/env bash
###############################################################################
# Train TD-MPC-like from sequence-preserving image rollouts.
#
# Usage:
#   ./run_train_tdmpc_like.sh
#   ROLLOUTS=500 BATCH_SIZE=16 DEVICE=cuda ./run_train_tdmpc_like.sh
#   JEPA_WEIGHT=1.0 SIGREG_WEIGHT=0.2 ./run_train_tdmpc_like.sh
#
# Tunables (env vars):
#   PYTHON                : Python interpreter
#   DATA_DIR              : rollout dataset directory (default: dataset/fetch-pick-and-place-random)
#   OUTPUT_DIR            : checkpoints and metrics directory (default: runs/tdmpc_like)
#   NUM_ENVS              : environments per collected rollout (default: 20)
#   MAX_STEPS             : maximum steps per environment episode (default: 200)
#   ROLLOUTS              : number of rollout training units (default: 500)
#   BATCH_SIZE            : sampled transitions per optimization batch (default: 128)
#   REPLAY_CAPACITY       : rollout files held in RAM (default: complete dataset)
#   SAMPLE_ROLLOUTS       : rollout batches combined per update (default: 2)
#   NUM_PARTICLES         : action particles per policy update (default: 1000)
#   PARTICLE_UPDATES      : particle resampling iterations (default: 5)
#   PARTICLE_SIGMA        : particle perturbation standard deviation (default: 0.1)
#   PARTICLE_TEMPERATURE  : softmax resampling temperature (default: 2.0)
#   PLANNING_HORIZON      : WM training and policy planning horizon (default: 20)
#   EVALUATION_ROLLOUTS   : online evaluation episodes per checkpoint (default: 10)
#   EPOCHS_PER_ROLLOUT    : optimization epochs for each rollout (default: 10)
#   CHECKPOINT_ROLLOUTS   : checkpoint frequency in rollouts (default: 10)
#   SEED                  : random seed (default: 0)
#   DEVICE                : torch device; empty selects CUDA when available
#   RESUME                : checkpoint path to resume (default: empty)
#   PRETRAINED_CHECKPOINT : pretraining checkpoint used to initialize training
#                           (default: runs/tdmpc_like_pretrain/checkpoint_best.pt)
#   LEARNING_RATE         : AdamW learning rate (default: 1e-4)
#   WEIGHT_DECAY          : AdamW weight decay (default: 1e-5)
#   GRAD_CLIP_NORM        : gradient clipping norm (default: 10.0)
#   JEPA_WEIGHT           : JEPA latent prediction loss weight (default: 1.0)
#   SIGREG_WEIGHT         : SIGReg regularization weight (default: 0.2)
#   SIGREG_PROJECTIONS    : random SIGReg projections (default: 256)
#   SIGREG_FREQUENCIES    : frequencies per SIGReg projection (default: 17)
#   SIGREG_MAX_FREQUENCY  : maximum SIGReg frequency (default: 5.0)
#   EXTRA_ARGS            : additional arguments forwarded to tdmpc_like.train
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
	echo "[run_train_tdmpc_like] error: Python executable not found: ${PYTHON}" >&2
	echo "[run_train_tdmpc_like] run ./venv.sh or set PYTHON=/path/to/python" >&2
	exit 1
fi

DATA_DIR="${DATA_DIR:-dataset/fetch-pick-and-place-random}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/tdmpc_like}"
NUM_ENVS="${NUM_ENVS:-20}"
MAX_STEPS="${MAX_STEPS:-200}"
ROLLOUTS="${ROLLOUTS:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-}"
SAMPLE_ROLLOUTS="${SAMPLE_ROLLOUTS:-2}"
NUM_PARTICLES="${NUM_PARTICLES:-1000}"
PARTICLE_UPDATES="${PARTICLE_UPDATES:-5}"
PARTICLE_SIGMA="${PARTICLE_SIGMA:-0.1}"
PARTICLE_TEMPERATURE="${PARTICLE_TEMPERATURE:-2.0}"
PLANNING_HORIZON="${PLANNING_HORIZON:-20}"
EVALUATION_ROLLOUTS="${EVALUATION_ROLLOUTS:-10}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-10}"
CHECKPOINT_ROLLOUTS="${CHECKPOINT_ROLLOUTS:-10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-}"
RESUME="${RESUME:-}"
if [ -n "${RESUME}" ]; then
	PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-}"
else
	PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-runs/tdmpc_like_pretrain/checkpoint_best.pt}"
fi
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
JEPA_WEIGHT="${JEPA_WEIGHT:-1.0}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.2}"
SIGREG_PROJECTIONS="${SIGREG_PROJECTIONS:-256}"
SIGREG_FREQUENCIES="${SIGREG_FREQUENCIES:-17}"
SIGREG_MAX_FREQUENCY="${SIGREG_MAX_FREQUENCY:-5.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "${RESUME}" ] && [ -n "${PRETRAINED_CHECKPOINT}" ]; then
	echo "[run_train_tdmpc_like] error: RESUME and PRETRAINED_CHECKPOINT are mutually exclusive" >&2
	exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
	--data-dir "${DATA_DIR}"
	--output-dir "${OUTPUT_DIR}"
	--num-envs "${NUM_ENVS}"
	--max-steps "${MAX_STEPS}"
	--rollouts "${ROLLOUTS}"
	--batch-size "${BATCH_SIZE}"
	--sample-rollouts "${SAMPLE_ROLLOUTS}"
	--num-particles "${NUM_PARTICLES}"
	--particle-updates "${PARTICLE_UPDATES}"
	--particle-sigma "${PARTICLE_SIGMA}"
	--particle-temperature "${PARTICLE_TEMPERATURE}"
	--planning-horizon "${PLANNING_HORIZON}"
	--evaluation-rollouts "${EVALUATION_ROLLOUTS}"
	--epochs-per-rollout "${EPOCHS_PER_ROLLOUT}"
	--checkpoint-rollouts "${CHECKPOINT_ROLLOUTS}"
	--seed "${SEED}"
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
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
	args+=(--pretrained-checkpoint "${PRETRAINED_CHECKPOINT}")
fi

echo "[run_train_tdmpc_like] data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"
echo "[run_train_tdmpc_like] pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-none}"
echo "[run_train_tdmpc_like] rollouts=${ROLLOUTS} num_envs=${NUM_ENVS} max_steps=${MAX_STEPS} epochs_per_rollout=${EPOCHS_PER_ROLLOUT} batch_size=${BATCH_SIZE} sample_rollouts=${SAMPLE_ROLLOUTS} evaluation_rollouts=${EVALUATION_ROLLOUTS} particles=${NUM_PARTICLES} horizon=${PLANNING_HORIZON} temperature=${PARTICLE_TEMPERATURE} replay_capacity=${REPLAY_CAPACITY:-all} device=${DEVICE:-auto} seed=${SEED}"
echo "[run_train_tdmpc_like] jepa_weight=${JEPA_WEIGHT} sigreg_weight=${SIGREG_WEIGHT}"

# shellcheck disable=SC2086
"${PYTHON}" -m tdmpc_like.train "${args[@]}" ${EXTRA_ARGS} "$@"
