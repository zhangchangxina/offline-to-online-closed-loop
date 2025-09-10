#!/usr/bin/env bash

set -euo pipefail

# Usage: bash experiments/scripts/pipelines/run_calql_redq_utd4.sh <ENV_ID> <GPU_ID>
# Example: bash experiments/scripts/pipelines/run_calql_redq_utd4.sh antmaze-umaze-diverse-v0 3

ENV_ID=${1:-antmaze-umaze-diverse-v0}
GPU_ID=${2:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
# Show full JAX traceback for debugging
export JAX_TRACEBACK_FILTERING=off

SEED=0
SAVE_ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log"

# Infer config and reward scaling from env
CONFIG_KEY=""
R_SCALE=1.0
R_BIAS=0.0
if [[ "${ENV_ID}" == antmaze-* ]]; then
  CONFIG_KEY="antmaze_cql"
  R_SCALE=10.0
  R_BIAS=-5.0
elif [[ "${ENV_ID}" == kitchen-* ]]; then
  CONFIG_KEY="kitchen_cql"
  R_SCALE=1.0
  R_BIAS=-4.0
elif [[ "${ENV_ID}" == pen-binary-v0 || "${ENV_ID}" == door-binary-v0 || "${ENV_ID}" == relocate-binary-v0 ]]; then
  CONFIG_KEY="adroit_cql"
  export DATA_DIR_PREFIX=/media/nudt3090/XYQ/ZCX/WSRL/datasets/adroit_data
  R_SCALE=10.0
  R_BIAS=5.0
else
  echo "Unknown ENV_ID: ${ENV_ID}" >&2
  exit 1
fi

echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain+online for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:${CONFIG_KEY} \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --utd 4 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps 1000000 \
  --save_interval 100000 \
  --exp_name calql_redq10 \
  --save_dir ${SAVE_ROOT} \
  2>&1 | tee -a ${SAVE_ROOT}/calql_redq10_${ENV_ID}_seed${SEED}.log


