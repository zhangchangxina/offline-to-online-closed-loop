#!/usr/bin/env bash

set -euo pipefail

# Usage: bash experiments/scripts/pipelines/run_halfcheetah_pipeline.sh <GPU_ID>

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export JAX_TRACEBACK_FILTERING=off

# Locomotion defaults (HalfCheetah)
ENV_ID="halfcheetah-medium-v0"
SEED=0
SAVE_ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log"
PROJECT_DIR="wsrl"

R_SCALE=1.0
R_BIAS=0.0

echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain+online for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:locomotion_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --utd 4 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps 250000 \
  --save_interval 100000 \
  --exp_name calql_ensemble_highutd \
  --save_dir ${SAVE_ROOT} \
  2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log

EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
CKPT_PATH="${RUN_DIR}/checkpoint_250000"
echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"

echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL-250K for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:locomotion_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 250000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --warmup_update_critic False \
  --config.agent_kwargs.log_actor_grad_terms=True \
  --config.agent_kwargs.actor_log_std_layer_name=Dense_1 \
  --exp_name wsrl \
  --save_dir ${SAVE_ROOT} | cat

#
# SAC-BC variant (TD-weighted BC; optional)
#
echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-250K for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:locomotion_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 250000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --warmup_update_critic True \
  --config.agent_kwargs.bc_loss_weight=1.0 \
  --config.agent_kwargs.bc_target=${CKPT_PATH} \
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --config.agent_kwargs.bc_td_weight_enabled=True \
  --config.agent_kwargs.bc_td_weight_clip=10.0 \
  --config.agent_kwargs.bc_td_weight_scale=0.5 \
  --config.agent_kwargs.bc_td_weight_power=2.0 \
  --config.agent_kwargs.bc_online_enable_for_steps=25000 \
  --exp_name wsrl_sacbc \
  --save_dir ${SAVE_ROOT} | cat

echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."


