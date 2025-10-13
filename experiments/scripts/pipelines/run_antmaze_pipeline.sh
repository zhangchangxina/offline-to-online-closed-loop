#!/usr/bin/env bash

set -euo pipefail

# Usage: bash experiments/scripts/pipelines/run_antmaze_pipeline.sh <GPU_ID>

GPU_ID=${1:-3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export JAX_TRACEBACK_FILTERING=off

ENV_ID="antmaze-medium-play-v2"
SEED=0
SAVE_ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log"
PROJECT_DIR="wsrl"

# AntMaze recommended scaling
R_SCALE=10.0
R_BIAS=-5.0

# echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain+online for ${ENV_ID}"
# python3 finetune.py \
#   --agent calql \
#   --config experiments/configs/train_config.py:antmaze_cql \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --use_redq True \
#   --utd 4 \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --num_offline_steps 1000000 \
#   --save_interval 100000 \
#   --exp_name calql_ensemble_highutd \
#   --save_dir ${SAVE_ROOT} \
#   2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log

EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
CKPT_PATH="${RUN_DIR}/checkpoint_1000000"
echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"

# echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL-1M for ${ENV_ID}"
# python3 finetune.py \
#   --agent sac \
#   --config experiments/configs/train_config.py:antmaze_wsrl \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --resume_path ${CKPT_PATH} \
#   --num_offline_steps 1000000 \
#   --use_redq True \
#   --utd 4 \
#   --batch_size 1024 \
#   --warmup_steps 5000 \
#   --exp_name wsrl \
#   --save_dir ${SAVE_ROOT} | cat



echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-1M for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 1000000 \
  --use_redq True \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=adaptive \
  --config.agent_kwargs.bc_constraint_mode=q_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-4 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=return \
  --config.agent_kwargs.bc_constraint=0.2 \
  --config.agent_kwargs.bc_target=dataset \
  --config.agent_kwargs.bc_weight_mode=none \
  --config.agent_kwargs.bc_uncert_action_source=dataset \
  --config.agent_kwargs.bc_uncert_q_source=current \
  --config.agent_kwargs.bc_weight_uncert_measure=std \
  --config.agent_kwargs.bc_weight_clip=10.0 \
  --config.agent_kwargs.bc_weight_normalize=True \
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --exp_name wsrl_sacbc \
  --save_dir ${SAVE_ROOT} | cat


echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-1M for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 1000000 \
  --use_redq True \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 0 \
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=adaptive \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-4 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=return \
  --config.agent_kwargs.bc_constraint=0.2 \
  --config.agent_kwargs.bc_target=dataset \
  --config.agent_kwargs.bc_weight_mode=td_inverse \
  --config.agent_kwargs.bc_uncert_action_source=dataset \
  --config.agent_kwargs.bc_uncert_q_source=current \
  --config.agent_kwargs.bc_weight_uncert_measure=std \
  --config.agent_kwargs.bc_weight_clip=10.0 \
  --config.agent_kwargs.bc_weight_normalize=True \
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --exp_name wsrl_sacbc \
  --save_dir ${SAVE_ROOT} | cat

echo "[GPU ${GPU_ID}] AWAC from CALQL-1M for ${ENV_ID}"
python3 finetune.py \
  --agent awac \
  --config experiments/configs/train_config.py:antmaze_awac \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 1000000 \
  --use_redq True \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --exp_name awac \
  --save_dir ${SAVE_ROOT} | cat
  

echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."


