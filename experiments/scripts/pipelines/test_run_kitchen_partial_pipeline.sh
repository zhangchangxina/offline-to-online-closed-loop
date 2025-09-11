#!/usr/bin/env bash

set -euo pipefail

# Usage: bash experiments/scripts/pipelines/run_kitchen_partial_pipeline.sh <GPU_ID>

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

ENV_ID="kitchen-partial-v0"
SEED=0
SAVE_ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log"
PROJECT_DIR="wsrl"

R_SCALE=1.0
R_BIAS=-4.0

# echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain+online for ${ENV_ID}"
# python3 finetune.py \
#   --agent calql \
#   --config experiments/configs/train_config.py:kitchen_cql \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --use_redq True \
#   --utd 4 \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --num_offline_steps 250000 \
#   --save_interval 100000 \
#   --exp_name calql_ensemble_highutd \
#   --save_dir ${SAVE_ROOT} \
#   2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log

EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
CKPT_PATH="${RUN_DIR}/checkpoint_250000"
echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"

echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL-1M for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 0 \
  --num_online_steps 1 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --warmup_update_critic False \
  --n_eval_trajs 20 \
  --save_interval 1 \
  --eval_interval 1 \
  --log_interval 1 \
  --config.agent_kwargs.log_actor_grad_terms=True \
  --config.agent_kwargs.actor_log_std_layer_name=Dense_1 \
  --exp_name wsrl \
  --save_dir ${SAVE_ROOT} | cat

#
# SAC-BC variant (single-step test)
#
echo "[GPU ${GPU_ID}] WSRL (SAC-BC single-step test) for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 0 \
  --num_online_steps 1 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 0 \
  --warmup_update_critic False \
  --n_eval_trajs 2 \
  --save_interval 1 \
  --eval_interval 1 \
  --log_interval 1 \
  --config.agent_kwargs.bc_loss_weight=1.0 \
  --config.agent_kwargs.bc_target=actor_target \
  --config.agent_kwargs.bc_weight_mode=td \
  --config.agent_kwargs.bc_weight_clip=10.0 \
  --exp_name wsrl_sacbc_test \
  --save_dir ${SAVE_ROOT} | cat

# echo "[GPU ${GPU_ID}] Closed-Loop SAC from CALQL-1M for ${ENV_ID}"
# python3 finetune.py \
#   --agent closed_loop_sac \
#   --config experiments/configs/train_config.py:kitchen_closed_loop_sac \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --use_redq True \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --resume_path ${CKPT_PATH} \
#   --num_offline_steps 250000 \
#   --utd 4 \
#   --batch_size 1024 \
#   --warmup_steps 5000 \
#   --config.agent_kwargs.policy_loss_variant=q_trust \
#   --config.agent_kwargs.q_trust_beta=0.0 \
#   --config.agent_kwargs.lambda_schedule=linear \
#   --config.agent_kwargs.lam_align=1 \
#   --config.agent_kwargs.align_steps=1 \
#   --config.agent_kwargs.align_constraint=0.1 \
#   --config.agent_kwargs.align_lagrange_optimizer_kwargs.learning_rate=1e-3 \
#   --exp_name clsac \
#   --save_dir ${SAVE_ROOT} | cat

# python3 finetune.py \
#   --agent closed_loop_sac \
#   --config experiments/configs/train_config.py:kitchen_closed_loop_sac \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --use_redq True \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --resume_path ${CKPT_PATH} \
#   --num_offline_steps 250000 \
#   --utd 4 \
#   --batch_size 1024 \
#   --warmup_steps 5000 \
#   --config.agent_kwargs.policy_loss_variant=align \
#   --config.agent_kwargs.q_trust_beta=2.0 \
#   --config.agent_kwargs.lambda_schedule=linear \
#   --config.agent_kwargs.lam_align=10 \
#   --config.agent_kwargs.align_steps=20000 \
#   --config.agent_kwargs.align_constraint=0.1 \
#   --config.agent_kwargs.align_lagrange_optimizer_kwargs.learning_rate=1e-3 \
#   --exp_name clsac \
#   --save_dir ${SAVE_ROOT} | cat

echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."


