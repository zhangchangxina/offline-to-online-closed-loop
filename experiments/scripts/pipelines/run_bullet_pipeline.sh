#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a
# Usage: bash experiments/scripts/pipelines/run_bullet_pipeline.sh <GPU_ID>

GPU_ID=${1:-3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# # 关键优化：设置环境变量避免 JIT 重新编译
# export TF_ENABLE_ONEDNN_OPTS=0
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# export JAX_ENABLE_X64=false
# export JAX_TRACEBACK_FILTERING=off
# export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true"

# # 其他环境变量
# export PYOPENGL_PLATFORM=egl
# export MUJOCO_GL=egl
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export D4RL_DATASET_DIR=../datasets/d4rl
# export WANDB_BASE_URL=https://api.bandw.top

# Bullet defaults (PyBullet physics simulation)
ENV_ID="bullet-halfcheetah-medium-v0"
SEED=0
PROJECT_DIR="wsrl"

# Bullet recommended scaling (similar to MuJoCo but with different physics)
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
  --save_interval 250000 \
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
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=adaptive \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-4 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=success \
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

echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."
