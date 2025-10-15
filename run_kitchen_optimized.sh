#!/usr/bin/env bash

# JAX JIT 优化版本 - 解决训练卡顿问题
set -euo pipefail

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 关键优化：设置环境变量避免 JIT 重新编译
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_ENABLE_X64=false
export JAX_TRACEBACK_FILTERING=off
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true"

# 其他环境变量
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=../datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top

ENV_ID="kitchen-partial-v0"
SEED=0
SAVE_ROOT="../wsrl_log"
PROJECT_DIR="wsrl"

R_SCALE=1.0
R_BIAS=-4.0

EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
CKPT_PATH="${RUN_DIR}/checkpoint_250000"
echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"

echo "[GPU ${GPU_ID}] WSRL (SAC-BC) OPTIMIZED for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 250000 \
  --num_online_steps 300000 \
  --utd 2 \  # 从 4 降到 2，减少计算量
  --batch_size 512 \  # 从 1024 降到 512，减少内存使用
  --warmup_steps 5000 \
  --config.agent_kwargs.critic_ensemble_size=5 \  # 从 10 降到 5，减少 Q 网络计算
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=0.1 \
  --config.agent_kwargs.bc_lambda_schedule=adaptive \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-2 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=success \
  --config.agent_kwargs.bc_constraint=0.2 \
  --config.agent_kwargs.bc_target=dataset \
  --config.agent_kwargs.bc_weight_mode=uniform \  # 从 td_inverse 改为 uniform，减少计算
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --exp_name wsrl_sacbc_optimized \
  --save_dir ${SAVE_ROOT} | cat

echo "[GPU ${GPU_ID}] 优化训练完成"
