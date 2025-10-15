#!/usr/bin/env bash

set -euo pipefail
set -a
source .env
set +a
# Usage: bash experiments/scripts/pipelines/run_flow_pipeline.sh <GPU_ID>

GPU_ID=${1:-4}
export CUDA_VISIBLE_DEVICES=${GPU_ID}


# Flow defaults (traffic simulation)
ENV_ID="flow-ring-random-v0"
SEED=0
PROJECT_DIR="wsrl"

# Flow recommended scaling (traffic control rewards)
R_SCALE=1.0
R_BIAS=0.0

echo "[GPU ${GPU_ID}] Checking env availability: ${ENV_ID}"
python3 - <<PY
import sys
try:
    import gym, d4rl  # noqa: F401
    gym.make("${ENV_ID}")
    print("[OK] ${ENV_ID} available.")
except Exception as e:
    print(f"[ERROR] ${ENV_ID} unavailable: {e}")
    sys.exit(1)
PY

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
  --num_offline_steps 200000 \
  --save_interval 200000 \
  --exp_name calql_ensemble_highutd \
  --save_dir ${SAVE_ROOT} \
  2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log

EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
CKPT_PATH="${RUN_DIR}/checkpoint_200000"
echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"

echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL-200K for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:locomotion_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 200000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --exp_name wsrl \
  --save_dir ${SAVE_ROOT} | cat

echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-200K for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:locomotion_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps 200000 \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=adaptive \
  --config.agent_kwargs.bc_constraint_mode=q_drop \
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
