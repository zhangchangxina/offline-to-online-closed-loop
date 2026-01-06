#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

# Usage: bash experiments/scripts/pipelines/run_antmaze_pipeline.sh <GPU_ID>
# Following WSRL paper (Zhou et al., 2024) settings

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# antmaze-medium-play-v2, antmaze-medium-diverse-v2, antmaze-large-play-v2, antmaze-large-diverse-v2
ENV_ID="antmaze-large-diverse-v2"
SEED=0
PROJECT_DIR="wsrl"

# AntMaze recommended scaling
R_SCALE=10.0
R_BIAS=-5.0

# Paper settings: 1M offline + 300k online for Antmaze
num_offline_steps=1000000
num_online_steps=300000
save_interval=100000000

# ============================================================================
# Offline Pre-training Phase
# Paper uses 2 Q-functions (default) for IQL/CQL/CalQL
# ============================================================================

# IQL pretrain (2 Q-functions, default UTD)
echo "[GPU ${GPU_ID}] IQL pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent iql \
  --config experiments/configs/train_config.py:antmaze_iql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name iql \
  --save_dir ${SAVE_ROOT}

# Get IQL checkpoint path
EXP_DESC_IQL="iql_${ENV_ID}_iql_seed${SEED}"
RUN_DIR_IQL=""
for dir in $(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC_IQL}_* 2>/dev/null); do
  if [[ -d "${dir}/checkpoint_${num_offline_steps}" ]]; then
    RUN_DIR_IQL="$dir"
    break
  fi
done
if [[ -z "$RUN_DIR_IQL" ]]; then
  echo "[ERROR] No IQL checkpoint found with ${num_offline_steps} steps for ${ENV_ID}"
  exit 1
fi
CKPT_PATH_IQL="${RUN_DIR_IQL}/checkpoint_${num_offline_steps}"
echo "[GPU ${GPU_ID}] Using IQL checkpoint: ${CKPT_PATH_IQL}"

# CQL pretrain (2 Q-functions, default UTD)
echo "[GPU ${GPU_ID}] CQL pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent cql \
  --config experiments/configs/train_config.py:antmaze_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name cql \
  --save_dir ${SAVE_ROOT}

# Get CQL checkpoint path
EXP_DESC_CQL="cql_${ENV_ID}_cql_seed${SEED}"
RUN_DIR_CQL=""
for dir in $(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC_CQL}_* 2>/dev/null); do
  if [[ -d "${dir}/checkpoint_${num_offline_steps}" ]]; then
    RUN_DIR_CQL="$dir"
    break
  fi
done
if [[ -z "$RUN_DIR_CQL" ]]; then
  echo "[ERROR] No CQL checkpoint found with ${num_offline_steps} steps for ${ENV_ID}"
  exit 1
fi
CKPT_PATH_CQL="${RUN_DIR_CQL}/checkpoint_${num_offline_steps}"
echo "[GPU ${GPU_ID}] Using CQL checkpoint: ${CKPT_PATH_CQL}"

# CalQL pretrain (2 Q-functions, default UTD)
echo "[GPU ${GPU_ID}] CalQL pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:antmaze_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name calql \
  --save_dir ${SAVE_ROOT}

# Get CalQL checkpoint path
EXP_DESC_CALQL="calql_${ENV_ID}_calql_seed${SEED}"
RUN_DIR_CALQL=""
for dir in $(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC_CALQL}_* 2>/dev/null); do
  if [[ -d "${dir}/checkpoint_${num_offline_steps}" ]]; then
    RUN_DIR_CALQL="$dir"
    break
  fi
done
if [[ -z "$RUN_DIR_CALQL" ]]; then
  echo "[ERROR] No CalQL checkpoint found with ${num_offline_steps} steps for ${ENV_ID}"
  exit 1
fi
CKPT_PATH_CALQL="${RUN_DIR_CALQL}/checkpoint_${num_offline_steps}"
echo "[GPU ${GPU_ID}] Using CalQL checkpoint: ${CKPT_PATH_CALQL}"

# AWAC pretrain (2 Q-functions, default UTD)
echo "[GPU ${GPU_ID}] AWAC pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent awac \
  --config experiments/configs/train_config.py:antmaze_awac \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name awac \
  --save_dir ${SAVE_ROOT}

# Get AWAC checkpoint path
EXP_DESC_AWAC="awac_${ENV_ID}_awac_seed${SEED}"
RUN_DIR_AWAC=""
for dir in $(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC_AWAC}_* 2>/dev/null); do
  if [[ -d "${dir}/checkpoint_${num_offline_steps}" ]]; then
    RUN_DIR_AWAC="$dir"
    break
  fi
done
if [[ -z "$RUN_DIR_AWAC" ]]; then
  echo "[ERROR] No AWAC checkpoint found with ${num_offline_steps} steps for ${ENV_ID}"
  exit 1
fi
CKPT_PATH_AWAC="${RUN_DIR_AWAC}/checkpoint_${num_offline_steps}"
echo "[GPU ${GPU_ID}] Using AWAC checkpoint: ${CKPT_PATH_AWAC}"

# CalQL pretrain with REDQ (10 Q-functions, UTD=4) - for WSRL initialization
echo "[GPU ${GPU_ID}] CalQL (REDQ10, UTD=4) pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:antmaze_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --utd 4 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name calql_redq \
  --save_dir ${SAVE_ROOT}

# Get CalQL REDQ checkpoint path (for WSRL)
EXP_DESC_CALQL_REDQ="calql_redq_${ENV_ID}_calql_seed${SEED}"
RUN_DIR_CALQL_REDQ=""
for dir in $(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC_CALQL_REDQ}_* 2>/dev/null); do
  if [[ -d "${dir}/checkpoint_${num_offline_steps}" ]]; then
    RUN_DIR_CALQL_REDQ="$dir"
    break
  fi
done
if [[ -z "$RUN_DIR_CALQL_REDQ" ]]; then
  echo "[ERROR] No CalQL REDQ checkpoint found with ${num_offline_steps} steps for ${ENV_ID}"
  exit 1
fi
CKPT_PATH_CALQL_REDQ="${RUN_DIR_CALQL_REDQ}/checkpoint_${num_offline_steps}"
echo "[GPU ${GPU_ID}] Using CalQL REDQ checkpoint: ${CKPT_PATH_CALQL_REDQ}"

# ============================================================================
# No-retention fine-tuning (resume from checkpoint, no offline data)
# Paper: This is the key comparison - fine-tuning without retaining offline data
# ============================================================================

# IQL no retention
echo "[GPU ${GPU_ID}] IQL no retention for ${ENV_ID}"
python3 finetune.py \
  --agent iql \
  --config experiments/configs/train_config.py:antmaze_iql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH_IQL} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name iql_no_retain \
  --save_dir ${SAVE_ROOT}

# CQL no retention
echo "[GPU ${GPU_ID}] CQL no retention for ${ENV_ID}"
python3 finetune.py \
  --agent cql \
  --config experiments/configs/train_config.py:antmaze_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH_CQL} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name cql_no_retain \
  --save_dir ${SAVE_ROOT}

# CalQL no retention
echo "[GPU ${GPU_ID}] CalQL no retention for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:antmaze_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH_CALQL} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name calql_no_retain \
  --save_dir ${SAVE_ROOT}

# AWAC no retention
echo "[GPU ${GPU_ID}] AWAC no retention for ${ENV_ID}"
python3 finetune.py \
  --agent awac \
  --config experiments/configs/train_config.py:antmaze_awac \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH_AWAC} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name awac_no_retain \
  --save_dir ${SAVE_ROOT}

# ============================================================================
# Online RL from scratch baselines
# Paper: RLPD/SAC(fast) use 10 Q-functions, UTD=4, batch_size=256
# ============================================================================

# RLPD (SAC with offline data in buffer, 50% ratio)
echo "[GPU ${GPU_ID}] RLPD (SAC with offline data ratio=0.5) for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --num_offline_steps 0 \
  --offline_data_ratio 0.5 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 256 \
  --exp_name rlpd \
  --save_dir ${SAVE_ROOT}

# SAC (fast) - online RL from scratch with high UTD
echo "[GPU ${GPU_ID}] SAC(fast) for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --num_offline_steps 0 \
  --offline_data_ratio 0 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 256 \
  --exp_name sac_fast \
  --save_dir ${SAVE_ROOT}

# ============================================================================
# WSRL: Fine-tuning from CalQL with warmup (no data retention)
# Paper: 10 Q-functions, UTD=4, batch_size=256, warmup=5000
# ============================================================================

echo "[GPU ${GPU_ID}] WSRL (SAC from CalQL) for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_CALQL_REDQ} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 256 \
  --warmup_steps 5000 \
  --exp_name wsrl \
  --save_dir ${SAVE_ROOT}

# ============================================================================
# WSRL with SAC-BC variants (your extensions)
# ============================================================================

echo "[GPU ${GPU_ID}] WSRL (SAC-BC, uncert_inverse) from CalQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_CALQL_REDQ} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 256 \
  --warmup_steps 5000 \
  --config.agent_kwargs.bc_steps=${num_online_steps} \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=lagrangian \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-4 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=success \
  --config.agent_kwargs.bc_constraint=0.1 \
  --config.agent_kwargs.bc_target=dataset \
  --config.agent_kwargs.bc_weight_mode=uncert_inverse \
  --config.agent_kwargs.bc_uncert_action_source=dataset \
  --config.agent_kwargs.bc_uncert_q_source=current \
  --config.agent_kwargs.bc_weight_uncert_measure=std \
  --config.agent_kwargs.bc_weight_clip=10.0 \
  --config.agent_kwargs.bc_weight_normalize=True \
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --exp_name wsrl_sacbc_uncert \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] WSRL (SAC-BC, td_inverse) from CalQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_CALQL_REDQ} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 256 \
  --warmup_steps 5000 \
  --config.agent_kwargs.bc_steps=${num_online_steps} \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=lagrangian \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-4 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=success \
  --config.agent_kwargs.bc_constraint=0.1 \
  --config.agent_kwargs.bc_target=dataset \
  --config.agent_kwargs.bc_weight_mode=td_inverse \
  --config.agent_kwargs.bc_uncert_action_source=dataset \
  --config.agent_kwargs.bc_uncert_q_source=current \
  --config.agent_kwargs.bc_weight_uncert_measure=std \
  --config.agent_kwargs.bc_weight_clip=10.0 \
  --config.agent_kwargs.bc_weight_normalize=True \
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --exp_name wsrl_sacbc_td \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] WSRL (SAC-BC, none) from CalQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:antmaze_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_CALQL_REDQ} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 256 \
  --warmup_steps 5000 \
  --config.agent_kwargs.bc_steps=${num_online_steps} \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=lagrangian \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_lagrangian_lr=1e-4 \
  --config.agent_kwargs.bc_drop_metric=relative \
  --config.agent_kwargs.bc_perf_source=success \
  --config.agent_kwargs.bc_constraint=0.1 \
  --config.agent_kwargs.bc_target=dataset \
  --config.agent_kwargs.bc_weight_mode=none \
  --config.agent_kwargs.bc_uncert_action_source=dataset \
  --config.agent_kwargs.bc_uncert_q_source=current \
  --config.agent_kwargs.bc_weight_uncert_measure=std \
  --config.agent_kwargs.bc_weight_clip=10.0 \
  --config.agent_kwargs.bc_weight_normalize=True \
  --config.agent_kwargs.bc_teacher_deterministic=True \
  --exp_name wsrl_sacbc_none \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."
