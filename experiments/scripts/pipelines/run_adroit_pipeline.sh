#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}


# door-binary-v0, pen-binary-v0, relocate-binary-v0
ENV_ID="pen-binary-v0"
SEED=0
PROJECT_DIR="wsrl"

# Adroit binary (sparse -1/0) recommended scaling
R_SCALE=10.0
R_BIAS=5.0

num_offline_steps=300000
num_online_steps=300000
save_interval=300000

# CQL: run non-append version first
echo "[GPU ${GPU_ID}] CQL pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent cql \
  --config experiments/configs/train_config.py:adroit_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name cql \
  --save_dir ${SAVE_ROOT}

# Get CQL checkpoint path (find directory that has the required checkpoint)
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

# # CQL append: load CQL offline checkpoint
echo "[GPU ${GPU_ID}] CQL append for ${ENV_ID}"
python3 finetune.py \
  --agent cql \
  --config experiments/configs/train_config.py:adroit_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH_CQL} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name cql_append \
  --save_dir ${SAVE_ROOT} \
  --online_sampling_method append

# IQL: run non-append version first
echo "[GPU ${GPU_ID}] IQL pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent iql \
  --config experiments/configs/train_config.py:adroit_iql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name iql \
  --save_dir ${SAVE_ROOT}

# Get IQL checkpoint path (find directory that has the required checkpoint)
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

# IQL append: load IQL offline checkpoint
echo "[GPU ${GPU_ID}] IQL append for ${ENV_ID}"
python3 finetune.py \
  --agent iql \
  --config experiments/configs/train_config.py:adroit_iql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --resume_path ${CKPT_PATH_IQL} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name iql_append \
  --save_dir ${SAVE_ROOT} \
  --online_sampling_method append

# echo "[GPU ${GPU_ID}] RLPD (SAC with offline data ratio=0.5) for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:adroit_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --num_offline_steps 0 \
  --offline_data_ratio 0.5 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
  --exp_name rlpd \
  --save_dir ${SAVE_ROOT}

# echo "[GPU ${GPU_ID}] FASTSAC (SAC with high utd) for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:adroit_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --num_offline_steps 0 \
  --offline_data_ratio 0 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
  --exp_name fastsac \
  --save_dir ${SAVE_ROOT}

# AWAC: run non-append version first
echo "[GPU ${GPU_ID}] AWAC pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent awac \
  --config experiments/configs/train_config.py:adroit_awac \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
  --exp_name awac \
  --save_dir ${SAVE_ROOT}

# Get AWAC checkpoint path (find directory that has the required checkpoint)
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

# AWAC append: load AWAC offline checkpoint
echo "[GPU ${GPU_ID}] AWAC append for ${ENV_ID}"
python3 finetune.py \
  --agent awac \
  --config experiments/configs/train_config.py:adroit_awac \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH_AWAC} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
  --online_sampling_method append \
  --exp_name awac_append \
  --save_dir ${SAVE_ROOT}

# CALQL: run non-append version first
echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:adroit_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --utd 4 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --exp_name calql_ensemble_highutd \
  --save_dir ${SAVE_ROOT}

# Get CALQL checkpoint path
# EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
# RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
# CKPT_PATH="${RUN_DIR}/checkpoint_${num_offline_steps}"
# echo "[GPU ${GPU_ID}] Using CALQL checkpoint: ${CKPT_PATH}"

# CALQL append: load CALQL offline checkpoint
echo "[GPU ${GPU_ID}] CALQL-APPEND (REDQ10, UTD=4) for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:adroit_cql \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --resume_path ${CKPT_PATH} \
  --use_redq True \
  --utd 4 \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --online_sampling_method append \
  --exp_name calql_ensemble_highutd_append \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:adroit_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_IQL} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --exp_name wsrl \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:adroit_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_IQL} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
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
  --exp_name wsrl_sacbc \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] WSRL (SAC-BC, td_inverse) from CALQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:adroit_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_IQL} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
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
  --exp_name wsrl_sacbc \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] WSRL (SAC-BC, none) from CALQL for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:adroit_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH_IQL} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --save_interval ${save_interval} \
  --utd 4 \
  --batch_size 1024 \
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
  --exp_name wsrl_sacbc \
  --save_dir ${SAVE_ROOT}

echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."
