#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a
# Usage: bash experiments/scripts/pipelines/run_kitchen_pipeline.sh <GPU_ID>

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}


ENV_ID="kitchen-complete-v0"
SEED=1
PROJECT_DIR="wsrl"

R_SCALE=1.0
R_BIAS=-4.0

num_offline_steps=300000
num_online_steps=300000
save_interval=100000

# Ensure MuJoCo 2.1.0 binaries are on the path for D4RL kitchen mujoco_py
# export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin



# echo "[GPU ${GPU_ID}] CQL pretrain for ${ENV_ID}"
# python3 finetune.py \
#   --agent cql \
#   --config experiments/configs/train_config.py:kitchen_cql \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --num_offline_steps ${num_offline_steps} \
#   --num_online_steps ${num_online_steps} \
#   --save_interval ${save_interval} \
#   --exp_name cql \
#   --save_dir ${SAVE_ROOT} \
#   --online_sampling_method append \
#   2>&1 | tee -a ${SAVE_ROOT}/cql_${ENV_ID}_seed${SEED}.log


# echo "[GPU ${GPU_ID}] IQL pretrain for ${ENV_ID}"
# python3 finetune.py \
#   --agent iql \
#   --config experiments/configs/train_config.py:kitchen_iql \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --num_offline_steps ${num_offline_steps} \
#   --num_online_steps ${num_online_steps} \
#   --save_interval ${save_interval} \
#   --exp_name iql \
#   --save_dir ${SAVE_ROOT} \
#   --online_sampling_method append \
#   2>&1 | tee -a ${SAVE_ROOT}/iql_${ENV_ID}_seed${SEED}.log


# echo "[GPU ${GPU_ID}] RLPD (SAC with offline data ratio=0.5) for ${ENV_ID}"
# python3 finetune.py \
#   --agent sac \
#   --config experiments/configs/train_config.py:kitchen_wsrl \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --num_offline_steps 0 \
#   --offline_data_ratio 0.5 \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --num_online_steps ${num_online_steps} \
#   --save_interval ${save_interval} \
#   --utd 4 \
#   --batch_size 1024 \
#   --warmup_steps 5000 \
#   --exp_name rlpd \
#   --save_dir ${SAVE_ROOT} | cat


# echo "[GPU ${GPU_ID}] AWAC from CALQL-20K for ${ENV_ID}"
# python3 finetune.py \
#   --agent awac \
#   --config experiments/configs/train_config.py:kitchen_awac \
#   --env ${ENV_ID} \
#   --seed ${SEED} \
#   --use_redq True \
#   --reward_scale ${R_SCALE} \
#   --reward_bias ${R_BIAS} \
#   --num_offline_steps ${num_offline_steps} \
#   --num_online_steps ${num_online_steps} \
#   --save_interval ${save_interval} \
#   --utd 4 \
#   --batch_size 1024 \
#   --online_sampling_method append \
#   --exp_name awac \
#   --save_dir ${SAVE_ROOT} | cat


echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:kitchen_cql \
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
  --save_dir ${SAVE_ROOT} \
  2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log



EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
CKPT_PATH="${RUN_DIR}/checkpoint_${num_offline_steps}"
echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"



echo "[GPU ${GPU_ID}] CALQL-APPEND (REDQ10, UTD=4) pretrain for ${ENV_ID}"
python3 finetune.py \
  --agent calql \
  --config experiments/configs/train_config.py:kitchen_cql \
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
  --save_dir ${SAVE_ROOT} \
  2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log


echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL-20K for ${ENV_ID}"
python3 finetune.py \
  --agent sac \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --exp_name wsrl \
  --save_dir ${SAVE_ROOT} | cat






echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-20K for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --warmup_update_critic True \
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=lagrangian \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_qdrop_reference=dataset \
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
  --save_dir ${SAVE_ROOT} | cat




  echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-20K for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --warmup_update_critic True \
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=lagrangian \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_qdrop_reference=dataset \
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
  --save_dir ${SAVE_ROOT} | cat


  echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL-20K for ${ENV_ID}"
python3 finetune.py \
  --agent sac_bc \
  --config experiments/configs/train_config.py:kitchen_wsrl \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --use_redq True \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --resume_path ${CKPT_PATH} \
  --num_offline_steps ${num_offline_steps} \
  --num_online_steps ${num_online_steps} \
  --utd 4 \
  --batch_size 1024 \
  --warmup_steps 5000 \
  --warmup_update_critic True \
  --config.agent_kwargs.bc_steps=300000 \
  --config.agent_kwargs.bc_lambda_init=1 \
  --config.agent_kwargs.bc_lambda_schedule=lagrangian \
  --config.agent_kwargs.bc_constraint_mode=j_drop \
  --config.agent_kwargs.bc_qdrop_reference=dataset \
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
  --save_dir ${SAVE_ROOT} | cat


echo "[GPU ${GPU_ID}] Pipeline for ${ENV_ID} completed."


