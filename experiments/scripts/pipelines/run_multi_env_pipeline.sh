#!/usr/bin/env bash

set -euo pipefail

# Usage: bash experiments/scripts/pipelines/run_multi_env_pipeline.sh <ENV_CATEGORY> <GPU_ID>
# ENV_CATEGORY options: locomotion, adroit, antmaze, kitchen, maze2d, bullet, flow, carla, minigrid

ENV_CATEGORY=${1:-locomotion}
GPU_ID=${2:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 关键优化：设置环境变量避免 JIT 重新编译
export TF_ENABLE_ONEDNN_OPTS=0
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
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top

SAVE_ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log"
PROJECT_DIR="wsrl"

# Define environment lists for each category
declare -A ENV_LISTS
ENV_LISTS[locomotion]="halfcheetah-medium-v0 hopper-medium-v0 walker2d-medium-v0 ant-medium-v0"
ENV_LISTS[adroit]="pen-cloned-v0 hammer-cloned-v0 relocate-cloned-v0 door-cloned-v0"
ENV_LISTS[antmaze]="antmaze-umaze-v2 antmaze-medium-play-v2 antmaze-large-play-v2"
ENV_LISTS[kitchen]="kitchen-partial-v0 kitchen-complete-v0 kitchen-mixed-v0"
ENV_LISTS[maze2d]="maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1"
ENV_LISTS[bullet]="bullet-halfcheetah-medium-v0 bullet-hopper-medium-v0 bullet-walker2d-medium-v0 bullet-ant-medium-v0"
ENV_LISTS[flow]="flow-ring-random-v0 flow-ring-controller-v0 flow-merge-random-v0 flow-merge-controller-v0"
ENV_LISTS[carla]="carla-lane-v0 carla-town-v0 carla-town-full-v0"
ENV_LISTS[minigrid]="minigrid-fourrooms-v0 minigrid-fourrooms-random-v0"

# Define configuration and parameters for each category
declare -A CONFIG_NAMES
CONFIG_NAMES[locomotion]="locomotion"
CONFIG_NAMES[adroit]="adroit"
CONFIG_NAMES[antmaze]="antmaze"
CONFIG_NAMES[kitchen]="kitchen"
CONFIG_NAMES[maze2d]="locomotion"  # Use locomotion config for maze2d
CONFIG_NAMES[bullet]="locomotion"  # Use locomotion config for bullet
CONFIG_NAMES[flow]="locomotion"    # Use locomotion config for flow
CONFIG_NAMES[carla]="locomotion"   # Use locomotion config for carla
CONFIG_NAMES[minigrid]="locomotion" # Use locomotion config for minigrid

# Define reward scaling for each category
declare -A REWARD_SCALES
REWARD_SCALES[locomotion]="1.0"
REWARD_SCALES[adroit]="10.0"
REWARD_SCALES[antmaze]="10.0"
REWARD_SCALES[kitchen]="1.0"
REWARD_SCALES[maze2d]="1.0"
REWARD_SCALES[bullet]="1.0"
REWARD_SCALES[flow]="1.0"
REWARD_SCALES[carla]="1.0"
REWARD_SCALES[minigrid]="1.0"

declare -A REWARD_BIASES
REWARD_BIASES[locomotion]="0.0"
REWARD_BIASES[adroit]="5.0"
REWARD_BIASES[antmaze]="-5.0"
REWARD_BIASES[kitchen]="-4.0"
REWARD_BIASES[maze2d]="0.0"
REWARD_BIASES[bullet]="0.0"
REWARD_BIASES[flow]="0.0"
REWARD_BIASES[carla]="0.0"
REWARD_BIASES[minigrid]="0.0"

# Define training steps for each category
declare -A OFFLINE_STEPS
OFFLINE_STEPS[locomotion]="250000"
OFFLINE_STEPS[adroit]="20000"
OFFLINE_STEPS[antmaze]="1000000"
OFFLINE_STEPS[kitchen]="250000"
OFFLINE_STEPS[maze2d]="500000"
OFFLINE_STEPS[bullet]="250000"
OFFLINE_STEPS[flow]="200000"
OFFLINE_STEPS[carla]="300000"
OFFLINE_STEPS[minigrid]="100000"

# Get environment list for the category
ENVS=(${ENV_LISTS[$ENV_CATEGORY]})
CONFIG_NAME=${CONFIG_NAMES[$ENV_CATEGORY]}
R_SCALE=${REWARD_SCALES[$ENV_CATEGORY]}
R_BIAS=${REWARD_BIASES[$ENV_CATEGORY]}
OFFLINE_STEPS=${OFFLINE_STEPS[$ENV_CATEGORY]}

echo "[GPU ${GPU_ID}] Running ${ENV_CATEGORY} pipeline for environments: ${ENVS[*]}"

# Function to run pipeline for a single environment
run_single_env() {
    local ENV_ID=$1
    local SEED=0
    
    echo "[GPU ${GPU_ID}] =========================================="
    echo "[GPU ${GPU_ID}] Starting pipeline for ${ENV_ID}"
    echo "[GPU ${GPU_ID}] =========================================="
    
    # Step 1: CALQL pretraining
    echo "[GPU ${GPU_ID}] CALQL (REDQ10, UTD=4) pretrain for ${ENV_ID}"
    python3 finetune.py \
      --agent calql \
      --config experiments/configs/train_config.py:${CONFIG_NAME}_cql \
      --env ${ENV_ID} \
      --seed ${SEED} \
      --use_redq True \
      --utd 4 \
      --reward_scale ${R_SCALE} \
      --reward_bias ${R_BIAS} \
      --num_offline_steps ${OFFLINE_STEPS} \
      --save_interval ${OFFLINE_STEPS} \
      --exp_name calql_ensemble_highutd \
      --save_dir ${SAVE_ROOT} \
      2>&1 | tee -a ${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log
    
    # Find the checkpoint path
    EXP_DESC="calql_ensemble_highutd_${ENV_ID}_calql_seed${SEED}"
    RUN_DIR=$(ls -1dt ${SAVE_ROOT}/${PROJECT_DIR}/${EXP_DESC}_* | head -n 1)
    CKPT_PATH="${RUN_DIR}/checkpoint_${OFFLINE_STEPS}"
    echo "[GPU ${GPU_ID}] Using checkpoint: ${CKPT_PATH}"
    
    # Step 2: WSRL (SAC)
    echo "[GPU ${GPU_ID}] WSRL (SAC) from CALQL for ${ENV_ID}"
    python3 finetune.py \
      --agent sac \
      --config experiments/configs/train_config.py:${CONFIG_NAME}_wsrl \
      --env ${ENV_ID} \
      --seed ${SEED} \
      --use_redq True \
      --reward_scale ${R_SCALE} \
      --reward_bias ${R_BIAS} \
      --resume_path ${CKPT_PATH} \
      --num_offline_steps ${OFFLINE_STEPS} \
      --utd 4 \
      --batch_size 1024 \
      --warmup_steps 5000 \
      --exp_name wsrl \
      --save_dir ${SAVE_ROOT} | cat
    
    # Step 3: WSRL (SAC-BC)
    echo "[GPU ${GPU_ID}] WSRL (SAC-BC) from CALQL for ${ENV_ID}"
    python3 finetune.py \
      --agent sac_bc \
      --config experiments/configs/train_config.py:${CONFIG_NAME}_wsrl \
      --env ${ENV_ID} \
      --seed ${SEED} \
      --use_redq True \
      --reward_scale ${R_SCALE} \
      --reward_bias ${R_BIAS} \
      --resume_path ${CKPT_PATH} \
      --num_offline_steps ${OFFLINE_STEPS} \
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
    
    echo "[GPU ${GPU_ID}] Completed pipeline for ${ENV_ID}"
}

# Run pipeline for each environment in the category
for ENV_ID in "${ENVS[@]}"; do
    # Check availability before running
    python3 - <<PY
import sys
try:
    import gym, d4rl  # noqa: F401
    gym.make("${ENV_ID}")
    print("[OK] ${ENV_ID} available.")
except Exception as e:
    print(f"[SKIP] ${ENV_ID} unavailable: {e}")
    sys.exit(2)
PY
    if [ $? -eq 0 ]; then
        run_single_env "${ENV_ID}"
    else
        echo "[GPU ${GPU_ID}] Skipping ${ENV_ID} due to unavailability."
    fi
done

echo "[GPU ${GPU_ID}] =========================================="
echo "[GPU ${GPU_ID}] All ${ENV_CATEGORY} pipelines completed!"
echo "[GPU ${GPU_ID}] =========================================="
