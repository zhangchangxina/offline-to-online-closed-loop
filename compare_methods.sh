#!/bin/bash

# Comparison script for different method combinations
# This script runs experiments to compare:
# 1. Standard SAC + WSRL
# 2. Closed-Loop SAC + WSRL
# 3. Closed-Loop SAC (standalone)

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 

export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Method Comparison Experiments"
echo "=========================================="

# Environment settings
ENV="antmaze-large-diverse-v2"
REWARD_SCALE=10.0
REWARD_BIAS=-5.0
NUM_OFFLINE_STEPS=1_000_000
WARMUP_STEPS=5000
NUM_ONLINE_STEPS=500_000
BATCH_SIZE=1024
UTD=4

echo "Environment: $ENV"
echo "Offline steps: $NUM_OFFLINE_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Online steps: $NUM_ONLINE_STEPS"
echo ""

# Experiment 1: Standard SAC + WSRL
echo "=========================================="
echo "Experiment 1: Standard SAC + WSRL"
echo "=========================================="

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--project method-comparison \
--exp_name standard_sac_wsrl \
--reward_scale $REWARD_SCALE \
--reward_bias $REWARD_BIAS \
--num_offline_steps $NUM_OFFLINE_STEPS \
--env $ENV \
--utd $UTD \
--batch_size $BATCH_SIZE \
--warmup_steps $WARMUP_STEPS \
--num_online_steps $NUM_ONLINE_STEPS \
--offline_data_ratio 0.0 \
--online_sampling_method mixed \
--online_use_cql_loss False \
--seed 0

echo "Standard SAC + WSRL completed!"
echo ""

# Experiment 2: Closed-Loop SAC + WSRL
echo "=========================================="
echo "Experiment 2: Closed-Loop SAC + WSRL"
echo "=========================================="

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project method-comparison \
--exp_name closed_loop_sac_wsrl \
--reward_scale $REWARD_SCALE \
--reward_bias $REWARD_BIAS \
--num_offline_steps $NUM_OFFLINE_STEPS \
--env $ENV \
--utd $UTD \
--batch_size $BATCH_SIZE \
--warmup_steps $WARMUP_STEPS \
--num_online_steps $NUM_ONLINE_STEPS \
--offline_data_ratio 0.0 \
--online_sampling_method mixed \
--online_use_cql_loss False \
--seed 0

echo "Closed-Loop SAC + WSRL completed!"
echo ""

# Experiment 3: Closed-Loop SAC (standalone, no WSRL)
echo "=========================================="
echo "Experiment 3: Closed-Loop SAC (standalone)"
echo "=========================================="

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project method-comparison \
--exp_name closed_loop_sac_standalone \
--reward_scale $REWARD_SCALE \
--reward_bias $REWARD_BIAS \
--num_offline_steps 0 \
--env $ENV \
--utd $UTD \
--batch_size $BATCH_SIZE \
--warmup_steps 0 \
--num_online_steps $NUM_ONLINE_STEPS \
--offline_data_ratio 0.0 \
--online_sampling_method mixed \
--online_use_cql_loss False \
--seed 0

echo "Closed-Loop SAC (standalone) completed!"
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results will be available in wandb project: method-comparison"
echo ""
echo "Experiment summary:"
echo "1. standard_sac_wsrl: Standard SAC with WSRL warm-start"
echo "2. closed_loop_sac_wsrl: Closed-Loop SAC with WSRL warm-start"
echo "3. closed_loop_sac_standalone: Closed-Loop SAC without warm-start"
echo ""
echo "Compare the results to see the benefits of:"
echo "- WSRL warm-start vs no warm-start"
echo "- Closed-loop updates vs standard updates"
echo "- Combined approach vs individual methods"
