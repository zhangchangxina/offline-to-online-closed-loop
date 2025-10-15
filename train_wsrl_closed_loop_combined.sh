#!/bin/bash

# Training script for WSRL + Closed-Loop SAC combination
# This combines the warm-start benefits of WSRL with the stable training of closed-loop updates

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 

export D4RL_DATASET_DIR=../datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "WSRL + Closed-Loop SAC Combination Training"
echo "=========================================="

# Example 1: Antmaze with WSRL + Closed-Loop SAC
echo "Training WSRL + Closed-Loop SAC on antmaze-large-diverse-v2..."

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project wsrl-closed-loop-combined \
--exp_name antmaze_wsrl_closed_loop \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--env antmaze-large-diverse-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--num_online_steps 500_000 \
--offline_data_ratio 0.0 \
--online_sampling_method mixed \
--online_use_cql_loss False \
--seed 0

# Example 2: Adroit with WSRL + Closed-Loop SAC
echo "Training WSRL + Closed-Loop SAC on pen-binary-v0..."

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:adroit_closed_loop_sac \
--project wsrl-closed-loop-combined \
--exp_name adroit_wsrl_closed_loop \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env pen-binary-v0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--num_online_steps 500_000 \
--offline_data_ratio 0.0 \
--online_sampling_method mixed \
--online_use_cql_loss False \
--seed 0

# Example 3: Locomotion with WSRL + Closed-Loop SAC
echo "Training WSRL + Closed-Loop SAC on halfcheetah-medium-replay-v2..."

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:locomotion_closed_loop_sac \
--project wsrl-closed-loop-combined \
--exp_name locomotion_wsrl_closed_loop \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env halfcheetah-medium-replay-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--num_online_steps 500_000 \
--offline_data_ratio 0.0 \
--online_sampling_method mixed \
--online_use_cql_loss False \
--seed 0

echo "=========================================="
echo "WSRL + Closed-Loop SAC training completed!"
echo "=========================================="
echo ""
echo "This combination provides:"
echo "1. WSRL: Warm-start with offline data collection"
echo "2. Closed-Loop: Stable online fine-tuning with alignment constraints"
echo "3. Enhanced monitoring: Both WSRL and closed-loop metrics"
