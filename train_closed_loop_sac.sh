#!/bin/bash

# Example training script for SAC with closed-loop updates
# This script shows how to train the SAC agent with the closed-loop mechanism

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 

export D4RL_DATASET_DIR=../datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

# Example 1: Antmaze with closed-loop SAC
echo "Training SAC with closed-loop updates on antmaze-large-diverse-v2..."

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project sac-closed-loop \
--exp_name antmaze_closed_loop \
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

# Example 2: Adroit with closed-loop SAC
echo "Training SAC with closed-loop updates on pen-binary-v0..."

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:adroit_closed_loop_sac \
--project sac-closed-loop \
--exp_name adroit_closed_loop \
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

# Example 3: Locomotion with closed-loop SAC
echo "Training SAC with closed-loop updates on halfcheetah-medium-replay-v2..."

python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:locomotion_closed_loop_sac \
--project sac-closed-loop \
--exp_name locomotion_closed_loop \
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

echo "Training completed! Check the logs for results."
