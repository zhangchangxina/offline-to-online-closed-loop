# export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export PYOPENGL_PLATFORM=egl
# export MUJOCO_GL=egl
# export WANDB_BASE_URL=https://api.bandw.top
# export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
# export MUJOCO_PY_MUJOCO_LIB=$HOME/.mujoco/mujoco210/bin/libmujoco210.so
# export D4RL_SUPPRESS_IMPORT_ERROR=1
# export PYTHONPATH=/media/nudt3090/XYQ/ZCX/WSRL/wsrl-main/D4RL:${PYTHONPATH}
# export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
# export CUDA_VISIBLE_DEVICES=0

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 

export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

# env: antmaze-large-diverse-v2, antmaze-large-play-v2, antmaze-medium-diverse-v2, antmaze-medium-play-v2
ENV=${1:-antmaze-large-diverse-v2}

echo "=========================================="
echo "Antmaze Environment Comparison: $ENV"
echo "=========================================="

echo "Running WSRL (Standard SAC)..."
python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--project antmaze-comparison \
--exp_name ${ENV}_wsrl \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--env $ENV \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--seed 0

echo "Running WSRL + Closed-Loop SAC..."
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project antmaze-comparison \
--exp_name ${ENV}_wsrl_closed_loop \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--env $ENV \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--seed 0

echo "=========================================="
echo "Comparison completed for $ENV!"
echo "Check wandb project: antmaze-comparison"
echo "=========================================="
