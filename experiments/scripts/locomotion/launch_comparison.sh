export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 

export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

# env: halfcheetah-medium-replay-v2, halfcheetah-medium-v2, hopper-medium-replay-v2, hopper-medium-v2, walker2d-medium-replay-v2, walker2d-medium-v2
ENV=${1:-halfcheetah-medium-replay-v2}

echo "=========================================="
echo "Locomotion Environment Comparison: $ENV"
echo "=========================================="

echo "Running WSRL (Standard SAC)..."
python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:locomotion_wsrl \
--project locomotion-comparison \
--exp_name ${ENV}_wsrl \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env $ENV \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--seed 0

echo "Running WSRL + Closed-Loop SAC..."
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:locomotion_closed_loop_sac \
--project locomotion-comparison \
--exp_name ${ENV}_wsrl_closed_loop \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env $ENV \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--seed 0

echo "=========================================="
echo "Comparison completed for $ENV!"
echo "Check wandb project: locomotion-comparison"
echo "=========================================="
