export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 

export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0
ENV=${1:-pen-binary-v0}

echo "=========================================="
echo "Adroit Environment Comparison: $ENV"
echo "=========================================="

echo "Running WSRL (Standard SAC)..."
python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_wsrl \
--project adroit-comparison \
--exp_name ${ENV}_wsrl \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env $ENV \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--seed 0

echo "Running WSRL + Closed-Loop SAC..."
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:adroit_closed_loop_sac \
--project adroit-comparison \
--exp_name ${ENV}_wsrl_closed_loop \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env $ENV \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--seed 0

echo "=========================================="
echo "Comparison completed for $ENV!"
echo "Check wandb project: adroit-comparison"
echo "=========================================="
