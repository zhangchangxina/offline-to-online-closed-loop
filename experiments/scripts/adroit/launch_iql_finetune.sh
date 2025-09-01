export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 


export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python finetune.py \
--agent iql \
--config experiments/configs/train_config.py:adroit_iql \
--project baselines-section \
--group no-redq-utd1 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--num_offline_steps 20_000 \
--log_interval 1_000 \
--eval_interval 10_000 \
--save_interval 20_000 \
$@
