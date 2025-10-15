export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 

export D4RL_DATASET_DIR=../datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top 
# export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0


python finetune.py \
--agent calql \
--config experiments/configs/train_config.py:antmaze_cql \
--project baselines-section \
--group no-redq-utd1 \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--env antmaze-umaze-v2 \
$@
