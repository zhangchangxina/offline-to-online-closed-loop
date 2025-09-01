export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 

export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=0

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--project baselines-section \
--config.agent_kwargs.critic_subsample_size 1 \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 0 \
--num_online_steps 500_000 \
--offline_data_ratio 0.5 \
--env antmaze-large-diverse-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@
