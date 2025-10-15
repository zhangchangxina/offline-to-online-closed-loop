export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 


export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_DATASET_DIR=../datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1
python3 finetune.py \
--agent cql \
--config experiments/configs/train_config.py:locomotion_cql \
--env halfcheetah-medium-replay-v2 \
--project locomotion-finetune \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
$@
