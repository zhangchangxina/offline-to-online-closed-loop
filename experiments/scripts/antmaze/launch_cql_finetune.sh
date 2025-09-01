export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 

export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=2

 
# export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210 
# export MUJOCO_PY_MUJOCO_LIB=$HOME/.mujoco/mujoco210/bin/libmujoco210.so 


python finetune.py \
--agent cql \
--config experiments/configs/train_config.py:antmaze_cql \
--project baselines-section \
--group no-redq-utd1 \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--env antmaze-large-diverse-v2 \
$@


#  FLAX_USE_ORBAX_CHECKPOINTING=false D4RL_SUPPRESS_IMPORT_ERROR=1 PYOPENGL_PLATFORM=egl MUJOCO_GL=egl MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210 MUJOCO_PY_MUJOCO_LIB=$HOME/.mujoco/mujoco210/bin/libmujoco210.so LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia CUDA_VISIBLE_DEVICES=0 bash experiments/scripts/antmaze/launch_cql_finetune.sh --debug True --num_offline_steps 100 --eval_interval 1000000 --save_interval 1000000 --log_interval 50 --env antmaze-medium-play-v2 | cat