export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 


export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

echo "=== 快速训练模式 ==="
echo "步数: 1000 (离线) + 1000 (在线) = 2000 步"
echo "Batch size: 256"
echo "关闭评估和保存"

python finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--project fast-test \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1000 \
--num_online_steps 1000 \
--env antmaze-large-diverse-v2 \
--utd 1 \
--batch_size 256 \
--log_interval 100 \
--eval_interval 500 \
--save_interval 500 \
--debug
