export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 


export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:locomotion_wsrl \
--project method-section \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env halfcheetah-medium-replay-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@

# Example: SAC-BC variant (replace --agent sac with --agent sac_bc to enable)
# python3 finetune.py \
# --agent sac_bc \
# --config experiments/configs/train_config.py:locomotion_wsrl \
# --project method-section \
# --reward_scale 1.0 \
# --reward_bias 0.0 \
# --num_offline_steps 250_000 \
# --env halfcheetah-medium-replay-v2 \
# --utd 4 \
# --batch_size 1024 \
# --warmup_steps 5000 \
# --config.agent_kwargs.bc_loss_weight=1.0 \
# --config.agent_kwargs.bc_target=actor_target \
# --config.agent_kwargs.bc_teacher_eval_mode=True \
# --config.agent_kwargs.bc_weight_mode=td \
# --config.agent_kwargs.bc_weight_normalize=True \
# --config.agent_kwargs.bc_weight_clip=10.0 \
# $@
