export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia 


export DATA_DIR_PREFIX=/media/nudt3090/XYQ/ZCX/WSRL/datasets/adroit_data
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=1

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_wsrl \
--project method-section \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env pen-binary-v0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@

# Example: SAC-BC variant (replace --agent sac with --agent sac_bc to enable)
# python3 finetune.py \
# --agent sac_bc \
# --config experiments/configs/train_config.py:adroit_wsrl \
# --project method-section \
# --num_offline_steps 20_000 \
# --reward_scale 10.0 \
# --reward_bias 5.0 \
# --env pen-binary-v0 \
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
