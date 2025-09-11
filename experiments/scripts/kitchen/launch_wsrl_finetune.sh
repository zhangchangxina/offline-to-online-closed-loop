export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
# Ensure NVIDIA libs are available for EGL; avoid legacy mujoco210 forcing
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_VISIBLE_DEVICES=0

# Default: plain SAC
python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:kitchen_wsrl \
--project kitchen-finetune \
--num_offline_steps 250_000 \
--reward_scale 1.0 \
--reward_bias -4.0 \
--env kitchen-partial-v0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@

# Example: SAC-BC variant (enable by replacing --agent sac with --agent sac_bc)
# python3 finetune.py \
# --agent sac_bc \
# --config experiments/configs/train_config.py:kitchen_wsrl \
# --project kitchen-finetune \
# --num_offline_steps 250_000 \
# --reward_scale 1.0 \
# --reward_bias -4.0 \
# --env kitchen-partial-v0 \
# --utd 4 \
# --batch_size 1024 \
# --warmup_steps 5000 \
# --config.agent_kwargs.bc_loss_weight=1.0 \
# --config.agent_kwargs.bc_target=actor_target \
# --config.agent_kwargs.bc_weight_mode=td \
# --config.agent_kwargs.bc_weight_clip=10.0 \
# $@
