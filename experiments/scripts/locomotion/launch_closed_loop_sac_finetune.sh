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
--agent closed_loop_sac \
--config experiments/configs/train_config.py:locomotion_closed_loop_sac \
--project method-section \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env halfcheetah-medium-replay-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--config.agent_kwargs.policy_loss_variant=q_trust \
--config.agent_kwargs.q_trust_beta=2.0 \
--config.agent_kwargs.lambda_schedule=linear \
--config.agent_kwargs.lam_align=1 \
--config.agent_kwargs.align_steps=100000 \
--config.agent_kwargs.align_constraint=0.1 \
--config.agent_kwargs.align_lagrange_optimizer_kwargs.learning_rate=1e-3 \
$@
