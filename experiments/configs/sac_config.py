from ml_collections import ConfigDict


def get_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.soft_target_update_rate = 5e-3
    config.critic_ensemble_size = 2
    config.critic_subsample_size = None
    config.autotune_entropy = True
    config.temperature_init = 1.0

    # arch
    config.critic_network_kwargs = ConfigDict(
        {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        }
    )
    config.policy_network_kwargs = ConfigDict(
        {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        }
    )
    config.policy_kwargs = ConfigDict(
        {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        }
    )

    # diagnostics/logging (allow CLI overrides like --config.agent_kwargs.log_actor_grad_terms)
    config.log_actor_grad_terms = False
    config.actor_log_std_layer_name = "Dense_1"

    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 1e-4,
        }
    )
    config.critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.temperature_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 1e-4,
        }
    )

    # BC augmentation defaults (used by SACBCWithTargetAgent and can be set for SAC variants)
    config.bc_loss_weight = 0.0
    config.bc_mode = "dataset"  # "dataset" | "actor_target"
    config.bc_combine_mode = "sum"  # "sum" | "interpolate"
    config.bc_teacher_deterministic = True
    config.bc_teacher_eval_mode = True  # If True, teacher uses train=False (no dropout)
    # Unified target selector: "dataset" | "actor_target" | "offline_checkpoint"
    # If not set, code will fall back to bc_mode/bc_teacher_source for backward-compat
    config.bc_target = "dataset"
    config.bc_teacher_source = "actor_target"  # legacy; kept for backward-compat
    config.bc_offline_ckpt_teacher = ""  # path to offline teacher checkpoint dir
    # Enable BC only for first N ONLINE steps; <0 disables online gating
    config.bc_online_enable_for_steps = -1
    # Will be set at online switch time in finetune.py
    config.bc_online_start_step = -1
    config.bc_td_weight_enabled = False
    config.bc_td_weight_abs = True
    config.bc_td_weight_power = 1.0
    config.bc_td_weight_scale = 1.0
    # Use a float sentinel for CLI override compatibility; <=0 means "disabled"
    config.bc_td_weight_clip = -1.0
    config.bc_td_weight_normalize = False  # If True, divide weights by mean
    # Optional: inverse mapping (only meaningful when bc_target=="dataset")
    config.bc_td_weight_inverse = False  # If True, weights = 1/(|delta|+eps)
    config.bc_td_weight_inverse_eps = 1e-3

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config
