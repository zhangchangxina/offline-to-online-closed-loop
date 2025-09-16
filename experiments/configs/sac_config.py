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

    # BC augmentation defaults (used by SACBCWithTargetAgent)
    # For base SAC: optional BC loss blending weight
    config.bc_loss_weight = 0.0
    config.bc_lambda_init = 0.0
    config.bc_combine_mode = "sum"  # "sum" | "interpolate"
    config.bc_teacher_deterministic = True
    config.bc_teacher_eval_mode = True  # teacher uses train=False (no dropout)
    # Unified target: "dataset" | "actor_target" | path to offline checkpoint
    config.bc_target = "dataset"
    # Lagrange/schedule options for BC weighting
    config.bc_lambda_schedule = "fixed"  # "fixed" | "adaptive" | "linear" | "exp" | "exp_decay" | "fast_slow"
    config.bc_lambda_exp_rate = 5.0  # decay rate for exponential schedule
    # Constraint mode for adaptive BC Lagrange (supported in SAC-BC agent)
    # one of: "bc_loss" | "q_drop" | "j_drop"
    config.bc_constraint_mode = "bc_loss"
    config.bc_constraint = 0.1
    # Reference for q_drop constraint: "dataset" | "actor_target" | "offline_checkpoint"
    config.bc_qdrop_reference = "dataset"
    # Adaptive q_drop constraint options within a batch
    # one of: "none" | "batch_quantile" | "batch_normalized"
    config.bc_qdrop_adaptive_mode = "none"
    config.bc_qdrop_quantile = 0.9
    config.bc_qdrop_ema_alpha = 0.0  # placeholder (not used without host-side update)
    config.bc_qdrop_norm_c = 0.5
    config.bc_qdrop_eps = 1e-6
    # Q drop metric: "absolute" | "relative" | "percent" | "percentage"
    config.bc_qdrop_metric = "absolute"
    config.bc_qdrop_rel_eps = 1e-6  # epsilon for relative drop computation

    # Unified drop metric aliases for both internal (j_drop/q_drop) and external paths
    # Preferred to set these (specific keys still supported and take precedence)
    config.bc_drop_metric = "absolute"  # or "relative"|"percent"|"percentage"
    config.bc_drop_rel_eps = 1e-6

    # External lambda control defaults
    # Mode: "proportional" (new_lambda = k * J_drop) or "dual_ascent" (stateful)
    config.bc_lambda_external_mode = "proportional"
    # Unified LR controlling both internal Lagrange (if set) and external defaults
    config.bc_lagrangian_lr = 3e-4

    # Performance source for drop computation: "return" | "success"
    config.bc_perf_source = "return"

    # Online-only BC gating helper removed; use bc_steps windowing instead
    # Optimizer kwargs container to allow CLI subkey overrides like
    # --config.agent_kwargs.bc_lagrange_optimizer_kwargs.learning_rate=1e-3
    config.bc_lagrange_optimizer_kwargs = ConfigDict(dict(
        learning_rate=3e-4,
    ))
    config.bc_steps = -1  # >0 uses [start, start+bc_steps); <=0 disables

    # Simplified BC per-sample weighting mode (preferred API)
    # one of: "none" | "td" | "td_inverse" | "uncert" | "uncert_inverse"
    # - td:        weights = |q_delta|
    # - td_inverse:weights = 1/(|q_delta|+eps)
    # - uncert:    weights = std_ensemble(Q(s,a))
    # - uncert_inverse: weights = 1/(std_ensemble(Q)+eps)
    config.bc_weight_mode = "none"
    config.bc_weight_eps = 1e-3
    config.bc_weight_scale = 1.0
    config.bc_weight_clip = -1.0    # <=0 disables clip
    config.bc_weight_normalize = False
    # Uncertainty options
    config.bc_uncert_q_source = "current"  # target|current|q
    config.bc_uncert_action_source = "policy"  # bc|policy|dataset|teacher
    config.bc_weight_uncert_measure = "std"  # std|var

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config
