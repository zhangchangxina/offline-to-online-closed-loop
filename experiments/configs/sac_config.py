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
    config.bc_loss_weight = 0.0
    config.bc_combine_mode = "sum"  # "sum" | "interpolate"
    config.bc_teacher_deterministic = True
    config.bc_teacher_eval_mode = True  # teacher uses train=False (no dropout)
    # Unified target: "dataset" | "actor_target" | path to offline checkpoint
    config.bc_target = "dataset"
    # Online gating: 0=off; >0: window; -1: always on after online start
    config.bc_online_enable_for_steps = -1
    # Set at online switch time in finetune.py
    config.bc_online_start_step = -1

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
