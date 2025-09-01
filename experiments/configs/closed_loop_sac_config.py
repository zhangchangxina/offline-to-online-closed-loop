from ml_collections import ConfigDict

from experiments.configs import sac_config


def get_config(updates=None):
    config = sac_config.get_config()

    # SAC specific settings
    config.critic_ensemble_size = 10
    config.critic_subsample_size = 2

    # Closed-loop update mechanism parameters
    config.closed_loop_enabled = False  # disable during offline to match wsrl
    config.align_constraint = 0.1  # Constraint for E[q_delta^2] <= c
    config.lam_align = 1.0  # Weight for alignment loss
    # Lambda weighting schedule: "fixed" | "linear" | "adaptive"
    # - fixed: lam_align = lam_align_init
    # - linear: lam_align = lam_align_init × (1 - progress)
    # - adaptive: lam_align = align_lagrange (learned by positive_violation)
    config.lambda_schedule = "fixed"
    config.lam_eff_linear_steps = 10000  # used when lambda_schedule == "linear"
    config.lam_eff_linear_start_step = 0  # allow offsetting the schedule start (e.g., after resume)
    # For adaptive, align_lagrange init defaults to lam_align unless overridden in agent.create

    # Default optimizer settings for the align Lagrange multiplier
    # Added to allow subkey CLI overrides like
    # --config.agent_kwargs.align_lagrange_optimizer_kwargs.learning_rate=1e-3
    config.align_lagrange_optimizer_kwargs = ConfigDict(dict(
        learning_rate=3e-4,
    ))

    # Network architecture
    config.policy_network_kwargs.use_layer_norm = True
    config.critic_network_kwargs.use_layer_norm = True

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
