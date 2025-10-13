from ml_collections import ConfigDict


def get_config(updates=None):
    config = ConfigDict()

    # AWAC specific parameters
    config.discount = 0.99
    config.expectile = 0.9
    config.temperature = 1.0
    config.target_update_rate = 5e-3
    config.update_actor_with_target_adv = True
    config.adv_clip_max = 100.0

    # Network architecture
    config.critic_ensemble_size = 2
    config.critic_subsample_size = None

    config.policy_network_kwargs = ConfigDict(
        dict(
            hidden_dims=(256, 256),
            kernel_init_type="var_scaling",
            kernel_scale_final=1e-2,
        )
    )
    config.critic_network_kwargs = ConfigDict(
        dict(
            hidden_dims=(256, 256),
            kernel_init_type="var_scaling",
        )
    )
    config.policy_kwargs = ConfigDict(
        dict(
            tanh_squash_distribution=False,
            std_parameterization="exp",
        )
    )

    # Optimizer settings
    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.value_critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())

    return config
