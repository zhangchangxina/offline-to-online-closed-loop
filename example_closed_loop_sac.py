#!/usr/bin/env python3
"""
Example script demonstrating the ClosedLoopSACAgent.

This script shows how to:
1. Create a SAC agent with closed-loop updates
2. Train it on a simple environment
3. Use the alignment loss and Lagrangian constraints
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Set up environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

from wsrl.agents import ClosedLoopSACAgent
from wsrl.common.common import JaxRLTrainState
from wsrl.networks.mlp import MLP
from wsrl.networks.actor_critic_nets import Policy, Critic, ensemblize
from wsrl.networks.lagrange import GeqLagrangeMultiplier
from wsrl.common.optimizers import make_optimizer


def create_simple_encoder():
    """Create a simple encoder for demonstration"""
    return MLP(hidden_dims=[64, 64], activate_final=True)


def main():
    """Main function demonstrating ClosedLoopSACAgent"""
    
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(42)
    
    # Create dummy data for initialization
    batch_size = 256
    obs_dim = 17  # Typical observation dimension for MuJoCo environments
    action_dim = 6  # Typical action dimension for MuJoCo environments
    
    observations = jax.random.normal(rng, (batch_size, obs_dim))
    actions = jax.random.normal(rng, (batch_size, action_dim))
    
    # Create encoder
    encoder_def = create_simple_encoder()
    
    print("Creating SAC agent with closed-loop update mechanism...")
    
    # Create ClosedLoopSACAgent
    agent = ClosedLoopSACAgent.create(
        rng=rng,
        observations=observations,
        actions=actions,
        encoder_def=encoder_def,
        shared_encoder=True,
        critic_network_kwargs={
            "hidden_dims": [256, 256],
            "activate_final": True,
        },
        policy_network_kwargs={
            "hidden_dims": [256, 256],
            "activate_final": True,
        },
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        critic_ensemble_size=10,
        critic_subsample_size=2,
        temperature_init=1.0,
        # Closed-loop specific parameters
        align_constraint=0.1,  # Constraint for E[q_delta^2] <= 0.1
        lam_align=1.0,  # Weight for alignment loss (init)
        # use lambda_schedule instead of deprecated fields
        lam_eff_linear_steps=100000,  # Steps for linear decay
    )
    
    print(f"Agent created successfully!")
    print(f"Agent config: {agent.config}")
    
    # Create dummy batch for demonstration
    batch = {
        "observations": jax.random.normal(rng, (batch_size, obs_dim)),
        "actions": jax.random.normal(rng, (batch_size, action_dim)),
        "next_observations": jax.random.normal(rng, (batch_size, obs_dim)),
        "rewards": jax.random.normal(rng, (batch_size,)),
        "masks": jnp.ones((batch_size,)),
    }
    
    print("\nPerforming a single update step...")
    
    # Perform a single update step
    updated_agent, info = agent.update(batch)
    
    print("Update completed successfully!")
    print(f"Update info keys: {list(info.keys())}")
    
    # Print some key metrics
    print(f"\nKey metrics from the update:")
    print(f"  Available keys: {list(info.keys())}")
    
    # Print standard metrics if available
    if 'actor' in info:
        actor_info = info['actor']
        print(f"  Actor loss: {actor_info['actor_loss']:.4f}")
        print(f"  Entropy: {actor_info['entropy']:.4f}")
    if 'critic' in info:
        critic_info = info['critic']
        print(f"  Critic loss: {critic_info['critic_loss']:.4f}")
    if 'temperature' in info:
        print(f"  Temperature: {info['temperature']}")
    
    # Print closed-loop specific metrics
    if 'actor' in info and 'align_loss' in info['actor']:
        actor_info = info['actor']
        print(f"\nClosed-loop specific metrics:")
        print(f"  Alignment loss: {actor_info['align_loss']:.4f}")
        print(f"  Constraint violation: {actor_info['constraint_violation']:.4f}")
        print(f"  Q-delta mean: {actor_info['q_delta_mean']:.4f}")
        print(f"  Q-delta std: {actor_info['q_delta_std']:.4f}")
        print(f"  Lambda effective: {actor_info['lam_eff']:.4f}")
        print(f"  Lambda Q effective: {actor_info['lambda_q_eff']:.4f}")
        print(f"  Q-term: {actor_info['q_term']:.4f}")
        print(f"  Entropy term: {actor_info['entropy_term']:.4f}")
    else:
        print(f"\nNote: Closed-loop metrics not found in info. This might be because the policy loss function is not being called.")
    
    # Demonstrate action sampling
    print("\nDemonstrating action sampling...")
    test_obs = jax.random.normal(rng, (10, obs_dim))
    actions = agent.sample_actions(test_obs, seed=rng)
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    print("\nClosedLoopSACAgent is ready to use!")
    print("\nTo use this in training:")
    print("1. Use --agent closed_loop_sac in your training script")
    print("2. Use --config experiments/configs/train_config.py:antmaze_closed_loop_sac")
    print("3. Adjust align_constraint, lam_align, and lambda_schedule as needed")
    print("\nThis agent can be used with WSRL by:")
    print("1. First collecting offline data for warm-start")
    print("2. Then using this agent for online fine-tuning")


if __name__ == "__main__":
    main()
