"""
Implementation of SAC with closed-loop update mechanism.
This extends SAC with alignment loss and Lagrangian constraints for better training stability.
"""
import copy
from functools import partial
from typing import Optional, Tuple, Union

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from wsrl.agents.sac import SACAgent
from wsrl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import *
from wsrl.networks.actor_critic_nets import Critic, Policy, ensemblize
from wsrl.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from wsrl.networks.mlp import MLP


class ClosedLoopSACAgent(SACAgent):
    """
    SAC agent with closed-loop update mechanism.
    Implements alignment loss and Lagrangian constraints for better training stability.
    This can be used with any warm-start strategy including WSRL.
    """
    
    def forward_align_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the alignment Lagrange multiplier
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="align_lagrange",
        )

    def _compute_align_loss(self, batch, rng: PRNGKey, grad_params: Optional[Params] = None):
        """
        Compute alignment loss for closed-loop update mechanism.
        This implements the constraint E[q_delta^2] <= c where q_delta = r + γQ(s',a') - Q(s,a)
        """
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key = jax.random.split(rng)
        
        # Get current Q values
        qs_now = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=rng,
        )
        q_now = qs_now.min(axis=0)
        
        # Get next Q values with target network
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )
        qs_next = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )
        q_next_min = qs_next.min(axis=0)
        # Ensure target next Qs shape matches policy sampling setup
        # When multiple actions are sampled (n_actions), reduce along that axis
        chex.assert_equal_shape([q_next_min, next_actions_log_probs])
        q_next_min = self._process_target_next_qs(
            q_next_min, next_actions_log_probs
        )
        reward_term = batch["rewards"]
        
        # Compute Q delta (mask terminals)
        q_delta = reward_term + self.config["discount"] * batch["masks"] * q_next_min - q_now
        
        # Alignment loss: E[q_delta^2]
        align_loss = jnp.mean(q_delta ** 2)
        
        return align_loss, q_delta

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Enhanced policy loss with closed-loop update mechanism.
        Implements the Lagrangian objective for constraint E[q_delta^2] <= c
        """
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng, align_rng = jax.random.split(rng, 5)
        
        # Standard SAC policy components
        action_distributions = self.forward_policy(
            batch["observations"],
            rng=policy_rng,
            grad_params=params,
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.min(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        # Standard SAC objectives
        nll_objective = -jnp.mean(
            action_distributions.log_prob(jnp.clip(batch["actions"], -0.99, 0.99))
        )
        actor_objective = predicted_q
        standard_actor_loss = -jnp.mean(actor_objective) + jnp.mean(temperature * log_probs)

        # If closed-loop is disabled, behave like standard SAC (match offline speed)
        if not self.config.get("closed_loop_enabled", False):
            info = {
                "actor_loss": standard_actor_loss,
                "actor_nll": nll_objective,
                "temperature": temperature,
                "entropy": -log_probs.mean(),
                "log_probs": log_probs,
                "actions_mse": ((actions - batch["actions"]) ** 2).sum(axis=-1).mean(),
                "dataset_rewards": batch["rewards"],
                "mc_returns": batch.get("mc_returns", None),
                "actions": actions,
            }
            # optionally add BC regularization
            if self.config.get("bc_loss_weight", 0.0) > 0:
                bc_loss = -action_distributions.log_prob(batch["actions"]).mean()
                info["bc_loss"] = bc_loss
                info["actor_bc_loss_weight"] = self.config["bc_loss_weight"]
                actor_loss = (
                    standard_actor_loss * (1 - self.config["bc_loss_weight"]) +
                    bc_loss * self.config["bc_loss_weight"]
                )
                info["actor_loss"] = actor_loss
                return actor_loss, info
            return standard_actor_loss, info

        # Closed-loop update mechanism
        # Compute alignment loss
        align_loss, q_delta = self._compute_align_loss(batch, align_rng, params)
        
        # Get current Q values for the new policy
        qs_new = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        q_new = qs_new.min(axis=0)
        q_term = (-q_new).mean()
        entropy_term = (temperature * log_probs).mean()

        # Compute lam_align with schedule: fixed | linear | adaptive
        lam_align_init = jnp.asarray(self.config.get("lam_align", 1.0), dtype=jnp.float32)

        schedule = self.config.get("lambda_schedule", "fixed")
        # Precompute debug helpers for logging
        steps = self.config.get("lam_eff_linear_steps", 100000)
        start = self.config.get("lam_eff_linear_start_step", 0)
        eff_step = jnp.maximum(self.state.step - start, 0)
        progress = jnp.minimum(eff_step, steps) / jnp.maximum(steps, 1)

        if schedule == "linear":
            schedule_code = jnp.asarray(1, dtype=jnp.int32)
            lam_align = jnp.maximum(0.0, lam_align_init * (1.0 - progress))
        elif schedule == "adaptive":
            schedule_code = jnp.asarray(2, dtype=jnp.int32)
            lam_align = jnp.maximum(0.0, self.forward_align_lagrange())
        else:
            schedule_code = jnp.asarray(0, dtype=jnp.int32)
            lam_align = jnp.maximum(0.0, lam_align_init)

        # Final objective: q_term + lam_align * align_loss + entropy_term
        policy_loss = q_term + lam_align * align_loss + entropy_term

        info = {
            "actor_loss": policy_loss,
            "actor_nll": nll_objective,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((actions - batch["actions"]) ** 2).sum(axis=-1).mean(),
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
            "actions": actions,
            # Closed-loop specific metrics
            "align_loss": align_loss,
            "q_delta_mean": jnp.mean(q_delta),
            "q_delta_std": jnp.std(q_delta),
            "q_term": q_term,
            "entropy_term": entropy_term,
            "lam_align": lam_align,
        }

        # Optionally add BC regularization
        if self.config.get("bc_loss_weight", 0.0) > 0:
            bc_loss = -action_distributions.log_prob(batch["actions"]).mean()
            info["bc_loss"] = bc_loss
            info["actor_bc_loss_weight"] = self.config["bc_loss_weight"]
            policy_loss = (
                policy_loss * (1 - self.config["bc_loss_weight"]) +
                bc_loss * self.config["bc_loss_weight"]
            )
            info["actor_loss"] = policy_loss

        return policy_loss, info

    def align_lagrange_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Loss for the alignment Lagrange multiplier lambda_align, updated by positive_violation.

        This mirrors the temperature loss update pattern: minimize multiplier * violation.
        """
        # Gate by schedule and closed_loop_enabled
        lambda_schedule = self.config.get("lambda_schedule", "fixed")
        if (not self.config.get("closed_loop_enabled", False)) or (lambda_schedule != "adaptive"):
            return 0.0, {"align_lagrange_loss": 0.0}

        # Recompute positive_violation (scalar)
        align_loss, _ = self._compute_align_loss(batch, rng)
        align_constraint = self.config.get("align_constraint", 0.1)
        constraint_violation = align_loss - align_constraint
        positive_violation = jnp.maximum(constraint_violation, 0.0)

        # Penalty for the align Lagrange multiplier
        # Equivalent form to temperature loss: minimize lambda * positive_violation
        # For leq constraint, loss is -lambda * (lhs - rhs)
        align_penalty = self.state.apply_fn(
            {"params": params},
            lhs=positive_violation,
            rhs=jnp.zeros_like(positive_violation),
            name="align_lagrange",
        )

        return align_penalty, {"align_lagrange_loss": align_penalty}

    def loss_fns(self, batch):
        """Override to include alignment loss and its multiplier."""
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
            "align_lagrange": partial(self.align_lagrange_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature", "align_lagrange"}
        ),
    ) -> Tuple["ClosedLoopSACAgent", dict]:
        """
        Same as base update but includes 'align_lagrange' by default.
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, key = jax.random.split(self.state.rng)

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["ClosedLoopSACAgent", dict]:
        """
        High-UTD version that also updates 'align_lagrange' with the actor/temperature step.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple[ClosedLoopSACAgent], data: Tuple[Batch]):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(
                minibatch,
                pmap_axis=pmap_axis,
                networks_to_update=frozenset({"critic"}),
            )
            return (agent,), info

        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (utd_ratio, minibatch_size) + data.shape[1:])

        minibatches = jax.tree_util.tree_map(make_minibatch, batch)

        (agent,), critic_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

        critic_infos = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)
        del critic_infos["actor"]
        del critic_infos["temperature"]
        del critic_infos["align_lagrange"]

        # One step for actor, temperature, and align_lagrange
        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"actor", "temperature", "align_lagrange"}),
        )
        del actor_temp_infos["critic"]

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        # Closed-loop specific parameters
        align_constraint: float = 0.1,
        lam_align: float = 1.0,
        # Deprecated fields (kept for backward-compat in configs):
        lambda_sac: float = 1.0,
        lam_eff_linear: bool = False,
        lam_eff_linear_steps: int = 100000,
        # Adaptive align Lagrange options
        adaptive_align: bool = False,
        # align_lagrange_init removed; initialized from lam_align
        align_lagrange_optimizer_kwargs: dict = {
            "learning_rate": 3e-4,
        },
        **kwargs,
    ):
        """
        Create a new SAC agent with closed-loop update mechanism.
        This can be used with any warm-start strategy including WSRL.
        Adds an optional align Lagrange multiplier updated by positive_violation.
        """

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "critic": copy.deepcopy(encoder_def),
            }

        # Define networks
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )

        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic,
            encoder=encoders["critic"],
            network=critic_backbone,
        )(name="critic")

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        # New: align lagrange multiplier (scalar)
        # Use lam_align as init for adaptive mode if not provided
        align_lagrange_def = LeqLagrangeMultiplier(
            init_value=lam_align,
            constraint_shape=(),
            constraint_type="leq",
            name="align_lagrange",
        )

        # Build model dict
        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
            "align_lagrange": align_lagrange_def,
        }
        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**kwargs.get("actor_optimizer_kwargs", {"learning_rate": 3e-4})),
            "critic": make_optimizer(**kwargs.get("critic_optimizer_kwargs", {"learning_rate": 3e-4})),
            "temperature": make_optimizer(**kwargs.get("temperature_optimizer_kwargs", {"learning_rate": 3e-4})),
            "align_lagrange": make_optimizer(**align_lagrange_optimizer_kwargs),
        }

        # Initialize parameters
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
            align_lagrange=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        target_entropy = kwargs.get("target_entropy", None)
        if target_entropy is None or target_entropy >= 0.0:
            target_entropy = -actions.shape[-1]

        # Base config
        base_config = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=kwargs.get("discount", 0.99),
            soft_target_update_rate=kwargs.get("soft_target_update_rate", 0.005),
            target_entropy=target_entropy,
            backup_entropy=kwargs.get("backup_entropy", False),
            bc_loss_weight=kwargs.get("bc_loss_weight", 0.0),
            n_actions=kwargs.get("n_actions", 10),
            max_target_backup=kwargs.get("max_target_backup", False),
        )

        # Closed-loop specific configuration
        closed_loop_config = {
            "align_constraint": align_constraint,
            "lam_align": lam_align,
            "lam_eff_linear_steps": lam_eff_linear_steps,
        }

        new_agent = cls(
            state=state,
            config={**base_config, **kwargs, **closed_loop_config},
        )

        return new_agent
