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
import optax
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
        rng, next_action_sample_key, q_now_rng, q_next_rng = jax.random.split(rng, 4)
        
        # Get current Q values
        qs_now = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=q_now_rng,
        )
        q_now = qs_now.min(axis=0)
        
        # Get next Q values with target network
        # Sample next actions with current actor params to enable gradients to the actor
        sample_n_actions = (
            self.config["n_actions"] if self.config["max_target_backup"] else None
        )
        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["next_observations"],
            next_action_sample_key,
            grad_params=grad_params,
            repeat=sample_n_actions,
        )
        qs_next = self.forward_target_critic(
        # qs_next = self.forward_critic(
            batch["next_observations"],
            next_actions,
            rng=q_next_rng,
        )
        q_next_min = qs_next.min(axis=0)
        # Ensure target next Qs shape matches policy sampling setup
        # When multiple actions are sampled (n_actions), reduce along that axis
        chex.assert_equal_shape([q_next_min, next_actions_log_probs])
        q_next_min = self._process_target_next_qs(
            q_next_min, next_actions_log_probs
        )
        # Do not propagate gradients through q_next_min in align_loss
        q_next_min = jax.lax.stop_gradient(q_next_min)
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

        rng, policy_rng, sample_rng, critic_pred_rng, critic_new_rng, align_rng = jax.random.split(rng, 6)
        
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
            rng=critic_pred_rng,
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
        # Choose between two policy loss variants: 'align' (default) or 'q_trust'
        loss_variant = self.config.get("policy_loss_variant", "align")
        loss_variant_code = jnp.asarray(0 if loss_variant == "align" else 1, dtype=jnp.int32)
        # Will compute TD error (q_delta) only when needed:
        # - Always for 'align' variant
        # - For 'q_trust' variant only before align_steps
        # Get current Q values for the new policy
        qs_new = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_new_rng,
        )
        q_new = qs_new.min(axis=0)
        # TD-based trust for Q term (variant: 'q_trust'): larger |q_delta| => smaller trust
        q_term_unweighted = (-q_new).mean()
        # Apply q_trust only before align_steps (optionally offset by lam_eff_linear_start_step)
        steps = self.config.get("align_steps", 100000)
        start = self.config.get("lam_eff_linear_start_step", 0)
        eff_step = jnp.maximum(self.state.step - start, 0)
        use_q_trust_flag = jnp.asarray(1 if loss_variant == "q_trust" else 0, dtype=jnp.int32)
        is_align_flag = jnp.asarray(1 if loss_variant == "align" else 0, dtype=jnp.int32)

        def _compute_td_fn(_):
            return self._compute_align_loss(batch, align_rng, params)

        def _skip_td_fn(_):
            return jnp.asarray(0.0, dtype=jnp.float32), jnp.zeros_like(q_new)

        need_td = jnp.logical_or(is_align_flag == 1, jnp.logical_and(use_q_trust_flag == 1, eff_step < steps))
        align_loss, q_delta = jax.lax.cond(need_td, _compute_td_fn, _skip_td_fn, operand=None)

        def _q_term_with_trust(_):
            q_trust_beta = jnp.asarray(self.config.get("q_trust_beta", 1.0), dtype=jnp.float32)
            weight = 1.0 / (1.0 + q_trust_beta * (q_delta ** 2))
            weight = jax.lax.stop_gradient(weight)
            return (-(weight * q_new)).mean(), weight

        def _q_term_without_trust(_):
            return q_term_unweighted, jnp.ones_like(q_new)

        cond = jnp.logical_and(use_q_trust_flag == 1, eff_step < steps)
        q_term, q_trust_weight = jax.lax.cond(cond, _q_term_with_trust, _q_term_without_trust, operand=None)
        entropy_term = (temperature * log_probs).mean()

        # Compute lam_align with schedule: fixed | linear | adaptive (only used in 'align' variant)
        if loss_variant == "align":
            lam_align_init = jnp.asarray(self.config.get("lam_align", 1.0), dtype=jnp.float32)
            schedule = self.config.get("lambda_schedule", "fixed")
            # Precompute debug helpers for logging
            steps = self.config.get("align_steps", 100000)
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
        else:
            lam_align = jnp.asarray(0.0, dtype=jnp.float32)
            schedule_code = jnp.asarray(3, dtype=jnp.int32)  # disabled in q_trust variant

        # Final objective per variant
        if loss_variant == "align":
            policy_loss =  q_term + lam_align * align_loss + entropy_term
        else:
            policy_loss =  q_term + entropy_term
        

        info = {
            "actor_loss": policy_loss,
            "actor_nll": nll_objective,
            "temperature": temperature,
            "target_entropy": self.config["target_entropy"],
            "entropy": -log_probs.mean(),
            "log_probs": log_probs,
            "actions_mse": ((actions - batch["actions"]) ** 2).sum(axis=-1).mean(),
            "batch_actions_var": jnp.var(batch["actions"], axis=0).mean(),
            "pi_actions_var": jnp.var(actions, axis=0).mean(),
            "batch_actions_mean": jnp.mean(batch["actions"], axis=0).mean(),
            "pi_actions_mean": jnp.mean(actions, axis=0).mean(),
            "batch_actions_sq_mean": jnp.mean(jnp.square(batch["actions"]), axis=0).mean(),
            "pi_actions_sq_mean": jnp.mean(jnp.square(actions), axis=0).mean(),
            "batch_actions_max": jnp.max(batch["actions"]),
            "batch_actions_min": jnp.min(batch["actions"]),
            "pi_actions_max": jnp.max(actions),
            "pi_actions_min": jnp.min(actions),
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
            "lambda_schedule_code": schedule_code,
            "policy_loss_variant_code": loss_variant_code,
        }
        if loss_variant == "q_trust":
            info.update({
                "q_term_unweighted": q_term_unweighted,
                "q_trust_weight_mean": jnp.mean(q_trust_weight),
                "q_trust_weight_min": jnp.min(q_trust_weight),
                "q_trust_weight_max": jnp.max(q_trust_weight),
            })

        # Optional: per-term gradient norms for actor (and log_std head)
        if self.config.get("log_actor_grad_terms", False):
            log_std_layer_name = self.config.get("actor_log_std_layer_name", "Dense_1")

            def q_term_only_fn(p):
                # Recompute actions and q_new with same RNGs
                dist = self.forward_policy(batch["observations"], rng=policy_rng, grad_params=p)
                a, _ = dist.sample_and_log_prob(seed=sample_rng)
                qs_new_local = self.forward_critic(batch["observations"], a, rng=critic_new_rng)
                q_new_local = qs_new_local.min(axis=0)

                if (self.config.get("policy_loss_variant", "align") == "q_trust") and (eff_step < steps):
                    # Trust-weighted q-term
                    # Reuse q_delta computation with same RNGs
                    sample_n_actions = (
                        self.config["n_actions"] if self.config["max_target_backup"] else None
                    )
                    next_actions, next_actions_log_probs = self.forward_policy_and_sample(
                        batch["next_observations"], align_rng, grad_params=p, repeat=sample_n_actions,
                    )
                    qs_next_local = self.forward_target_critic(
                        batch["next_observations"], next_actions, rng=critic_new_rng,
                    )
                    q_next_min_local = qs_next_local.min(axis=0)
                    chex.assert_equal_shape([q_next_min_local, next_actions_log_probs])
                    q_next_min_local = self._process_target_next_qs(q_next_min_local, next_actions_log_probs)
                    q_now_local = self.forward_critic(batch["observations"], batch["actions"], rng=critic_new_rng).min(axis=0)
                    q_delta_local = batch["rewards"] + self.config["discount"] * batch["masks"] * q_next_min_local - q_now_local
                    q_trust_beta = jnp.asarray(self.config.get("q_trust_beta", 1.0), dtype=jnp.float32)
                    weight_local = 1.0 / (1.0 + q_trust_beta * (q_delta_local ** 2))
                    weight_local = jax.lax.stop_gradient(weight_local)
                    return (-(weight_local * q_new_local)).mean()
                else:
                    # Unweighted q-term
                    return (-(q_new_local)).mean()

            def entropy_term_only_fn(p):
                dist = self.forward_policy(batch["observations"], rng=policy_rng, grad_params=p)
                _, lp = dist.sample_and_log_prob(seed=sample_rng)
                return (temperature * lp).mean()

            def align_term_only_fn(p):
                # Only relevant if loss_variant == 'align'
                if self.config.get("policy_loss_variant", "align") != "align":
                    return jnp.asarray(0.0, dtype=jnp.float32)
                align_loss_local, _ = self._compute_align_loss(batch, align_rng, p)
                # Use current lam_align as constant weight
                lam_local = jnp.maximum(0.0, (
                    self.forward_align_lagrange(grad_params=p) if self.config.get("lambda_schedule", "fixed") == "adaptive" else lam_align
                ))
                lam_local = jax.lax.stop_gradient(lam_local)
                return (lam_local * align_loss_local)

            g_q_full = jax.grad(q_term_only_fn)(params)
            g_ent_full = jax.grad(entropy_term_only_fn)(params)
            g_align_full = jax.grad(align_term_only_fn)(params)

            g_q_actor = g_q_full.get("actor", {})
            g_ent_actor = g_ent_full.get("actor", {})
            g_align_actor = g_align_full.get("actor", {})

            info["actor_grad_norm_q"] = optax.global_norm(g_q_actor)
            info["actor_grad_norm_entropy"] = optax.global_norm(g_ent_actor)
            info["actor_grad_norm_align"] = optax.global_norm(g_align_actor)

            try:
                g_q_logstd = g_q_actor.get(log_std_layer_name, None)
                g_ent_logstd = g_ent_actor.get(log_std_layer_name, None)
                g_align_logstd = g_align_actor.get(log_std_layer_name, None)
                info["actor_grad_norm_q_log_std"] = (
                    optax.global_norm(g_q_logstd) if g_q_logstd is not None else jnp.array(0.0, jnp.float32)
                )
                info["actor_grad_norm_entropy_log_std"] = (
                    optax.global_norm(g_ent_logstd) if g_ent_logstd is not None else jnp.array(0.0, jnp.float32)
                )
                info["actor_grad_norm_align_log_std"] = (
                    optax.global_norm(g_align_logstd) if g_align_logstd is not None else jnp.array(0.0, jnp.float32)
                )
            except Exception:
                info["actor_grad_norm_q_log_std"] = jnp.array(0.0, jnp.float32)
                info["actor_grad_norm_entropy_log_std"] = jnp.array(0.0, jnp.float32)
                info["actor_grad_norm_align_log_std"] = jnp.array(0.0, jnp.float32)

            # Also log ||dQ/da|| as diagnostic
            def q_min_mean_fn(a):
                qs_local = self.forward_critic(batch["observations"], a, rng=critic_pred_rng)
                return qs_local.min(axis=0).mean()

            dq_da = jax.grad(q_min_mean_fn)(actions)
            info["dq_da_l2_mean"] = jnp.linalg.norm(dq_da, axis=-1).mean()

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

        # KKT diagnostics
        lambda_value = self.forward_align_lagrange(grad_params=params)
        complementarity = lambda_value * positive_violation

        info = {
            "align_lagrange_loss": align_penalty,
            "align_constraint_lhs": align_loss,
            "align_constraint_rhs": align_constraint,
            "align_violation": constraint_violation,
            "align_positive_violation": positive_violation,
            "align_lambda": lambda_value,
            "align_lambda_times_violation": complementarity,
            "align_kkt_residual": jnp.abs(complementarity),
        }

        return align_penalty, info

    def loss_fns(self, batch):
        """Override to include alignment loss and its multiplier."""
        loss_map = {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }
        # Only expose align_lagrange loss when module exists (adaptive schedule)
        if self.config.get("lambda_schedule", "fixed") == "adaptive":
            loss_map["align_lagrange"] = partial(self.align_lagrange_loss_fn, batch)
        return loss_map

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

        rng, _ = jax.random.split(self.state.rng)

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified steps
        # Filter out steps that are not available (e.g., align_lagrange when module is absent)
        available_steps = frozenset(loss_fns.keys())
        requested_steps = networks_to_update
        filtered_steps = requested_steps & available_steps
        for key in loss_fns.keys() - filtered_steps:
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
        critic_infos.pop("actor", None)
        critic_infos.pop("temperature", None)
        critic_infos.pop("align_lagrange", None)

        # One step for actor, temperature, and optionally align_lagrange (if present)
        requested = {"actor", "temperature", "align_lagrange"}
        available = set(agent.loss_fns(batch).keys())
        to_update = frozenset(requested & available)
        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=to_update,
        )
        actor_temp_infos.pop("critic", None)

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
        align_steps: int = 100000,
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

        # Only create align_lagrange module if lambda_schedule is adaptive
        need_align_module = kwargs.get("lambda_schedule", "fixed") == "adaptive"
        if need_align_module:
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
        }
        if need_align_module:
            networks["align_lagrange"] = align_lagrange_def
        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**kwargs.get("actor_optimizer_kwargs", {"learning_rate": 3e-4})),
            "critic": make_optimizer(**kwargs.get("critic_optimizer_kwargs", {"learning_rate": 3e-4})),
            "temperature": make_optimizer(**kwargs.get("temperature_optimizer_kwargs", {"learning_rate": 3e-4})),
        }
        if need_align_module:
            txs["align_lagrange"] = make_optimizer(**align_lagrange_optimizer_kwargs)

        # Initialize parameters
        rng, init_rng = jax.random.split(rng)
        init_kwargs = {
            "actor": [observations],
            "critic": [observations, actions],
            "temperature": [],
        }
        if need_align_module:
            init_kwargs["align_lagrange"] = []
        params = model_def.init(init_rng, **init_kwargs)["params"]

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
            "align_steps": align_steps,
            # TD-based trust weighting for policy Q-term (optional)
            "q_trust_enabled": kwargs.get("q_trust_enabled", False),
            "q_trust_beta": kwargs.get("q_trust_beta", 1.0),
            # Policy loss variant: 'align' | 'q_trust'
            "policy_loss_variant": kwargs.get("policy_loss_variant", "align"),
            # Current build includes align_lagrange module only if lambda_schedule=='adaptive'
            "lambda_schedule": kwargs.get("lambda_schedule", "fixed"),
        }

        # Prevent external kwargs from overriding computed target_entropy
        safe_kwargs = dict(kwargs)
        safe_kwargs.pop("target_entropy", None)

        new_agent = cls(
            state=state,
            config={**base_config, **safe_kwargs, **closed_loop_config},
        )

        return new_agent
