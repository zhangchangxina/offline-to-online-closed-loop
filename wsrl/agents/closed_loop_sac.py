"""
Implementation of SAC with closed-loop update mechanism.
This extends SAC with alignment loss and Lagrangian constraints for better training stability.
"""
import copy
from functools import partial
from typing import Optional, Tuple, Union

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from absl import flags

from wsrl.agents.sac import SACAgent
from wsrl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import *
from wsrl.networks.actor_critic_nets import Critic, Policy, ensemblize
from wsrl.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from wsrl.networks.mlp import MLP


FLAGS = flags.FLAGS


class ClosedLoopSACAgent(SACAgent):
    """
    SAC agent with closed-loop update mechanism.
    Implements alignment loss and Lagrangian constraints for better training stability.
    This can be used with any warm-start strategy including WSRL.
    """
    # Optional offline critic params loaded from a checkpoint (not part of JAX PyTree)
    offline_critic_params: Optional[Params] = nonpytree_field(default=None)

    @staticmethod
    def _normalize_lambda_schedule(schedule: str) -> str:
        schedule_lower = str(schedule).strip().lower()
        if schedule_lower in ("lagrangian", "lagrange", "lag"):
            return "lagrangian"
        if schedule_lower in ("augmented_lagrangian", "aug_lagrangian", "aug-lagrangian", "augmented"):
            return "aug_lagrangian"
        allowed = {
            "fixed",
            "linear",
            "exp",
            "exp_decay",
            "fast_slow",
            "external",
        }
        if schedule_lower in allowed:
            return schedule_lower
        raise ValueError("Unsupported lambda_schedule value")

    def set_offline_critic_params(self, critic_params: Params):
        return self.replace(offline_critic_params=critic_params)
    
    def forward_align_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the alignment Lagrange multiplier
        """
        value = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="align_lagrange",
        )
        # Project to feasible set: non-negative and optionally clipped to lambda_clip
        value = jnp.maximum(value, jnp.asarray(0.0, dtype=jnp.float32))
        lam_max = float(self.config.get("lambda_clip", 10.0))
        if lam_max > 0.0:
            value = jnp.minimum(value, jnp.asarray(lam_max, dtype=jnp.float32))
        return value

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Enhanced policy loss with closed-loop update mechanism.
        Implements the Lagrangian objective for constraint E[q_delta^2] <= c
        """
        batch_size = batch["rewards"].shape[0]
        # When alignment is effectively disabled, delegate to base SAC for exact parity
        schedule = self._normalize_lambda_schedule(self.config.get("lambda_schedule", "fixed"))
        lam_align_init_val = float(self.config.get("lam_align", 0.0))
        lam_external_val = float(self.config.get("align_lambda_external", self.config.get("lambda_external", 0.0)))
        if (schedule not in ("lagrangian", "aug_lagrangian")) and (lam_align_init_val <= 0.0) and (lam_external_val <= 0.0):
            return super().policy_loss_fn(batch, params, rng)
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

        # Individual terms
        q_term = -jnp.mean(actor_objective)
        entropy_term = jnp.mean(temperature * log_probs)
        standard_actor_loss = q_term + entropy_term


        # Q-drop align loss using offline-saved critic as reference on dataset actions
        # q_pi: current critic on current policy actions
        q_pi = predicted_q
        # Reference actions: dataset actions
        ref_actions = jnp.clip(batch["actions"], -0.99, 0.99)
        # Prefer explicitly provided offline critic params; fallback to target params
        offline_params = getattr(self, "offline_critic_params", None)
        # if offline_params is None:
        #     offline_params = self.state.target_params
        qs_ref = self.forward_critic(
            batch["observations"],
            ref_actions,
            rng=critic_new_rng,
            grad_params=offline_params,
        )
        q_ref = qs_ref.min(axis=0)
        q_drop = q_ref - q_pi
        # Penalize only positive drops
        align_loss = jnp.mean(jnp.maximum(q_drop, 0.0))


        # Lambda schedule (fixed | linear | exp | exp_decay | fast_slow | lagrangian | aug_lagrangian | external)
        lambda_schedule_raw = self.config.get("lambda_schedule", "fixed")
        lambda_schedule = self._normalize_lambda_schedule(lambda_schedule_raw)

        lam_align_init = jnp.asarray(self.config.get("lam_align", 1.0), dtype=jnp.float32)
        align_steps = int(self.config.get("align_steps", 0))
        steps_val = jnp.asarray(align_steps, dtype=jnp.int32)
        offline_end = jnp.asarray(int(getattr(FLAGS, "num_offline_steps", 0)), dtype=jnp.int32)
        start_offset = jnp.asarray(int(self.config.get("lam_eff_linear_start_step", 0)), dtype=jnp.int32)
        start_val = offline_end + start_offset
        inner_step = self.state.step.astype(jnp.int32)
        utd_val = jnp.asarray(int(getattr(FLAGS, "utd", 1)), dtype=jnp.int32)
        utd_effective = jnp.where(utd_val > 1, utd_val + 1, jnp.asarray(1, dtype=jnp.int32))
        outer_step = jnp.minimum(inner_step, offline_end) + jnp.maximum(inner_step - offline_end, 0) // utd_effective

        if lambda_schedule in ("lagrangian", "aug_lagrangian"):
            lam_align = jnp.maximum(0.0, jnp.asarray(self.forward_align_lagrange(), dtype=jnp.float32))
        elif lambda_schedule == "external":
            lam_align = jnp.asarray(
                float(
                    self.config.get(
                        "align_lambda_external",
                        self.config.get("lambda_external", self.config.get("lam_align", 1.0)),
                    )
                ),
                dtype=jnp.float32,
            )
        elif lambda_schedule == "linear":
            lam_init = jnp.maximum(0.0, lam_align_init)
            eff_step = jnp.maximum(outer_step - start_val, 0)
            steps_safe = jnp.maximum(steps_val, jnp.asarray(1, dtype=jnp.int32))
            progress = jnp.clip(
                eff_step.astype(jnp.float32) / steps_safe.astype(jnp.float32),
                0.0,
                1.0,
            )
            lam_align = lam_init * (1.0 - progress)
        elif lambda_schedule in ("exp", "exp_decay", "fast_slow"):
            lam_init = jnp.maximum(0.0, lam_align_init)
            eff_step = jnp.maximum(outer_step - start_val, 0)
            steps_safe = jnp.maximum(steps_val, jnp.asarray(1, dtype=jnp.int32))
            progress = jnp.clip(
                eff_step.astype(jnp.float32) / steps_safe.astype(jnp.float32),
                0.0,
                1.0,
            )
            decay_rate = jnp.asarray(float(self.config.get("lambda_exp_rate", 5.0)), dtype=jnp.float32)
            lam_align = lam_init * jnp.exp(-decay_rate * progress)
        else:  # fixed
            lam_align = jnp.maximum(0.0, lam_align_init)

        lam_max = float(self.config.get("lambda_clip", -1.0))
        if lam_max > 0.0:
            lam_align = jnp.minimum(lam_align, jnp.asarray(lam_max, dtype=jnp.float32))

        has_duration = steps_val > 0
        align_enabled_bool = jnp.where(
            has_duration,
            (outer_step >= start_val) & (outer_step < (start_val + steps_val)),
            outer_step >= start_val,
        )
        align_mask = align_enabled_bool.astype(jnp.float32)

        align_aug_penalty = jnp.asarray(0.0, dtype=jnp.float32)
        if lambda_schedule == "aug_lagrangian":
            aug_coeff = jnp.asarray(self.config.get("align_aug_coeff", 1.0), dtype=jnp.float32)
            align_aug_penalty = 0.5 * aug_coeff * align_loss

        policy_loss = standard_actor_loss + lam_align * align_loss * align_mask
        if lambda_schedule == "aug_lagrangian":
            policy_loss = policy_loss + align_aug_penalty * align_mask

        schedule_code = jnp.asarray(
            {
                "fixed": 0,
                "lagrangian": 1,
                "aug_lagrangian": 2,
                "external": 3,
                "linear": 4,
                "exp": 5,
                "exp_decay": 6,
                "fast_slow": 7,
            }.get(lambda_schedule, -1),
            dtype=jnp.int32,
        )
        

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
            "q_drop_mean": jnp.mean(q_drop),
            "q_drop_std": jnp.std(q_drop),
            "q_term": q_term,
            "entropy_term": entropy_term,
            "lam_align": lam_align,
            "lambda_schedule_code": schedule_code,
            "align_mask": align_mask,
            "align_aug_penalty": align_aug_penalty * align_mask,
        }


        return policy_loss, info


    def align_lagrange_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Lagrangian loss for alignment multiplier (lambda_align).

        Constraint: E[align_loss] <= align_constraint, where align_loss uses q-drop.
        """
        # Only active when schedule is lagrangian
        lambda_schedule = self._normalize_lambda_schedule(self.config.get("lambda_schedule", "fixed"))
        if lambda_schedule not in ("lagrangian", "aug_lagrangian"):
            return 0.0, {"align_lagrange_loss": 0.0}

        batch_size = batch["rewards"].shape[0]
        rng, policy_rng, sample_rng, q_pi_rng, q_ref_rng = jax.random.split(rng, 5)

        # Current policy actions and Q
        pi_dist = self.forward_policy(batch["observations"], rng=policy_rng, grad_params=params)
        actions_pi, _ = pi_dist.sample_and_log_prob(seed=sample_rng)
        qs_pi = self.forward_critic(batch["observations"], actions_pi, rng=q_pi_rng)
        q_pi = qs_pi.min(axis=0)
        chex.assert_shape(q_pi, (batch_size,))

        # Reference Q under offline critic on dataset actions
        ref_actions = jnp.clip(batch["actions"], -0.99, 0.99)
        offline_params = getattr(self, "offline_critic_params", None)
        if offline_params is None:
            offline_params = self.state.target_params
        qs_ref = self.forward_critic(batch["observations"], ref_actions, rng=q_ref_rng, grad_params=offline_params)
        q_ref = qs_ref.min(axis=0)
        chex.assert_shape(q_ref, (batch_size,))

        q_drop = q_ref - q_pi
        align_loss = jnp.mean(jnp.maximum(q_drop, 0.0))

        # Constraint and penalty
        align_constraint = jnp.asarray(self.config.get("align_constraint", 0.1), dtype=jnp.float32)
        constraint_violation = jax.lax.stop_gradient(align_loss - align_constraint)
        align_penalty = self.state.apply_fn(
            {"params": params},
            lhs=constraint_violation,
            rhs=jnp.zeros_like(constraint_violation),
            name="align_lagrange",
        )

        lambda_value = self.forward_align_lagrange(grad_params=params)
        info = {
            "align_lagrange_loss": align_penalty,
            "align_constraint_lhs": align_loss,
            "align_constraint_rhs": align_constraint,
            "align_lambda": lambda_value,
            "align_violation": constraint_violation,
        }
        return align_penalty, info


    def loss_fns(self, batch):
        """Include align_lagrange loss when using Lagrangian schedule."""
        loss_map = {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }
        schedule = self._normalize_lambda_schedule(self.config.get("lambda_schedule", "fixed"))
        if schedule in ("lagrangian", "aug_lagrangian"):
            loss_map["align_lagrange"] = partial(self.align_lagrange_loss_fn, batch)
        return loss_map


    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset({
            "actor", "critic", "temperature", "align_lagrange"
        }),
    ) -> Tuple["ClosedLoopSACAgent", dict]:
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, _ = jax.random.split(self.state.rng)
        loss_fns = self.loss_fns(batch)

        available_steps = frozenset(loss_fns.keys())
        requested_steps = networks_to_update
        filtered_steps = requested_steps & available_steps
        for key in loss_fns.keys() - filtered_steps:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        new_state = new_state.replace(rng=rng)
        for name, opt_state in new_state.opt_states.items():
            if hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams.keys():
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
        batch_size = batch["rewards"].shape[0]
        assert batch_size % utd_ratio == 0, (
            f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        )
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
        # Lagrangian align options
        lagrangian_align: bool = False,
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

        raw_lambda_schedule = kwargs.get("lambda_schedule", "fixed")
        normalized_lambda_schedule = cls._normalize_lambda_schedule(raw_lambda_schedule)

        # Only create align_lagrange module if lambda_schedule is lagrangian-derived
        need_align_module = normalized_lambda_schedule in ("lagrangian", "aug_lagrangian")
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

        # Closed-loop specific configuration (minimal)
        closed_loop_config = {
            "lam_align": lam_align,
            "align_steps": align_steps,
            # Store normalized schedule for downstream usage
            "lambda_schedule": normalized_lambda_schedule,
        }

        # Prevent external kwargs from overriding computed target_entropy
        safe_kwargs = dict(kwargs)
        safe_kwargs.pop("target_entropy", None)

        new_agent = cls(
            state=state,
            config={**base_config, **safe_kwargs, **closed_loop_config},
        )

        return new_agent
