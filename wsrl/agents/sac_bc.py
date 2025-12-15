import copy
import jax
import jax.numpy as jnp
import chex
import distrax
import flax
import flax.linen as nn

from functools import partial
from typing import Optional, Tuple, Union

from absl import flags
from wsrl.agents.sac import SACAgent
from wsrl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.networks.actor_critic_nets import Critic, Policy, ensemblize
from wsrl.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from wsrl.networks.mlp import MLP

FLAGS = flags.FLAGS


class SACBCWithTargetAgent(SACAgent):
    """
    SAC variant that adds an optional BC term on top of the standard policy loss.

    - BC target can be the dataset action (default) or the slowly updated actor_target
      stored in `state.target_params` (teacher).
    - During online finetuning, the TD error (q_delta) can be used as a per-sample
      coefficient to weight the BC term.

    Configuration keys (all optional):
      - bc_lambda_init: float >= 0, initial global BC coefficient (also used in fixed/linear)
      - bc_target: "dataset" | "actor_target" | "offline_checkpoint" | <ckpt_path>
      - bc_teacher_deterministic: bool, use teacher mode() if True else sample()
      - bc_td_weight_enabled: bool, enable TD-based per-sample weighting
      - bc_td_weight_abs: bool, use |q_delta| if True else q_delta (default: True)
      - bc_td_weight_power: float, apply power to weight (default: 1.0)
      - bc_td_weight_scale: float, additional scale (default: 1.0)
      - bc_td_weight_clip: Optional[float], clip weights to [0, clip]
      - bc_combine_mode: "sum" | "interpolate" (default: "sum")
    """

    # Optional offline teacher params loaded from a checkpoint (not part of JAX PyTree)
    offline_teacher_params: Optional[Params] = nonpytree_field(default=None)
    # Optional offline critic params loaded from a checkpoint (not part of JAX PyTree)
    offline_critic_params: Optional[Params] = nonpytree_field(default=None)

    def set_offline_teacher_params(self, teacher_params: Params):
        return self.replace(offline_teacher_params=teacher_params)
    
    def set_offline_critic_params(self, critic_params: Params):
        return self.replace(offline_critic_params=critic_params)

    @staticmethod
    def _normalize_bc_lambda_schedule(schedule: str) -> str:
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
        raise ValueError("Unsupported bc_lambda_schedule value")

    # ------------------------
    # Lagrange for BC strength
    # ------------------------
    def forward_bc_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the BC Lagrange multiplier. Returns the current λ_bc.
        """
        value = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="bc_lagrange",
        )
        # Project to feasible set so that if λ hits max at one step,
        # the next step starts from the clipped value.
        value = jnp.maximum(value, jnp.asarray(0.0, dtype=jnp.float32))
        lam_max = float(self.config.get("bc_lambda_max", 2.0))
        if lam_max > 0.0:
            value = jnp.minimum(value, jnp.asarray(lam_max, dtype=jnp.float32))
        return value


    def _compute_td_q_delta(
        self,
        batch: Batch,
        rng: PRNGKey,
        *,
        policy_grad_params: Optional[Params] = None,
    ) -> jax.Array:
        """
        Compute per-sample temporal-difference error:
            q_delta = r + γ Q_target(s', a') - Q(s, a)

        The next actions are sampled from the policy parameters provided via
        `policy_grad_params` (defaults to the agent parameters).
        """
        batch_size = batch["rewards"].shape[0]
        rng, sample_rng = jax.random.split(rng)

        sample_n_actions = (
            self.config["n_actions"] if self.config["max_target_backup"] else None
        )
        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["next_observations"],
            sample_rng,
            grad_params=policy_grad_params,
            repeat=sample_n_actions,
        )
        if sample_n_actions:
            chex.assert_shape(
                next_actions_log_probs, (batch_size, sample_n_actions)
            )
        else:
            chex.assert_shape(next_actions_log_probs, (batch_size,))

        rng, target_rng = jax.random.split(rng)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=target_rng,
        )

        critic_subsample = self.config["critic_subsample_size"]
        if critic_subsample is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_indices = jax.random.randint(
                subsample_key,
                (critic_subsample,),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_indices]

        target_next_min_q = target_next_qs.min(axis=0)
        target_next_min_q = self._process_target_next_qs(
            target_next_min_q,
            next_actions_log_probs,
        )
        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        rng, critic_rng = jax.random.split(rng)
        predicted_qs = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=critic_rng,
        )
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        current_q = predicted_qs.mean(axis=0)
        chex.assert_shape(current_q, (batch_size,))

        q_delta = target_q - current_q
        return jax.lax.stop_gradient(q_delta)


    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng, target_policy_rng, td_rng = jax.random.split(rng, 6)
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
        if self.config.get("debug_checks", False):
            chex.assert_shape(predicted_q, (batch_size,))
            chex.assert_shape(log_probs, (batch_size,))

        nll_objective = -jnp.mean(
            action_distributions.log_prob(jnp.clip(batch["actions"], -0.99, 0.99))
        )
        actor_objective = predicted_q
        # Individual terms for clearer logging/instrumentation
        q_term = -jnp.mean(actor_objective)
        entropy_term = jnp.mean(temperature * log_probs)
        actor_loss = q_term + entropy_term

        # Minimal logging by default to reduce host sync/compute
        if bool(self.config.get("light_logging", True)):
            info = {
                "actor_loss": actor_loss,
                "actor_nll": nll_objective,
                "temperature": temperature,
                "target_entropy": self.config["target_entropy"],
                "entropy": -log_probs.mean(),
                "q_term": q_term,
                "entropy_term": entropy_term,
            }
        else:
            info = {
                "actor_loss": actor_loss,
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
                "q_term": q_term,
                "entropy_term": entropy_term,
            }

        # Optional BC regularization (supports dataset or actor_target as teacher, and TD-weighted coefficient)
        # Online-only gating in a JAX-traceable way
        bc_lambda_schedule_raw = self.config.get("bc_lambda_schedule", "fixed")
        bc_lambda_schedule = self._normalize_bc_lambda_schedule(bc_lambda_schedule_raw)
        # Unify BC activation to outer (environment) steps
        start_val = jnp.asarray(int(FLAGS.num_offline_steps), dtype=jnp.int32)
        steps_val = jnp.asarray(int(self.config["bc_steps"]), dtype=jnp.int32)
        # Map inner optimizer steps to outer env steps:
        # - Offline phase: 1 inner step per env step
        # - Online phase: (utd>1) => (utd + 1) inner steps per env step; else 1
        offline_end = jnp.asarray(int(FLAGS.num_offline_steps), dtype=jnp.int32)
        inner_step = self.state.step.astype(jnp.int32)
        utd_val = jnp.asarray(int(FLAGS.utd), dtype=jnp.int32)
        utd_effective = jnp.where(utd_val > 1, utd_val + 1, jnp.asarray(1, dtype=jnp.int32))
        outer_step = jnp.minimum(inner_step, offline_end) + jnp.maximum(inner_step - offline_end, 0) // utd_effective
        if bc_lambda_schedule in ("lagrangian", "aug_lagrangian"):
            # Use Lagrange multiplier as dynamic coefficient (clamped to >= 0)
            bc_lambda = jnp.maximum(0.0, jnp.asarray(self.forward_bc_lagrange(), dtype=jnp.float32))
        elif bc_lambda_schedule == "external":
            # Externally provided lambda via config, updated by trainer based on J-drop
            bc_lambda = jnp.asarray(float(self.config.get("bc_lambda_external", self.config.get("bc_lambda_init", 0.0))), dtype=jnp.float32)
        elif bc_lambda_schedule == "linear":
            # Linearly decay from initial bc_loss_weight to 0 over steps (after start)
            lam_init = jnp.maximum(0.0, jnp.asarray(float(self.config["bc_lambda_init"]), dtype=jnp.float32))
            eff_step = jnp.maximum(outer_step - start_val, 0)
            steps_safe = jnp.maximum(steps_val, jnp.asarray(1, dtype=jnp.int32))
            progress = jnp.clip(eff_step.astype(jnp.float32) / steps_safe.astype(jnp.float32), 0.0, 1.0)
            bc_lambda = lam_init * (1.0 - progress)
        elif bc_lambda_schedule in ("exp", "exp_decay", "fast_slow"):
            # Exponential decay: fast early drop, slow later approach to zero
            lam_init = jnp.maximum(0.0, jnp.asarray(float(self.config["bc_lambda_init"]), dtype=jnp.float32))
            eff_step = jnp.maximum(outer_step - start_val, 0)
            steps_safe = jnp.maximum(steps_val, jnp.asarray(1, dtype=jnp.int32))
            progress = jnp.clip(eff_step.astype(jnp.float32) / steps_safe.astype(jnp.float32), 0.0, 1.0)
            decay_rate = jnp.asarray(float(self.config.get("bc_lambda_exp_rate", 5.0)), dtype=jnp.float32)
            bc_lambda = lam_init * jnp.exp(-decay_rate * progress)
        else:
            bc_lambda = jnp.asarray(float(self.config["bc_lambda_init"]), dtype=jnp.float32)

        # Online gating semantics (simplified):
        # len > 0 -> [start, start+len)
        bc_enabled_bool = (outer_step >= start_val) & (outer_step < (start_val + steps_val))
        bc_mask = bc_enabled_bool.astype(jnp.float32)

        # Resolve BC target via unified key bc_target only
        bc_target_cfg = self.config.get("bc_target", None)
        allowed_targets = ("dataset", "actor_target", "offline_checkpoint")
        if isinstance(bc_target_cfg, str):
            bc_target = bc_target_cfg if (bc_target_cfg in allowed_targets) else "offline_checkpoint"
        else:
            bc_target = "dataset"

        if bc_target != "dataset":
            # Teacher: target actor (slowly updated via target_params)
            teacher_eval_mode = bool(self.config.get("bc_teacher_eval_mode", True))
            teacher_source = (
                "offline_checkpoint" if bc_target == "offline_checkpoint" else self.config.get("bc_teacher_source", "actor_target")
            )
            if (teacher_source == "offline_checkpoint") and (self.offline_teacher_params is not None):
                teacher_params = self.offline_teacher_params
            else:
                teacher_params = self.state.target_params

            teacher_dist = self.forward_policy(
                batch["observations"],
                rng=target_policy_rng,
                grad_params=teacher_params,
                train=not teacher_eval_mode,
            )
            # Choose teacher actions
            if bool(self.config.get("bc_teacher_deterministic", True)):
                # Deterministic teacher: use mode()
                teacher_actions = teacher_dist.mode()
                # Also compare against dataset action; pick the one with higher min-ensemble Q
                rng, critic_rng_teacher = jax.random.split(rng)
                ds_action = jnp.clip(batch["actions"], -0.99, 0.99)
                qs_mode = self.forward_target_critic(
                    batch["observations"], teacher_actions, rng=critic_rng_teacher
                )  # (ensemble, batch)
                qs_ds = self.forward_target_critic(
                    batch["observations"], ds_action, rng=critic_rng_teacher
                )  # (ensemble, batch)
                qmin_mode = qs_mode.min(axis=0)
                qmin_ds = qs_ds.min(axis=0)
                use_ds = (qmin_ds > qmin_mode)  # (batch,)
                teacher_actions = jnp.where(use_ds[:, None], ds_action, teacher_actions)
            else:
                # Stochastic teacher: sample multiple candidates and pick the one with max Q(s, a)
                # Number of candidates (default 10)
                n_candidates = int(self.config.get("bc_teacher_n_candidates", 10))
                n_candidates = max(1, n_candidates)
                rng, teacher_sample_rng, critic_rng_teacher = jax.random.split(rng, 3)
                cand_actions = teacher_dist.sample(seed=teacher_sample_rng, sample_shape=(n_candidates,))
                # cand_actions: (n, batch, act_dim) -> (batch, n, act_dim)
                cand_actions = jnp.transpose(cand_actions, (1, 0, 2))
                # Also include one dataset action as an extra candidate per sample
                ds_action = jnp.clip(batch["actions"], -0.99, 0.99)  # (batch, act_dim)
                ds_action = ds_action[:, None, :]  # (batch, 1, act_dim)
                cand_actions = jnp.concatenate([cand_actions, ds_action], axis=1)  # (batch, n+1, act_dim)
                # Evaluate target critic Q(s, a) for all candidates; result: (ensemble, batch, n)
                qs_cand = self.forward_target_critic(
                    batch["observations"], cand_actions, rng=critic_rng_teacher
                )
                # Min over ensemble, then argmax over candidates
                q_min = qs_cand.min(axis=0)  # (batch, n+1)
                best_idx = jnp.argmax(q_min, axis=-1)  # (batch,)
                best_idx_exp = jnp.expand_dims(best_idx, axis=-1)
                teacher_actions = jnp.take_along_axis(cand_actions, best_idx_exp[..., None], axis=1).squeeze(axis=1)
            # Stabilize: keep teacher actions within valid bounds
            teacher_actions = jnp.clip(teacher_actions, -0.99, 0.99)
            # Per-sample negative log-likelihood under current policy
            bc_per = -action_distributions.log_prob(teacher_actions)
        else:
            # Default to dataset actions
            target_actions = jnp.clip(batch["actions"], -0.99, 0.99)
            bc_per = -action_distributions.log_prob(target_actions)


        # Per-sample BC weighting via `bc_weight_mode` (preferred path)
        # Use simplified bc_weight_mode (support aliases)
        mode_raw = str(self.config.get("bc_weight_mode", "none"))
        if mode_raw in ("pure", "pure_bc", "fixed", "uniform"):
            mode = "none"
        else:
            mode = mode_raw
        if mode != "none":
            # Choose actions to evaluate on (configurable)
            action_src = str(self.config.get("bc_uncert_action_source", "bc"))
            if action_src in ("policy", "q", "pi"):
                eval_actions = actions
            elif action_src == "dataset":
                eval_actions = jnp.clip(batch["actions"], -0.99, 0.99)
            elif action_src == "teacher":
                eval_actions = teacher_actions if (bc_target != "dataset") else jnp.clip(batch["actions"], -0.99, 0.99)
            else:
                # default: match BC target
                eval_actions = teacher_actions if (bc_target != "dataset") else jnp.clip(batch["actions"], -0.99, 0.99)
            # If uncertainty path needed, get ensemble Q(s,a)
            if (mode == "uncert") or (mode == "uncert_inverse"):
                q_src = str(self.config.get("bc_uncert_q_source", "target"))
                use_target = (q_src == "target")
                if use_target:
                    qs_eval = self.forward_target_critic(batch["observations"], eval_actions, rng=critic_rng)
                else:
                    qs_eval = self.forward_critic(batch["observations"], eval_actions, rng=critic_rng)
                measure = str(self.config.get("bc_weight_uncert_measure", "std"))
                base = jnp.var(qs_eval, axis=0) if (measure == "var") else jnp.std(qs_eval, axis=0)
            else:
                # td / td_inverse: need q_delta per sample
                q_delta = self._compute_td_q_delta(batch, td_rng, policy_grad_params=params)
                use_abs = bool(self.config.get("bc_td_weight_abs", True))
                base = jnp.abs(q_delta) if use_abs else q_delta
                info["bc_td_delta_mean"] = q_delta.mean()
                info["bc_td_delta_std"] = q_delta.std()
                if not bool(self.config.get("light_logging", True)):
                    info["bc_td_delta_min"] = q_delta.min()
                    info["bc_td_delta_max"] = q_delta.max()

            eps = float(self.config.get("bc_weight_eps", 1e-3))
            if mode.endswith("inverse"):
                weights = 1.0 / (base + eps)
            else:
                weights = base
            power = float(self.config.get("bc_weight_power", self.config.get("bc_uncert_power", 1.0)))
            if power != 1.0:
                weights = jnp.power(weights, power)
            clip_val = float(self.config.get("bc_weight_clip", -1.0))
            if clip_val > 0.0:
                weights = jnp.clip(weights, 0.0, clip_val)
            if bool(self.config.get("bc_weight_normalize", True)):
                weights = weights / (jnp.mean(weights) + 1e-8)
            scale = float(self.config.get("bc_weight_scale", 1.0))
            if scale != 1.0:
                weights = weights * scale
            weights = jax.lax.stop_gradient(weights)
            bc_loss = jnp.mean(weights * bc_per)
        # Back-compat path: uncertainty block (if configured via old flags)
        elif bool(self.config.get("bc_uncert_weight_enabled", False)):
            q_source = self.config.get("bc_uncert_q_source", "target")
            use_target = (q_source == "target")
            # Choose actions to evaluate uncertainty on (match BC target)
            eval_actions = teacher_actions if (bc_target != "dataset") else jnp.clip(batch["actions"], -0.99, 0.99)
            # Evaluate ensemble Q(s,a): shape (ensemble, batch)
            if use_target:
                qs_eval = self.forward_target_critic(batch["observations"], eval_actions, rng=critic_rng)
            else:
                qs_eval = self.forward_critic(batch["observations"], eval_actions, rng=critic_rng)
            # Uncertainty measure across ensemble
            if self.config.get("bc_uncert_measure", "std") == "std":
                uncert = jnp.std(qs_eval, axis=0)
            else:
                uncert = jnp.std(qs_eval, axis=0)
            # Map to weights
            if bool(self.config.get("bc_uncert_inverse", False)):
                eps = float(self.config.get("bc_uncert_inverse_eps", 1e-3))
                weights = 1.0 / (uncert + eps)
            else:
                weights = uncert
            power = float(self.config.get("bc_uncert_power", 1.0))
            if power != 1.0:
                weights = jnp.power(weights, power)
            clip_val = float(self.config.get("bc_uncert_clip", -1.0))
            if clip_val > 0.0:
                weights = jnp.clip(weights, 0.0, clip_val)
            if bool(self.config.get("bc_uncert_normalize", False)):
                weights = weights / (jnp.mean(weights) + 1e-8)
            scale = float(self.config.get("bc_uncert_scale", 1.0))
            if scale != 1.0:
                weights = weights * scale
            weights = jax.lax.stop_gradient(weights)
            bc_loss = jnp.mean(weights * bc_per)
        else:
            weights = jnp.ones((batch_size,), dtype=bc_per.dtype)
            bc_loss = bc_per.mean()

    
        
        # Combine with original SAC actor loss using a mask so it's traceable
        combine_mode = self.config.get("bc_combine_mode", "sum")  # "sum" | "interpolate"
        if combine_mode == "interpolate":
            mix = jnp.clip(bc_lambda * bc_mask, 0.0, 1.0)
            actor_loss = actor_loss * (1.0 - mix) + bc_loss * mix
        else:
            actor_loss = actor_loss + (bc_lambda * bc_loss * bc_mask)
        # Initialize aug_penalty for all schedules to avoid UnboundLocalError
        aug_penalty = jnp.asarray(0.0, dtype=jnp.float32)
        if bc_lambda_schedule == "aug_lagrangian":
            aug_coeff_val = self.config.get("bc_aug_penalty", None)
            if aug_coeff_val is None:
                aug_coeff_val = self.config.get("bc_aug_coeff", 1.0)
            aug_coeff = jnp.asarray(aug_coeff_val, dtype=jnp.float32)
            aug_penalty = 0.5 * aug_coeff * bc_loss
            actor_loss = actor_loss + aug_penalty * bc_mask

        # Log diagnostics (mask bc_loss so it's 0 outside the online window)
        info["bc_loss"] = bc_loss
        if not bool(self.config.get("light_logging", True)):
            info["bc_neglogp_mean"] = bc_per.mean()
            info["bc_weight_mean"] = weights.mean()
            info["bc_weight_std"] = weights.std()
            info["bc_weight_max"] = weights.max()
            info["bc_weight_min"] = weights.min()
        info["bc_target_code"] = jnp.asarray({"dataset":0, "actor_target":1, "offline_checkpoint":2}.get(bc_target, 0), dtype=jnp.int32)
        info["actor_loss"] = actor_loss
        info["bc_mask"] = bc_mask
        info["bc_lambda"] = bc_lambda
        info["bc_lambda_schedule_code"] = jnp.asarray({
            "fixed": 0,
            "lagrangian": 1,
            "aug_lagrangian": 2,
            "external": 3,
            "linear": 4,
            "exp": 5,
            "exp_decay": 6,
            "fast_slow": 7,
        }.get(bc_lambda_schedule, -1), dtype=jnp.int32)
        info["bc_aug_penalty"] = aug_penalty * bc_mask
        # Reduce logging payload to scalars for performance
        info["log_probs_mean"] = log_probs.mean()
        # Ensure no large arrays leak into logs
        info.pop("log_probs", None)
        info.pop("actions", None)

        return actor_loss, info



    # ------------------------
    # Lagrange update for BC term
    # ------------------------
    def _compute_bc_loss_only(self, batch, rng: PRNGKey, *, policy_grad_params: Optional[Params] = None):
        """
        Compute BC loss (scalar) using the same options as in policy_loss_fn, but without combining
        with the SAC terms. This is used for the Lagrange constraint update.
        """
        batch_size = batch["rewards"].shape[0]
        rng, policy_rng, critic_rng, target_policy_rng, td_rng = jax.random.split(rng, 5)

        # Current policy distribution
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=policy_grad_params
        )

        # Resolve BC target via unified key bc_target only
        bc_target_cfg = self.config.get("bc_target", None)
        allowed_targets = ("dataset", "actor_target", "offline_checkpoint")
        if isinstance(bc_target_cfg, str):
            bc_target = bc_target_cfg if (bc_target_cfg in allowed_targets) else "offline_checkpoint"
        else:
            bc_target = "dataset"

        if bc_target != "dataset":
            teacher_eval_mode = bool(self.config.get("bc_teacher_eval_mode", True))
            teacher_source = (
                "offline_checkpoint" if bc_target == "offline_checkpoint" else self.config.get("bc_teacher_source", "actor_target")
            )
            if (teacher_source == "offline_checkpoint") and (self.offline_teacher_params is not None):
                teacher_params = self.offline_teacher_params
            else:
                teacher_params = self.state.target_params

            teacher_dist = self.forward_policy(
                batch["observations"], rng=target_policy_rng, grad_params=teacher_params, train=not teacher_eval_mode
            )
            if bool(self.config.get("bc_teacher_deterministic", True)):
                teacher_actions = teacher_dist.mode()
                rng, critic_rng_teacher = jax.random.split(rng)
                ds_action = jnp.clip(batch["actions"], -0.99, 0.99)
                qs_mode = self.forward_target_critic(batch["observations"], teacher_actions, rng=critic_rng_teacher)
                qs_ds = self.forward_target_critic(batch["observations"], ds_action, rng=critic_rng_teacher)
                qmin_mode = qs_mode.min(axis=0)
                qmin_ds = qs_ds.min(axis=0)
                use_ds = (qmin_ds > qmin_mode)
                teacher_actions = jnp.where(use_ds[:, None], ds_action, teacher_actions)
            else:
                n_candidates = int(self.config.get("bc_teacher_n_candidates", 10))
                n_candidates = max(1, n_candidates)
                rng, teacher_sample_rng, critic_rng_teacher = jax.random.split(rng, 3)
                cand_actions = teacher_dist.sample(seed=teacher_sample_rng, sample_shape=(n_candidates,))
                cand_actions = jnp.transpose(cand_actions, (1, 0, 2))
                ds_action = jnp.clip(batch["actions"], -0.99, 0.99)
                ds_action = ds_action[:, None, :]
                cand_actions = jnp.concatenate([cand_actions, ds_action], axis=1)
                qs_cand = self.forward_target_critic(batch["observations"], cand_actions, rng=critic_rng_teacher)
                q_min = qs_cand.min(axis=0)
                best_idx = jnp.argmax(q_min, axis=-1)
                best_idx_exp = jnp.expand_dims(best_idx, axis=-1)
                teacher_actions = jnp.take_along_axis(cand_actions, best_idx_exp[..., None], axis=1).squeeze(axis=1)
            teacher_actions = jnp.clip(teacher_actions, -0.99, 0.99)
            bc_per = -action_distributions.log_prob(teacher_actions)
        else:
            target_actions = jnp.clip(batch["actions"], -0.99, 0.99)
            bc_per = -action_distributions.log_prob(target_actions)

        chex.assert_shape(bc_per, (batch_size,))

        # Per-sample BC weighting via unified bc_weight_mode
        mode = str(self.config.get("bc_weight_mode", "none"))
        if mode != "none":
            action_src = str(self.config.get("bc_uncert_action_source", "bc"))
            if action_src in ("policy", "q", "pi"):
                # Need actions sampled from current policy
                actions, _ = action_distributions.sample_and_log_prob(seed=policy_rng)
                eval_actions = actions
            elif action_src == "dataset":
                eval_actions = jnp.clip(batch["actions"], -0.99, 0.99)
            elif action_src == "teacher":
                eval_actions = teacher_actions if (bc_target != "dataset") else jnp.clip(batch["actions"], -0.99, 0.99)
            else:
                eval_actions = teacher_actions if (bc_target != "dataset") else jnp.clip(batch["actions"], -0.99, 0.99)

            if (mode == "uncert") or (mode == "uncert_inverse"):
                q_src = str(self.config.get("bc_uncert_q_source", "target"))
                use_target = (q_src == "target")
                if use_target:
                    qs_eval = self.forward_target_critic(batch["observations"], eval_actions, rng=critic_rng)
                else:
                    qs_eval = self.forward_critic(batch["observations"], eval_actions, rng=critic_rng)
                measure = str(self.config.get("bc_weight_uncert_measure", "std"))
                base = jnp.var(qs_eval, axis=0) if (measure == "var") else jnp.std(qs_eval, axis=0)
            else:
                q_delta = self._compute_td_q_delta(batch, td_rng, policy_grad_params=policy_grad_params)
                use_abs = bool(self.config.get("bc_td_weight_abs", True))
                base = jnp.abs(q_delta) if use_abs else q_delta

            eps = float(self.config.get("bc_weight_eps", 1e-3))
            if mode.endswith("inverse"):
                weights = 1.0 / (base + eps)
            else:
                weights = base
            power = float(self.config.get("bc_weight_power", self.config.get("bc_uncert_power", 1.0)))
            if power != 1.0:
                weights = jnp.power(weights, power)
            clip_val = float(self.config.get("bc_weight_clip", -1.0))
            if clip_val > 0.0:
                weights = jnp.clip(weights, 0.0, clip_val)
            if bool(self.config.get("bc_weight_normalize", True)):
                weights = weights / (jnp.mean(weights) + 1e-8)
            scale = float(self.config.get("bc_weight_scale", 1.0))
            if scale != 1.0:
                weights = weights * scale
            weights = jax.lax.stop_gradient(weights)
            bc_loss = jnp.mean(weights * bc_per)
        elif bool(self.config.get("bc_uncert_weight_enabled", False)):
            q_source = self.config.get("bc_uncert_q_source", "target")
            use_target = (q_source == "target")
            eval_actions = teacher_actions if (bc_target != "dataset") else jnp.clip(batch["actions"], -0.99, 0.99)
            if use_target:
                qs_eval = self.forward_target_critic(batch["observations"], eval_actions, rng=critic_rng)
            else:
                qs_eval = self.forward_critic(batch["observations"], eval_actions, rng=critic_rng)
            uncert = jnp.std(qs_eval, axis=0)
            if bool(self.config.get("bc_uncert_inverse", False)):
                eps = float(self.config.get("bc_uncert_inverse_eps", 1e-3))
                weights = 1.0 / (uncert + eps)
            else:
                weights = uncert
            power = float(self.config.get("bc_uncert_power", 1.0))
            if power != 1.0:
                weights = jnp.power(weights, power)
            clip_val = float(self.config.get("bc_uncert_clip", -1.0))
            if clip_val > 0.0:
                weights = jnp.clip(weights, 0.0, clip_val)
            if bool(self.config.get("bc_uncert_normalize", False)):
                weights = weights / (jnp.mean(weights) + 1e-8)
            scale = float(self.config.get("bc_uncert_scale", 1.0))
            if scale != 1.0:
                weights = weights * scale
            weights = jax.lax.stop_gradient(weights)
            bc_loss = jnp.mean(weights * bc_per)
        else:
            bc_loss = bc_per.mean()

        return bc_loss

    def bc_lagrange_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Loss for the BC Lagrange multiplier λ_bc, updated by positive_violation to enforce
        a chosen constraint. Two modes supported:
        - bc_constraint_mode = "bc_loss" (default): E[BC loss] <= bc_constraint
        - bc_constraint_mode = "q_drop": E[max(0, Q_ref - Q_pi)] <= bc_constraint
        We use the multiplier both as the BC coefficient and as the dual variable.
        """
        # Gate: only when using Lagrangian-driven schedules
        schedule = self._normalize_bc_lambda_schedule(
            self.config.get("bc_lambda_schedule", "fixed")
        )
        if schedule not in ("lagrangian", "aug_lagrangian"):
            return 0.0, {"bc_lagrange_loss": 0.0}

        mode = str(self.config.get("bc_constraint_mode", "bc_loss"))
        bc_constraint = jnp.asarray(self.config.get("bc_constraint", 0.1), dtype=jnp.float32)

        if mode == "j_drop":
            # Constraint on performance drop J_drop. Supports absolute or relative metric.
            # Accept either return-based or success-based baseline
            baseline = self.config.get("perf_baseline", None)
            perf_online = self.config.get("perf_online", None)
            # Optional gating: only update lambda when a fresh J-drop has been produced by the trainer
            if bool(self.config.get("bc_update_on_new_jdrop_only", True)) and not bool(self.config.get("bc_has_new_jdrop", False)):
                return 0.0, {"bc_lagrange_loss": 0.0}
            if (baseline is None) or (perf_online is None):
                # No-op until trainer provides metrics
                return 0.0, {"bc_lagrange_loss": 0.0, "debug_baseline": baseline, "debug_perf_online": perf_online}

            # Compute J_drop per metric
            jdrop_metric = str(self.config.get("bc_jdrop_metric", self.config.get("bc_drop_metric", "absolute")))
            rel_eps = jnp.asarray(
                float(self.config.get("bc_jdrop_rel_eps", self.config.get("bc_drop_rel_eps", 1e-6))),
                dtype=jnp.float32,
            )
            baseline_val = jnp.asarray(float(baseline), dtype=jnp.float32)
            perf_online_val = jnp.asarray(float(perf_online), dtype=jnp.float32)
            raw_drop = baseline_val - perf_online_val
            is_relative = jdrop_metric in ("relative")
            if is_relative:
                j_drop = raw_drop / (jnp.abs(baseline_val) + rel_eps)
            else:
                j_drop = raw_drop

            lhs_value = jax.lax.stop_gradient(j_drop)
            bc_constraint_eff = jax.lax.stop_gradient(bc_constraint)

            constraint_violation = lhs_value - bc_constraint_eff
            # Use full constraint_violation (can be negative) for proper gradient flow
            constraint_violation = jax.lax.stop_gradient(constraint_violation)

            # Standard Lagrange penalty: -lambda * constraint_violation
            # When constraint_violation < 0 (satisfied), gradient is positive -> lambda decreases
            # When constraint_violation > 0 (violated), gradient is negative -> lambda increases
            bc_penalty = self.state.apply_fn(
                {"params": params},
                lhs=constraint_violation,
                rhs=jnp.zeros_like(constraint_violation),
                name="bc_lagrange",
            )
            
            lambda_value = self.forward_bc_lagrange(grad_params=params)
            lambda_capped = lambda_value  # already projected inside forward_bc_lagrange
            complementarity = lambda_value * constraint_violation
            metric_code = jnp.asarray(1 if is_relative else 0, dtype=jnp.int32)
            info = {
                "bc_lagrange_loss": bc_penalty,
                "j_drop": lhs_value,
                "j_drop_metric_code": metric_code,
                "bc_constraint_jdrop": bc_constraint_eff,
                "bc_violation": constraint_violation,
                "bc_positive_violation": jnp.maximum(constraint_violation, 0.0),  # For logging only
                "bc_lambda": lambda_capped,
                "bc_lambda_raw": lambda_value,
                "bc_kkt_residual": jnp.abs(complementarity),
            }
            return bc_penalty, info

        if mode == "q_drop":
            # Constraint on average positive Q drop relative to a reference action
            batch_size = batch["rewards"].shape[0]
            rng, policy_rng, sample_rng, q_pi_rng, ref_rng, q_ref_rng = jax.random.split(rng, 6)

            # Current policy actions and Q
            pi_dist = self.forward_policy(batch["observations"], rng=policy_rng, grad_params=params)
            actions_pi, _ = pi_dist.sample_and_log_prob(seed=sample_rng)
            qs_pi = self.forward_critic(batch["observations"], actions_pi, rng=q_pi_rng)
            q_pi = qs_pi.min(axis=0)
            chex.assert_shape(q_pi, (batch_size,))

            # Reference actions: dataset or teacher (matching bc_target unless overridden)
            # Resolve reference mode with a single concise rule set
            allowed_targets = ("dataset", "actor_target", "offline_checkpoint")
            ref_src = self.config.get("bc_qdrop_reference", None)
            if isinstance(ref_src, str) and (ref_src in allowed_targets):
                ref_mode = ref_src
            else:
                bc_target_cfg = self.config.get("bc_target", None)
                ref_mode = (
                    bc_target_cfg
                    if isinstance(bc_target_cfg, str) and (bc_target_cfg in allowed_targets)
                    else "dataset"
                )

            if ref_mode in ("actor_target", "offline_checkpoint"):
                teacher_eval_mode = bool(self.config.get("bc_teacher_eval_mode", True))
                teacher_source = (
                    "offline_checkpoint" if ref_mode == "offline_checkpoint" else self.config.get("bc_teacher_source", "actor_target")
                )
                if (teacher_source == "offline_checkpoint") and (self.offline_teacher_params is not None):
                    teacher_params = self.offline_teacher_params
                else:
                    teacher_params = self.state.target_params
                teacher_dist = self.forward_policy(
                    batch["observations"], rng=ref_rng, grad_params=teacher_params, train=not teacher_eval_mode
                )
                if bool(self.config.get("bc_teacher_deterministic", True)):
                    ref_actions = teacher_dist.mode()
                else:
                    # Use a fresh RNG key for sampling to avoid correlation with forward rng
                    ref_rng, ref_sample_rng = jax.random.split(ref_rng)
                    ref_actions = teacher_dist.sample(seed=ref_sample_rng)
            else:
                ref_actions = jnp.clip(batch["actions"], -0.99, 0.99)

            ref_actions = jnp.clip(ref_actions, -0.99, 0.99)
            # Choose which critic to evaluate Q_ref with
            # bc_qdrop_q_source: "current" | "target" | "offline"
            q_src = str(self.config.get("bc_qdrop_q_source", "current"))
            if q_src == "target":
                qs_ref = self.forward_target_critic(batch["observations"], ref_actions, rng=q_ref_rng)
            elif q_src == "offline":
                offline_params = getattr(self, "offline_critic_params", None)
                if offline_params is None:
                    # Fallback to target critic if offline params are not provided
                    offline_params = self.state.target_params
                qs_ref = self.forward_critic(
                    batch["observations"], ref_actions, rng=q_ref_rng, grad_params=offline_params
                )
            else:
                qs_ref = self.forward_critic(batch["observations"], ref_actions, rng=q_ref_rng)
            q_ref = qs_ref.min(axis=0)
            chex.assert_shape(q_ref, (batch_size,))

            # Positive Q drop (supports absolute or relative/percentage drop)
            # bc_qdrop_metric: "absolute" | "relative" (aliases: "percent", "percentage")
            q_drop_metric = str(self.config.get("bc_qdrop_metric", self.config.get("bc_drop_metric", "absolute")))
            rel_eps = jnp.asarray(
                float(self.config.get("bc_qdrop_rel_eps", self.config.get("bc_drop_rel_eps", 1e-6))),
                dtype=jnp.float32,
            )
            q_drop_raw = q_ref - q_pi
            if q_drop_metric in ("relative"):
                # Relative drop: (Q_ref - Q_pi) / (|Q_ref| + eps)
                q_drop = q_drop_raw / (jnp.abs(q_ref) + rel_eps)
            else:
                # Absolute drop in Q units
                q_drop = q_drop_raw
            # Optionally adapt the constraint to the batch scale
            adapt_mode = str(self.config.get("bc_qdrop_dynamic_mode", "none"))
            if adapt_mode == "batch_quantile":
                q = float(self.config.get("bc_qdrop_quantile", 0.9))
                q = jnp.clip(jnp.asarray(q, dtype=jnp.float32), 0.0, 1.0)
                # compute per-batch quantile; simple linear interpolation via jnp.quantile
                bc_constraint_eff = jnp.quantile(q_drop, q)
            elif adapt_mode == "batch_normalized":
                norm_c = jnp.asarray(float(self.config.get("bc_qdrop_norm_c", 0.5)), dtype=jnp.float32)
                eps = jnp.asarray(float(self.config.get("bc_qdrop_eps", 1e-6)), dtype=jnp.float32)
                mean = jnp.mean(q_drop)
                std = jnp.std(q_drop)
                bc_constraint_eff = mean + norm_c * jnp.maximum(std, eps)
            else:
                bc_constraint_eff = bc_constraint

            lhs_value = jnp.mean(q_drop)
            # Stop gradients so the Lagrange update does not backprop into actor/critic
            lhs_value = jax.lax.stop_gradient(lhs_value)
            bc_constraint_eff = jax.lax.stop_gradient(bc_constraint_eff)

            constraint_violation = lhs_value - bc_constraint_eff
            # Use full constraint_violation (can be negative) for proper gradient flow
            constraint_violation = jax.lax.stop_gradient(constraint_violation)

            # Standard Lagrange penalty: -lambda * constraint_violation
            # When constraint_violation < 0 (satisfied), gradient is positive -> lambda decreases
            # When constraint_violation > 0 (violated), gradient is negative -> lambda increases
            bc_penalty = self.state.apply_fn(
                {"params": params},
                lhs=constraint_violation,
                rhs=jnp.zeros_like(constraint_violation),
                name="bc_lagrange",
            )
            
            lambda_value = self.forward_bc_lagrange(grad_params=params)
            # Cap for logging consistency (forward_bc_lagrange already projects to >=0)
            lambda_capped = lambda_value  # already projected; apply additional cap if desired
            complementarity = lambda_value * constraint_violation
            info = {
                "bc_qdrop_mean": lhs_value,
                "bc_qdrop_max": jnp.max(q_drop),
                "bc_qdrop_min": jnp.min(q_drop),
                "bc_lagrange_loss": bc_penalty,
                "bc_constraint_qdrop": bc_constraint_eff,
                "bc_violation": constraint_violation,
                "bc_violation_norm": jnp.abs(constraint_violation),
                "bc_positive_violation": jnp.maximum(constraint_violation, 0.0),  # For logging only
                "bc_lambda": lambda_capped,
                "bc_lambda_norm": jnp.abs(lambda_capped),
                "bc_lambda_times_violation": complementarity,
                "bc_kkt_residual": jnp.abs(complementarity),
            }
        else:
            # Default: constraint on BC loss magnitude
            bc_loss_value = self._compute_bc_loss_only(batch, rng)
            constraint_violation = bc_loss_value - bc_constraint
            # Use full constraint_violation (can be negative) for proper gradient flow
            constraint_violation = jax.lax.stop_gradient(constraint_violation)

            # Standard Lagrange penalty: -lambda * constraint_violation
            # When constraint_violation < 0 (satisfied), gradient is positive -> lambda decreases
            # When constraint_violation > 0 (violated), gradient is negative -> lambda increases
            bc_penalty = self.state.apply_fn(
                {"params": params},
                lhs=constraint_violation,
                rhs=jnp.zeros_like(constraint_violation),
                name="bc_lagrange",
            )
            
            lambda_value = self.forward_bc_lagrange(grad_params=params)
            lambda_capped = lambda_value  # already projected
            complementarity = lambda_value * constraint_violation
            info = {
                "bc_lagrange_loss": bc_penalty,
                "bc_constraint_lhs": bc_loss_value,
                "bc_constraint_rhs": bc_constraint,
                "bc_violation": constraint_violation,
                "bc_violation_norm": jnp.abs(constraint_violation),
                "bc_positive_violation": jnp.maximum(constraint_violation, 0.0),  # For logging only
                "bc_lambda": lambda_capped,
                "bc_lambda_norm": jnp.abs(lambda_capped),
                "bc_lambda_times_violation": complementarity,
                "bc_kkt_residual": jnp.abs(complementarity),
            }

        return bc_penalty, info

    def loss_fns(self, batch):
        """Override to include bc_lagrange loss when a Lagrangian schedule is enabled."""
        from functools import partial as _partial
        loss_map = {
            "critic": _partial(self.critic_loss_fn, batch),
            "actor": _partial(self.policy_loss_fn, batch),
            "temperature": _partial(self.temperature_loss_fn, batch),
        }
        schedule = self._normalize_bc_lambda_schedule(
            self.config.get("bc_lambda_schedule", "fixed")
        )
        if schedule in ("lagrangian", "aug_lagrangian"):
            loss_map["bc_lagrange"] = _partial(self.bc_lagrange_loss_fn, batch)
        return loss_map

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset(
            {"actor", "critic", "temperature", "bc_lagrange"}
        ),
    ) -> Tuple["SACBCWithTargetAgent", dict]:
        """
        Same as base update but includes 'bc_lagrange' by default when available.
        """
        batch_size = batch["rewards"].shape[0]
        if self.config.get("debug_checks", False):
            chex.assert_tree_shape_prefix(batch, (batch_size,))

        rng, _ = jax.random.split(self.state.rng)

        loss_fns = self.loss_fns(batch)

        # Filter out steps that are not available (e.g., bc_lagrange when absent)
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
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis", "bc_lambda_schedule"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
        bc_lambda_schedule: str = "fixed",
    ) -> Tuple["SACBCWithTargetAgent", dict]:
        """
        High-UTD updates with optional bc_lagrange update on the final step when a Lagrangian schedule is enabled.

        - Perform `utd_ratio` critic-only updates on minibatches.
        - Then one joint update for actor/temperature and, if Lagrangian-based, bc_lagrange.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        if self.config.get("debug_checks", False):
            chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple[SACBCWithTargetAgent], data: Tuple[Batch]):
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
        # Remove placeholders for non-updated networks during critic-only phase (if present)
        for k in ("actor", "temperature", "bc_lagrange"):
            try:
                del critic_infos[k]
            except Exception:
                pass

        # Final joint update: actor/temperature (+ bc_lagrange if Lagrangian schedule is enabled)
        normalized_schedule = self._normalize_bc_lambda_schedule(bc_lambda_schedule)
        final_networks = (
            frozenset({"actor", "temperature", "bc_lagrange"})
            if normalized_schedule in ("lagrangian", "aug_lagrangian")
            else frozenset({"actor", "temperature"})
        )

        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=final_networks,
        )
        # Remove critic entry from actor/temp infos
        try:
            del actor_temp_infos["critic"]
        except Exception:
            pass

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
        # BC Lagrange options
        bc_lambda_schedule: str = "fixed",  # "fixed" | "lagrangian" | "aug_lagrangian" | "linear" | "exp" | "exp_decay" | "fast_slow" | "external"
        bc_constraint: float = 0.1,
        bc_lagrange_optimizer_kwargs: dict = {
            "learning_rate": 3e-4,
        },
        **kwargs,
    ):
        """
        Create a SAC-BC agent with optional Lagrangian control to modulate the BC coefficient.
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

        # Optionally include bc_lagrange module when schedule requires it
        normalized_schedule = cls._normalize_bc_lambda_schedule(bc_lambda_schedule)
        need_bc_module = normalized_schedule in ("lagrangian", "aug_lagrangian")
        if need_bc_module:
            # Initialize from provided bc_lambda_init
            init_val = float(kwargs["bc_lambda_init"])  # must be provided
            init_val = max(0.0, init_val)
            bc_lagrange_def = LeqLagrangeMultiplier(
                init_value=init_val if init_val > 0.0 else 1.0,
                constraint_shape=(),
                constraint_type="leq",
                name="bc_lagrange",
            )

        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        if need_bc_module:
            networks["bc_lagrange"] = bc_lagrange_def
        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**kwargs.get("actor_optimizer_kwargs", {"learning_rate": 3e-4})),
            "critic": make_optimizer(**kwargs.get("critic_optimizer_kwargs", {"learning_rate": 3e-4})),
            "temperature": make_optimizer(**kwargs.get("temperature_optimizer_kwargs", {"learning_rate": 3e-4})),
        }
        if need_bc_module:
            # Allow unified alias bc_lagrangian_lr to override Lagrange optimizer lr
            eff_lr = kwargs.get("bc_lagrangian_lr", None)
            if eff_lr is not None:
                try:
                    eff_lr = float(eff_lr)
                    bc_lagrange_optimizer_kwargs = {**bc_lagrange_optimizer_kwargs, "learning_rate": eff_lr}
                except Exception:
                    pass
            txs["bc_lagrange"] = make_optimizer(**bc_lagrange_optimizer_kwargs)

        # Initialize parameters
        rng, init_rng = jax.random.split(rng)
        init_kwargs = {
            "actor": [observations],
            "critic": [observations, actions],
            "temperature": [],
        }
        if need_bc_module:
            init_kwargs["bc_lagrange"] = []
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

        base_config = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=kwargs.get("discount", 0.99),
            soft_target_update_rate=kwargs.get("soft_target_update_rate", 0.005),
            target_entropy=target_entropy,
            backup_entropy=kwargs.get("backup_entropy", False),
            bc_lambda_init=kwargs["bc_lambda_init"],
            n_actions=kwargs.get("n_actions", 10),
            max_target_backup=kwargs.get("max_target_backup", False),
            # BC Lagrange
            bc_lambda_schedule=normalized_schedule,
            bc_constraint=bc_constraint,
        )

        # Prevent external kwargs from overriding computed target_entropy
        safe_kwargs = dict(kwargs)
        safe_kwargs.pop("target_entropy", None)

        return cls(
            state=state,
            config={**base_config, **safe_kwargs},
        )

