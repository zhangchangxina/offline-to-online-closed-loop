import copy
import jax
import jax.numpy as jnp
import chex
import distrax
import flax
import flax.linen as nn

from functools import partial
from typing import Optional, Tuple, Union

from wsrl.agents.sac import SACAgent
from wsrl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.networks.actor_critic_nets import Critic, Policy, ensemblize
from wsrl.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from wsrl.networks.mlp import MLP


class SACBCWithTargetAgent(SACAgent):
    """
    SAC variant that adds an optional BC term on top of the standard policy loss.

    - BC target can be the dataset action (default) or the slowly updated actor_target
      stored in `state.target_params` (teacher).
    - During online finetuning, the TD error (q_delta) can be used as a per-sample
      coefficient to weight the BC term.

    Configuration keys (all optional):
      - bc_loss_weight: float > 0 enables BC term
      - bc_mode: "dataset" | "actor_target" (default: "dataset")
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

    def set_offline_teacher_params(self, teacher_params: Params):
        return self.replace(offline_teacher_params=teacher_params)

    # ------------------------
    # Lagrange for BC strength
    # ------------------------
    def forward_bc_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the BC Lagrange multiplier. Returns the current λ_bc.
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="bc_lagrange",
        )

    def _compute_td_q_delta(
        self,
        batch,
        rng: PRNGKey,
        *,
        policy_grad_params: Optional[Params] = None,
    ) -> jax.Array:
        """
        Compute per-sample TD error q_delta = r + γ Q_target(s', a') - Q(s, a).

        - Uses dataset actions for Q(s, a)
        - Uses current policy for a' with optional grad params
        - Uses target critic for Q_target
        """
        batch_size = batch["rewards"].shape[0]
        rng, next_action_sample_key, q_now_rng, q_next_rng = jax.random.split(rng, 4)

        # Q(s, a) with current critic on dataset actions
        qs_now = self.forward_critic(
            batch["observations"],
            batch["actions"],
            rng=q_now_rng,
        )
        q_now = qs_now.min(axis=0)

        # Sample a' from current policy on s'
        sample_n_actions = (
            self.config["n_actions"] if self.config["max_target_backup"] else None
        )
        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            batch["next_observations"],
            next_action_sample_key,
            grad_params=policy_grad_params,
            repeat=sample_n_actions,
        )

        # Q_target(s', a') with target critic
        qs_next = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=q_next_rng,
        )
        q_next_min = qs_next.min(axis=0)
        chex.assert_equal_shape([q_next_min, next_actions_log_probs])
        q_next_min = self._process_target_next_qs(
            q_next_min, next_actions_log_probs
        )
        q_next_min = jax.lax.stop_gradient(q_next_min)

        # TD error per sample
        q_delta = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * q_next_min
            - q_now
        )
        chex.assert_shape(q_delta, (batch_size,))
        return q_delta

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
        bc_lambda_schedule = self.config.get("bc_lambda_schedule", "fixed")
        if bc_lambda_schedule == "adaptive":
            # Use Lagrange multiplier as dynamic coefficient (clamped to >= 0)
            bc_weight = jnp.maximum(0.0, jnp.asarray(self.forward_bc_lagrange(), dtype=jnp.float32))
        else:
            bc_weight = jnp.asarray(float(self.config.get("bc_loss_weight", 0.0)), dtype=jnp.float32)
        step = self.state.step.astype(jnp.int32)
        online_start_val = jnp.asarray(int(self.config.get("bc_online_start_step", -1)), dtype=jnp.int32)
        online_for_val = jnp.asarray(int(self.config.get("bc_online_enable_for_steps", -1)), dtype=jnp.int32)

        # Online gating semantics (simplified):
        # len == 0 -> off; len > 0 -> [start, start+len); len == -1 -> [start, +∞)
        start_ok = (online_start_val >= 0)
        after_start = (step >= online_start_val)
        in_window = (online_for_val == jnp.asarray(-1, dtype=jnp.int32)) | (
            (online_for_val > 0) & (step < (online_start_val + online_for_val))
        )
        bc_enabled_bool = (bc_weight > 0.0) & start_ok & after_start & in_window
        bc_mask = bc_enabled_bool.astype(jnp.float32)

        # Resolve BC target (unified key bc_target preferred; fallback to bc_mode + bc_teacher_source)
        bc_target_cfg = self.config.get("bc_target", None)
        allowed_targets = ("dataset", "actor_target", "offline_checkpoint")
        if bc_target_cfg is None:
            # legacy fallback
            legacy_mode = self.config.get("bc_mode", "dataset")
            legacy_src = self.config.get("bc_teacher_source", "actor_target")
            if legacy_mode == "dataset":
                bc_target = "dataset"
            else:
                bc_target = "offline_checkpoint" if legacy_src == "offline_checkpoint" else "actor_target"
        else:
            # unified path: if not in allowed set, treat as path => offline_checkpoint
            if isinstance(bc_target_cfg, str) and (bc_target_cfg not in allowed_targets):
                bc_target = "offline_checkpoint"
            else:
                bc_target = bc_target_cfg

        bc_mode = "dataset" if bc_target == "dataset" else "actor_target"

        if bc_mode == "actor_target":
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

        chex.assert_shape(bc_per, (batch_size,))

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
                eval_actions = teacher_actions if (bc_mode == "actor_target") else jnp.clip(batch["actions"], -0.99, 0.99)
            else:
                # default: match BC target
                eval_actions = teacher_actions if (bc_mode == "actor_target") else jnp.clip(batch["actions"], -0.99, 0.99)
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
                base = jnp.abs(q_delta)

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
            eval_actions = teacher_actions if (bc_mode == "actor_target") else jnp.clip(batch["actions"], -0.99, 0.99)
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
            mix = jnp.clip(bc_weight * bc_mask, 0.0, 1.0)
            actor_loss = actor_loss * (1.0 - mix) + bc_loss * mix
        else:
            actor_loss = actor_loss + (bc_weight * bc_loss * bc_mask)

        # Log diagnostics (mask bc_loss so it's 0 outside the online window)
        info["bc_loss"] = bc_loss
        info["bc_neglogp_mean"] = bc_per.mean()
        info["bc_weight_mean"] = weights.mean()
        info["bc_weight_std"] = weights.std()
        info["bc_weight_max"] = weights.max()
        info["bc_weight_min"] = weights.min()
        info["bc_mode_code"] = jnp.asarray(1 if bc_mode == "actor_target" else 0, dtype=jnp.int32)
        info["bc_target_code"] = jnp.asarray({"dataset":0, "actor_target":1, "offline_checkpoint":2}.get(bc_target, 0), dtype=jnp.int32)
        info["actor_loss"] = actor_loss
        info["bc_enabled"] = bc_enabled_bool.astype(jnp.int32)
        info["bc_mask"] = bc_mask
        if bc_lambda_schedule == "adaptive":
            info["bc_lambda"] = bc_weight
        # Reduce logging payload to scalars for performance
        info["log_probs_mean"] = log_probs.mean()
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

        # Resolve BC target (reuse the exact logic)
        bc_target_cfg = self.config.get("bc_target", None)
        allowed_targets = ("dataset", "actor_target", "offline_checkpoint")
        if bc_target_cfg is None:
            legacy_mode = self.config.get("bc_mode", "dataset")
            legacy_src = self.config.get("bc_teacher_source", "actor_target")
            if legacy_mode == "dataset":
                bc_target = "dataset"
            else:
                bc_target = "offline_checkpoint" if legacy_src == "offline_checkpoint" else "actor_target"
        else:
            if isinstance(bc_target_cfg, str) and (bc_target_cfg not in allowed_targets):
                bc_target = "offline_checkpoint"
            else:
                bc_target = bc_target_cfg

        bc_mode = "dataset" if bc_target == "dataset" else "actor_target"

        if bc_mode == "actor_target":
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
        mode_raw = str(self.config.get("bc_weight_mode", "none"))
        if mode_raw in ("pure", "pure_bc", "fixed", "uniform"):
            mode = "none"
        else:
            mode = mode_raw
        if mode != "none":
            action_src = str(self.config.get("bc_uncert_action_source", "bc"))
            if action_src in ("policy", "q", "pi"):
                # Need actions sampled from current policy
                actions, _ = action_distributions.sample_and_log_prob(seed=policy_rng)
                eval_actions = actions
            elif action_src == "dataset":
                eval_actions = jnp.clip(batch["actions"], -0.99, 0.99)
            elif action_src == "teacher":
                eval_actions = teacher_actions if (bc_mode == "actor_target") else jnp.clip(batch["actions"], -0.99, 0.99)
            else:
                eval_actions = teacher_actions if (bc_mode == "actor_target") else jnp.clip(batch["actions"], -0.99, 0.99)

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
                base = jnp.abs(q_delta)

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
            eval_actions = teacher_actions if (bc_mode == "actor_target") else jnp.clip(batch["actions"], -0.99, 0.99)
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
        # Gate: only when using adaptive schedule
        if self.config.get("bc_lambda_schedule", "fixed") != "adaptive":
            return 0.0, {"bc_lagrange_loss": 0.0}

        mode = str(self.config.get("bc_constraint_mode", "bc_loss"))
        bc_constraint = jnp.asarray(self.config.get("bc_constraint", 0.1), dtype=jnp.float32)

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
            ref_src = self.config.get("bc_qdrop_reference", None)
            if ref_src is None:
                bc_target_cfg = self.config.get("bc_target", None)
                allowed_targets = ("dataset", "actor_target", "offline_checkpoint")
                if bc_target_cfg is None:
                    legacy_mode = self.config.get("bc_mode", "dataset")
                    legacy_src = self.config.get("bc_teacher_source", "actor_target")
                    if legacy_mode == "dataset":
                        ref_mode = "dataset"
                    else:
                        ref_mode = "offline_checkpoint" if legacy_src == "offline_checkpoint" else "actor_target"
                else:
                    if isinstance(bc_target_cfg, str) and (bc_target_cfg not in allowed_targets):
                        ref_mode = "offline_checkpoint"
                    else:
                        ref_mode = bc_target_cfg
            else:
                ref_mode = "dataset" if ref_src == "dataset" else "actor_target"

            if ref_mode == "actor_target":
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
                ref_actions = teacher_dist.mode() if bool(self.config.get("bc_teacher_deterministic", True)) else teacher_dist.sample(seed=ref_rng)
            else:
                ref_actions = jnp.clip(batch["actions"], -0.99, 0.99)

            ref_actions = jnp.clip(ref_actions, -0.99, 0.99)
            qs_ref = self.forward_target_critic(batch["observations"], ref_actions, rng=q_ref_rng)
            q_ref = qs_ref.min(axis=0)
            chex.assert_shape(q_ref, (batch_size,))

            # Positive Q drop
            q_drop = jnp.maximum(q_ref - q_pi, 0.0)
            lhs_value = jnp.mean(q_drop)

            constraint_violation = lhs_value - bc_constraint
            positive_violation = jnp.maximum(constraint_violation, 0.0)

            bc_penalty = self.state.apply_fn(
                {"params": params},
                lhs=positive_violation,
                rhs=jnp.zeros_like(positive_violation),
                name="bc_lagrange",
            )

            lambda_value = self.forward_bc_lagrange(grad_params=params)
            complementarity = lambda_value * positive_violation
            info = {
                "bc_lagrange_loss": bc_penalty,
                "bc_constraint_lhs": lhs_value,
                "bc_constraint_rhs": bc_constraint,
                "bc_violation": constraint_violation,
                "bc_positive_violation": positive_violation,
                "bc_lambda": lambda_value,
                "bc_lambda_times_violation": complementarity,
                "bc_kkt_residual": jnp.abs(complementarity),
                "bc_qdrop_ref_code": jnp.asarray(0 if ref_mode == "dataset" else 1, dtype=jnp.int32),
            }
        else:
            # Default: constraint on BC loss magnitude
            bc_loss_value = self._compute_bc_loss_only(batch, rng)
            constraint_violation = bc_loss_value - bc_constraint
            positive_violation = jnp.maximum(constraint_violation, 0.0)

            bc_penalty = self.state.apply_fn(
                {"params": params},
                lhs=positive_violation,
                rhs=jnp.zeros_like(positive_violation),
                name="bc_lagrange",
            )

            lambda_value = self.forward_bc_lagrange(grad_params=params)
            complementarity = lambda_value * positive_violation
            info = {
                "bc_lagrange_loss": bc_penalty,
                "bc_constraint_lhs": bc_loss_value,
                "bc_constraint_rhs": bc_constraint,
                "bc_violation": constraint_violation,
                "bc_positive_violation": positive_violation,
                "bc_lambda": lambda_value,
                "bc_lambda_times_violation": complementarity,
                "bc_kkt_residual": jnp.abs(complementarity),
            }

        return bc_penalty, info

    def loss_fns(self, batch):
        """Override to include bc_lagrange loss when adaptive schedule is enabled."""
        from functools import partial as _partial
        loss_map = {
            "critic": _partial(self.critic_loss_fn, batch),
            "actor": _partial(self.policy_loss_fn, batch),
            "temperature": _partial(self.temperature_loss_fn, batch),
        }
        if self.config.get("bc_lambda_schedule", "fixed") == "adaptive":
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
        bc_lambda_schedule: str = "fixed",  # "fixed" | "adaptive"
        bc_constraint: float = 0.1,
        bc_lagrange_optimizer_kwargs: dict = {
            "learning_rate": 3e-4,
        },
        **kwargs,
    ):
        """
        Create a SAC-BC agent with optional adaptive Lagrangian to control the BC coefficient.
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

        # Optionally include bc_lagrange module when schedule is adaptive
        need_bc_module = (bc_lambda_schedule == "adaptive")
        if need_bc_module:
            # Initialize from provided bc_loss_weight if available
            init_val = float(kwargs.get("bc_loss_weight", 0.0))
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
            bc_loss_weight=kwargs.get("bc_loss_weight", 0.0),
            n_actions=kwargs.get("n_actions", 10),
            max_target_backup=kwargs.get("max_target_backup", False),
            # BC Lagrange
            bc_lambda_schedule=bc_lambda_schedule,
            bc_constraint=bc_constraint,
        )

        # Prevent external kwargs from overriding computed target_entropy
        safe_kwargs = dict(kwargs)
        safe_kwargs.pop("target_entropy", None)

        return cls(
            state=state,
            config={**base_config, **safe_kwargs},
        )

