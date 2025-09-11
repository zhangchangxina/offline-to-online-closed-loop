import jax
import jax.numpy as jnp
import chex
import distrax
import flax

from typing import Optional

from wsrl.agents.sac import SACAgent
from wsrl.common.common import nonpytree_field
from wsrl.common.typing import Batch, Data, Params, PRNGKey


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
            "dataset_rewards": batch["rewards"],
            "mc_returns": batch.get("mc_returns", None),
            "actions": actions,
            "q_term": q_term,
            "entropy_term": entropy_term,
        }

        # Optional BC regularization (supports dataset or actor_target as teacher, and TD-weighted coefficient)
        # Online-only gating in a JAX-traceable way
        bc_weight = jnp.asarray(float(self.config.get("bc_loss_weight", 0.0)), dtype=jnp.float32)
        step = self.state.step.astype(jnp.int32)
        online_start_val = jnp.asarray(int(self.config.get("bc_online_start_step", -1)), dtype=jnp.int32)
        online_for_val = jnp.asarray(int(self.config.get("bc_online_enable_for_steps", -1)), dtype=jnp.int32)

        within_online_window = (
            (online_start_val >= 0)
            & (online_for_val > 0)
            & (step >= online_start_val)
            & (step < (online_start_val + online_for_val))
        )
        bc_enabled_bool = (bc_weight > 0.0) & within_online_window
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
            else:
                # Stochastic teacher: sample multiple candidates and pick the one with max Q(s, a)
                # Number of candidates (default 10)
                n_candidates = int(self.config.get("bc_teacher_n_candidates", 10))
                n_candidates = max(1, n_candidates)
                rng, teacher_sample_rng, critic_rng_teacher = jax.random.split(rng, 3)
                cand_actions = teacher_dist.sample(seed=teacher_sample_rng, sample_shape=(n_candidates,))
                # cand_actions: (n, batch, act_dim) -> (batch, n, act_dim)
                cand_actions = jnp.transpose(cand_actions, (1, 0, 2))
                # Evaluate target critic Q(s, a) for all candidates; result: (ensemble, batch, n)
                qs_cand = self.forward_target_critic(
                    batch["observations"], cand_actions, rng=critic_rng_teacher
                )
                # Min over ensemble, then argmax over candidates
                q_min = qs_cand.min(axis=0)  # (batch, n)
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

        # TD-weighted coefficient (optional). Weight is per-sample and stop-gradient.
        if bool(self.config.get("bc_td_weight_enabled", False)):
            q_delta = self._compute_td_q_delta(
                batch,
                td_rng,
                policy_grad_params=params,
            )
            use_abs = bool(self.config.get("bc_td_weight_abs", True))
            power = float(self.config.get("bc_td_weight_power", 1.0))
            scale = float(self.config.get("bc_td_weight_scale", 1.0))
            clip_val = float(self.config.get("bc_td_weight_clip", -1.0))

            weights = jnp.abs(q_delta) if use_abs else q_delta
            # Optional inverse mapping for dataset BC target: smaller |delta| -> larger weight
            if bool(self.config.get("bc_td_weight_inverse", False)) and (bc_mode != "actor_target"):
                eps = float(self.config.get("bc_td_weight_inverse_eps", 1e-3))
                weights = 1.0 / (jnp.abs(q_delta) + eps)
            if power != 1.0:
                weights = jnp.power(weights, power)
            if clip_val > 0.0:
                weights = jnp.clip(weights, 0.0, clip_val)
            if bool(self.config.get("bc_td_weight_normalize", False)):
                weights = weights / (jnp.mean(weights) + 1e-8)
            if scale != 1.0:
                weights = weights * scale
            # Final safety clip with a reasonable default cap
            weights = jnp.clip(
                weights, 0.0, float(self.config.get("bc_td_weight_clip", 3.0))
            )
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

        # Log diagnostics
        info["bc_loss"] = bc_loss
        info["bc_neglogp_mean"] = bc_per.mean()
        info["bc_weight_mean"] = weights.mean()
        info["bc_mode_code"] = jnp.asarray(1 if bc_mode == "actor_target" else 0, dtype=jnp.int32)
        info["bc_target_code"] = jnp.asarray({"dataset":0, "actor_target":1, "offline_checkpoint":2}.get(bc_target, 0), dtype=jnp.int32)
        info["actor_loss"] = actor_loss
        info["bc_enabled"] = bc_enabled_bool.astype(jnp.int32)
        # Reduce logging payload to scalars for performance
        info["log_probs_mean"] = log_probs.mean()
        info.pop("log_probs", None)
        info.pop("actions", None)

        return actor_loss, info


