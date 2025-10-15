"""
Implementation of AWAC (Advantage Weighted Actor-Critic) algorithm.
"""
import copy
from functools import partial
from typing import Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from wsrl.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.networks.actor_critic_nets import Critic, Policy, ValueCritic, ensemblize
from wsrl.networks.mlp import MLP


def awac_actor_loss(q, v, dist, actions, temperature=1.0, adv_clip_max=100.0, mask=None):
    """
    AWAC actor loss with advantage weighting.
    """
    adv = q - v
    
    # Clip advantages to prevent numerical instability
    adv = jnp.clip(adv, -adv_clip_max, adv_clip_max)
    
    # Compute advantage weights
    exp_adv = jnp.exp(adv * temperature)
    exp_adv = jnp.minimum(exp_adv, adv_clip_max)
    
    # Compute log probabilities
    log_probs = dist.log_prob(actions)
    
    # AWAC loss: weighted by advantage
    actor_loss = -(exp_adv * log_probs)
    
    if mask is not None:
        actor_loss *= mask
        actor_loss = jnp.sum(actor_loss) / jnp.sum(mask)
    else:
        actor_loss = jnp.mean(actor_loss)
    
    pi_actions = dist.mode()
    behavior_mse = jnp.square(pi_actions - actions).sum(-1)
    
    return actor_loss, {
        "actor_loss": actor_loss,
        "behavior_logprob": log_probs.mean(),
        "behavior_entropy": -log_probs.mean(),
        "behavior_mse": behavior_mse.mean(),
        "advantage_mean": adv.mean(),
        "advantage_max": adv.max(),
        "advantage_min": adv.min(),
        "exp_adv_mean": exp_adv.mean(),
        "exp_adv_max": exp_adv.max(),
        "exp_adv_min": exp_adv.min(),
        # Action stats (dataset vs policy)
        "batch_actions_var": jnp.var(actions, axis=0).mean(),
        "pi_actions_var": jnp.var(pi_actions, axis=0).mean(),
        "batch_actions_mean": jnp.mean(actions, axis=0).mean(),
        "pi_actions_mean": jnp.mean(pi_actions, axis=0).mean(),
        "batch_actions_sq_mean": jnp.mean(jnp.square(actions), axis=0).mean(),
        "pi_actions_sq_mean": jnp.mean(jnp.square(pi_actions), axis=0).mean(),
        "batch_actions_max": jnp.max(actions),
        "batch_actions_min": jnp.min(actions),
        "pi_actions_max": jnp.max(pi_actions),
        "pi_actions_min": jnp.min(pi_actions),
        "predicted_actions": pi_actions,
        "dataset_actions": actions,
    }


class AWACAgent(flax.struct.PyTreeNode):
    """
    AWAC (Advantage Weighted Actor-Critic) agent implementation.
    """
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients)
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        qs = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )
        return qs

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, train=False, grad_params=self.state.target_params
        )

    def forward_value(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for value network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="value",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_value(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target value network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_value(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def _get_ensemble_q_value(self, q, rng):
        """
        Subsample to a single critic value given an ensemble.
        """
        if self.config["critic_subsample_size"] is not None:
            # REDQ-style subsampling
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            q = q[subsample_idcs]
            q = jnp.min(q, axis=0)
        else:
            # Double Q-learning
            q = jnp.min(q, axis=0)
        return q, rng

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Critic loss using TD learning.
        """
        batch_size = batch["rewards"].shape[0]

        rng, key = jax.random.split(rng)
        next_v = self.forward_value(batch["next_observations"], key)
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        chex.assert_shape(target_q, (batch_size,))

        rng, key = jax.random.split(rng)
        q = self.forward_critic(
            batch["observations"],
            batch["actions"],
            key,
            grad_params=params,
        )
        chex.assert_shape(q, (self.config["critic_ensemble_size"], batch_size))

        # MSE loss
        critic_loss = jnp.square(q - target_q)
        # TD error stats (Q - Q_target) before TD loss
        td_error = q - target_q
        td_error_mean = jnp.mean(td_error)
        td_error_min = jnp.min(td_error)
        td_error_max = jnp.max(td_error)
        chex.assert_shape(
            critic_loss, (self.config["critic_ensemble_size"], batch_size)
        )

        return critic_loss.mean(), {
            "critic_loss": critic_loss.mean(),
            "q": q.mean(),
            "target_q": target_q.mean(),
            "td_error_mean": td_error_mean,
            "td_error_min": td_error_min,
            "td_error_max": td_error_max,
        }

    def value_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Value function loss using expectile regression.
        """
        rng, key = jax.random.split(rng)
        q = self.forward_target_critic(
            batch["observations"], batch["actions"], key
        )  # no gradient
        q, rng = self._get_ensemble_q_value(q, rng)  # min over Q functions

        rng, key = jax.random.split(rng)
        v = self.forward_value(batch["observations"], key, grad_params=params)

        # Expectile loss for value function
        diff = q - v
        weight = jnp.where(diff > 0, self.config["expectile"], (1 - self.config["expectile"]))
        value_loss = weight * (diff ** 2)

        return value_loss.mean(), {
            "value_loss": value_loss.mean(),
            "uncentered_loss": jnp.mean((q - v) ** 2),
            "v": v.mean(),
        }

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        AWAC policy loss with advantage weighting.
        """
        rng, key = jax.random.split(rng)

        if self.config["update_actor_with_target_adv"]:
            critic_fn = self.forward_target_critic
        else:
            critic_fn = self.forward_critic

        rng, key = jax.random.split(rng)
        dist = self.forward_policy(batch["observations"], key, grad_params=params)
        mask = batch.get("actor_loss_mask", None)

        # Get Q-values for current actions
        q = critic_fn(batch["observations"], batch["actions"], key)  # no gradient
        q, rng = self._get_ensemble_q_value(q, rng)  # min over Q functions

        rng, key = jax.random.split(rng)
        v = self.forward_value(batch["observations"], key)  # no gradients

        return awac_actor_loss(
            q,
            v,
            dist,
            batch["actions"],
            self.config["temperature"],
            self.config["adv_clip_max"],
            mask=mask,
        )

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        """
        Update all networks using AWAC algorithm.
        """
        rng, new_rng = jax.random.split(self.state.rng)
        batch_size = batch["rewards"].shape[0]

        loss_fns = {
            "critic": partial(self.critic_loss_fn, batch),
            "value": partial(self.value_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
        }

        # Compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # Update rng
        new_state = new_state.replace(rng=new_rng)

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
    ) -> Tuple["AWACAgent", dict]:
        """
        High-UTD update:
        - Perform `utd_ratio` critic+value-only updates on minibatches (with target updates each step).
        - Then perform one actor-only update on the full batch (no target update here).

        Batch dimension must be divisible by `utd_ratio`.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        if self.config.get("debug_checks", False):
            chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple["AWACAgent"], data: Tuple[Batch]):
            (agent,) = carry
            (minibatch,) = data

            rng, new_rng = jax.random.split(agent.state.rng)

            # Critic + value updates only (actor grads zeroed)
            loss_fns = {
                "critic": partial(agent.critic_loss_fn, minibatch),
                "value": partial(agent.value_loss_fn, minibatch),
                "actor": lambda params, rng: (jnp.array(0.0), {}),
            }
            new_state, info = agent.state.apply_loss_fns(
                loss_fns, pmap_axis=pmap_axis, has_aux=True
            )

            # Target updates after critic/value updates
            new_state = new_state.target_update(agent.config["target_update_rate"])
            # Update RNG
            new_state = new_state.replace(rng=new_rng)

            return (agent.replace(state=new_state),), info

        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (utd_ratio, minibatch_size) + data.shape[1:])

        minibatches = jax.tree_util.tree_map(make_minibatch, batch)

        (agent,), critic_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

        # Average critic/value infos across UTD steps
        critic_infos = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)
        # Remove actor info from critic-only phase if present
        try:
            del critic_infos["actor"]
        except Exception:
            pass

        # One actor-only update on the full batch (no target update)
        rng, new_rng = jax.random.split(agent.state.rng)
        loss_fns = {
            "critic": lambda params, rng: (jnp.array(0.0), {}),
            "value": lambda params, rng: (jnp.array(0.0), {}),
            "actor": partial(agent.policy_loss_fn, batch),
        }
        new_state, actor_infos = agent.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )
        new_state = new_state.replace(rng=new_rng)
        agent = agent.replace(state=new_state)

        # Remove critic/value placeholders from actor step if present
        try:
            del actor_infos["critic"]
        except Exception:
            pass
        try:
            del actor_infos["value"]
        except Exception:
            pass

        infos = {**critic_infos, **actor_infos}
        return agent, infos

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax=False,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy.
        """
        dist = self.forward_policy(observations, seed, train=False)
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """
        Get debug metrics for monitoring training.
        """
        dist = self.forward_policy(batch["observations"], train=False)
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).mean()

        v = self.forward_value(batch["observations"], train=False)
        next_v = self.forward_value(batch["next_observations"], train=False)
        target = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        q = self.forward_critic(batch["observations"], batch["actions"], train=False)
        q, _ = self._get_ensemble_q_value(q, self.state.rng)  # min over Q functions

        metrics = {
            "log_probs": log_probs,
            "action_mse": mse,
            "pi_actions": pi_actions,
            "v": v.mean(),
            "q": q.mean(),
            "advantage": target - v,
            "qf_advantage": q - v,
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "kernel_init_type": "var_scaling",
            "kernel_scale_final": 1e-2,
        },
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "kernel_init_type": "var_scaling",
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
        },
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        value_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount=0.99,
        expectile=0.9,
        temperature=1.0,
        target_update_rate=0.005,
        update_actor_with_target_adv=True,
        adv_clip_max=100.0,
        **kwargs,
    ):
        """
        Create a new AWAC agent.
        """
        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "value": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "value": copy.deepcopy(encoder_def),
                "critic": copy.deepcopy(encoder_def),
            }

        networks = {
            "actor": Policy(
                encoders["actor"],
                MLP(**policy_network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            ),
            "value": ValueCritic(encoders["value"], MLP(**critic_network_kwargs)),
            "critic": Critic(
                encoders["critic"],
                network=ensemblize(
                    partial(MLP, **critic_network_kwargs), critic_ensemble_size
                )(name="critic_ensemble"),
            ),
        }

        model_def = ModuleDict(networks)

        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "value": make_optimizer(**value_critic_optimizer_kwargs),
            "critic": make_optimizer(**value_critic_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            value=[observations],
            critic=[observations, actions],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                expectile=expectile,
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                update_actor_with_target_adv=update_actor_with_target_adv,
                adv_clip_max=adv_clip_max,
            )
        )
        return cls(state, config)
