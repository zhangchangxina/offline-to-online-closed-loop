import os
from functools import partial

import gym
import jax
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.core.frozen_dict import freeze, unfreeze
from flax.training import checkpoints
from ml_collections import config_flags

from experiments.configs.ensemble_config import add_redq_config
from wsrl.agents import agents
from wsrl.common.evaluation import evaluate_with_trajectories
from wsrl.common.wandb import WandBLogger
from wsrl.data.replay_buffer import ReplayBuffer, ReplayBufferMC
from wsrl.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation
from wsrl.envs.d4rl_dataset import (
    get_d4rl_dataset,
    get_d4rl_dataset_with_mc_calculation,
)
from wsrl.envs.env_common import get_env_type, make_gym_env
from wsrl.utils.timer_utils import Timer
from wsrl.utils.train_utils import concatenate_batches, subsample_batch

FLAGS = flags.FLAGS

# env
flags.DEFINE_string("env", "antmaze-large-diverse-v2", "Environemnt to use")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias.")
flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)

# training
flags.DEFINE_integer("num_offline_steps", 1_000_000, "Number of offline epochs.")
flags.DEFINE_integer("num_online_steps", 500_000, "Number of online epochs.")
flags.DEFINE_float(
    "offline_data_ratio",
    0.0,
    "How much offline data to retain in each online batch update",
)
flags.DEFINE_string(
    "online_sampling_method",
    "mixed",
    """Method of sampling data during online update: mixed or append.
    `mixed` samples from a mix of offline and online data according to offline_data_ratio.
    `append` adds offline data to replay buffer and samples from it.""",
)
flags.DEFINE_bool(
    "online_use_cql_loss",
    True,
    """When agent is CQL/CalQL, whether to use CQL loss for the online phase (use SAC loss if False)""",
)
flags.DEFINE_integer(
    "warmup_steps", 0, "number of warmup steps (WSRL) before performing online updates"
)
flags.DEFINE_bool(
    "warmup_update_critic",
    False,
    "If true, perform critic-only updates during warmup (after enough online data).",
)

# agent
flags.DEFINE_string("agent", "calql", "what RL agent to use")
flags.DEFINE_integer("utd", 1, "update-to-data ratio of the critic")
flags.DEFINE_integer("batch_size", 256, "batch size for training")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")
flags.DEFINE_bool("use_redq", False, "Use an ensemble of Q-functions for the agent")

# experiment house keeping
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "save_dir",
    "../wsrl_log",
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path", "", "Path to resume from")
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps.")
flags.DEFINE_integer(
    "n_eval_trajs", 20, "Number of trajectories to use for each evaluation."
)
flags.DEFINE_bool("deterministic_eval", True, "Whether to use deterministic evaluation")

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    """
    house keeping
    """
    assert FLAGS.online_sampling_method in [
        "mixed",
        "append",
    ], "incorrect online sampling method"

    if FLAGS.use_redq:
        FLAGS.config.agent_kwargs = add_redq_config(FLAGS.config.agent_kwargs)

    min_steps_to_update = FLAGS.batch_size * (1 - FLAGS.offline_data_ratio)
    if FLAGS.agent == "calql":
        min_steps_to_update = max(
            min_steps_to_update, gym.make(FLAGS.env)._max_episode_steps
        )

    """
    wandb and logging
    """
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "wsrl" or FLAGS.project,
            "group": "wsrl" or FLAGS.group,
            "exp_descriptor": f"{FLAGS.exp_name}_{FLAGS.env}_{FLAGS.agent}_seed{FLAGS.seed}",
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        random_str_in_identifier=True,
        disable_online_logging=FLAGS.debug,
    )

    save_dir = os.path.join(
        FLAGS.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    """
    env
    """
    # do not clip adroit actions online following CalQL repo
    # https://github.com/nakamotoo/Cal-QL
    env_type = get_env_type(FLAGS.env)
    finetune_env = make_gym_env(
        env_name=FLAGS.env,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed,
    )
    eval_env = make_gym_env(
        env_name=FLAGS.env,
        scale_and_clip_action=env_type in ("antmaze", "kitchen", "locomotion"),
        action_clip_lim=FLAGS.clip_action,
        seed=FLAGS.seed + 1000,
    )

    """
    load dataset
    """
    if env_type == "adroit-binary":
        dataset = get_hand_dataset_with_mc_calculation(
            FLAGS.env,
            gamma=FLAGS.config.agent_kwargs.discount,
            reward_scale=FLAGS.reward_scale,
            reward_bias=FLAGS.reward_bias,
            clip_action=FLAGS.clip_action,
        )
    else:
        if FLAGS.agent == "calql":
            # need dataset with mc return
            dataset = get_d4rl_dataset_with_mc_calculation(
                FLAGS.env,
                reward_scale=FLAGS.reward_scale,
                reward_bias=FLAGS.reward_bias,
                clip_action=FLAGS.clip_action,
                gamma=FLAGS.config.agent_kwargs.discount,
            )
        else:
            dataset = get_d4rl_dataset(
                FLAGS.env,
                reward_scale=FLAGS.reward_scale,
                reward_bias=FLAGS.reward_bias,
                clip_action=FLAGS.clip_action,
            )

    """
    replay buffer
    """
    replay_buffer_type = ReplayBufferMC if FLAGS.agent == "calql" else ReplayBuffer
    replay_buffer = replay_buffer_type(
        finetune_env.observation_space,
        finetune_env.action_space,
        capacity=FLAGS.replay_buffer_capacity,
        seed=FLAGS.seed,
        discount=FLAGS.config.agent_kwargs.discount if FLAGS.agent == "calql" else None,
    )

    """
    Initialize agent
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, construct_rng = jax.random.split(rng)
    example_batch = subsample_batch(dataset, FLAGS.batch_size)
    agent = agents[FLAGS.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        actions=example_batch["actions"],
        encoder_def=None,
        **FLAGS.config.agent_kwargs,
    )

    if FLAGS.resume_path != "":
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        try:
            agent = checkpoints.restore_checkpoint(FLAGS.resume_path, target=agent)
        except Exception as e:
            logging.warning(
                "Standard restore failed (%s). Falling back to partial restore for backward compatibility.",
                repr(e),
            )
            # Load raw checkpoint without enforcing exact structure
            ckpt_obj = checkpoints.restore_checkpoint(FLAGS.resume_path, target=None)

            # Extract state (supports both saved agent objects or raw state dicts)
            ckpt_state = getattr(ckpt_obj, "state", ckpt_obj)

            def merge_params_like(target_params, source_params):
                tgt = unfreeze(target_params)
                try:
                    src = unfreeze(source_params)
                except Exception:
                    src = source_params

                def _merge(a, b):
                    if isinstance(a, dict) and isinstance(b, dict):
                        out = {}
                        for k, v in a.items():
                            if k in b:
                                out[k] = _merge(v, b[k])
                            else:
                                out[k] = v
                        return out
                    # leaf: prefer source if shape compatible
                    try:
                        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape:
                            return b
                    except Exception:
                        pass
                    return a

                merged = _merge(tgt, src if isinstance(src, dict) else {})
                return freeze(merged)

            # Safely merge params and target_params; keep current opt_states/txs
            merged_params = merge_params_like(agent.state.params, getattr(ckpt_state, "params", {}))
            merged_target_params = merge_params_like(
                agent.state.target_params, getattr(ckpt_state, "target_params", {})
            )

            # Step and rng if present
            new_step = getattr(ckpt_state, "step", agent.state.step)
            new_rng = getattr(ckpt_state, "rng", agent.state.rng)

            agent = agent.replace(
                state=agent.state.replace(
                    params=merged_params,
                    target_params=merged_target_params,
                    step=new_step,
                    rng=new_rng,
                )
            )

    """
    eval function
    """

    def evaluate_and_log_results(
        eval_env,
        policy_fn,
        eval_func,
        step_number,
        wandb_logger,
        n_eval_trajs=FLAGS.n_eval_trajs,
    ):
        stats, trajs = eval_func(
            policy_fn,
            eval_env,
            n_eval_trajs,
        )

        eval_info = {
            "average_return": np.mean([np.sum(t["rewards"]) for t in trajs]),
            "average_traj_length": np.mean([len(t["rewards"]) for t in trajs]),
        }
        if env_type == "adroit-binary":
            # adroit
            eval_info["success_rate"] = np.mean(
                [any(d["goal_achieved"] for d in t["infos"]) for t in trajs]
            )
        elif env_type == "kitchen":
            # kitchen
            eval_info["num_stages_solved"] = np.mean([t["rewards"][-1] for t in trajs])
            eval_info["success_rate"] = np.mean([t["rewards"][-1] for t in trajs]) / 4
        else:
            # d4rl antmaze, locomotion
            eval_info["success_rate"] = eval_info[
                "average_normalized_return"
            ] = np.mean(
                [eval_env.get_normalized_score(np.sum(t["rewards"])) for t in trajs]
            )

        wandb_logger.log({"evaluation": eval_info}, step=step_number)

    """
    training loop
    """
    timer = Timer()
    step = int(agent.state.step)  # 0 for new agents, or load from pre-trained
    is_online_stage = False
    observation, info = finetune_env.reset()
    done = False  # env done signal

    for _ in tqdm.tqdm(range(step, FLAGS.num_offline_steps + FLAGS.num_online_steps)):
        """
        Switch from offline to online
        """
        if not is_online_stage and step >= FLAGS.num_offline_steps:
            logging.info("Switching to online training")
            is_online_stage = True

            # upload offline data to online buffer
            if FLAGS.online_sampling_method == "append":
                offline_dataset_size = dataset["actions"].shape[0]
                dataset_items = dataset.items()
                for j in range(offline_dataset_size):
                    transition = {k: v[j] for k, v in dataset_items}
                    replay_buffer.insert(transition)

            # option for CQL and CalQL to change the online alpha, and whether to use CQL regularizer
            if FLAGS.agent in ("cql", "calql"):
                online_agent_configs = {
                    "cql_alpha": FLAGS.config.agent_kwargs.get(
                        "online_cql_alpha", None
                    ),
                    "use_cql_loss": FLAGS.online_use_cql_loss,
                }
                agent.update_config(online_agent_configs)

            # enable closed-loop extras only for online stage
            if FLAGS.agent == "closed_loop_sac":
                # Enable closed-loop extras only for online stage
                agent.update_config({"closed_loop_enabled": True})
                # If using linear schedule, start decay at the online switch step
                if agent.config.get("lambda_schedule", "fixed") == "linear":
                    agent.update_config({
                        "lam_eff_linear_start_step": int(agent.state.step)
                    })

            # If using SAC-BC, optionally enable BC only for first N online steps
            if FLAGS.agent == "sac_bc":
                online_bc_steps = int(FLAGS.config.agent_kwargs.get("bc_online_enable_for_steps", -1))
                if online_bc_steps and online_bc_steps > 0:
                    agent.update_config({
                        "bc_online_start_step": int(agent.state.step),
                        "bc_online_enable_for_steps": online_bc_steps,
                    })

                # Optionally load an offline checkpoint as the fixed BC teacher
                # Support unified bc_target: if it's a path (not a known keyword), use it as teacher
                target_cfg = FLAGS.config.agent_kwargs.get("bc_target", "")
                if isinstance(target_cfg, str) and target_cfg not in ("dataset", "actor_target", "offline_checkpoint") and len(target_cfg) > 0:
                    FLAGS.config.agent_kwargs["bc_offline_ckpt_teacher"] = target_cfg
                    FLAGS.config.agent_kwargs["bc_teacher_source"] = "offline_checkpoint"
                ckpt_teacher_path = FLAGS.config.agent_kwargs.get("bc_offline_ckpt_teacher", "")
                if ckpt_teacher_path:
                    try:
                        ckpt_obj = checkpoints.restore_checkpoint(ckpt_teacher_path, target=None)
                        ckpt_state = getattr(ckpt_obj, "state", ckpt_obj)
                        teacher_params = getattr(ckpt_state, "target_params", None) or getattr(ckpt_state, "params", None)
                        if teacher_params is not None:
                            from wsrl.agents.sac_bc import SACBCWithTargetAgent  # local import to avoid cycles
                            assert isinstance(agent, SACBCWithTargetAgent) or hasattr(agent, "set_offline_teacher_params")
                            agent = agent.set_offline_teacher_params(teacher_params)
                            agent.update_config({"bc_teacher_source": "offline_checkpoint"})
                            logging.info("Loaded offline checkpoint teacher from %s", ckpt_teacher_path)
                        else:
                            logging.warning("Checkpoint at %s did not contain params/target_params; skipping offline teacher.", ckpt_teacher_path)
                    except Exception as e:
                        logging.warning("Failed to load offline teacher from %s: %s", ckpt_teacher_path, repr(e))

        timer.tick("total")

        """
        Env Step
        """
        with timer.context("env step"):
            if is_online_stage:
                rng, action_rng = jax.random.split(rng)
                action = agent.sample_actions(observation, seed=action_rng)
                # Guard against NaNs/Infs and enforce clip range before env step
                action = jax.device_get(action)
                action = np.asarray(action, dtype=np.float32)
                action = np.nan_to_num(
                    action,
                    nan=0.0,
                    posinf=FLAGS.clip_action,
                    neginf=-FLAGS.clip_action,
                )
                if FLAGS.clip_action is not None:
                    action = np.clip(action, -FLAGS.clip_action, FLAGS.clip_action)
                next_observation, reward, done, truncated, info = finetune_env.step(
                    action
                )

                transition = dict(
                    observations=observation,
                    next_observations=next_observation,
                    actions=action,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=1.0 if (done or truncated) else 0,
                )
                replay_buffer.insert(transition)

                observation = next_observation
                if done or truncated:
                    observation, info = finetune_env.reset()
                    done = False

        """
        Updates
        """
        with timer.context("update"):
            # offline updates
            if not is_online_stage:
                batch = subsample_batch(dataset, FLAGS.batch_size)
                agent, update_info = agent.update(
                    batch,
                )

            # online updates
            else:
                if step - FLAGS.num_offline_steps <= max(
                    FLAGS.warmup_steps, min_steps_to_update
                ):
                    # Warmup phase: optionally update critic only if enough online data
                    if FLAGS.warmup_update_critic and len(replay_buffer) >= min_steps_to_update:
                        if FLAGS.online_sampling_method == "mixed":
                            # batch from a mixing ratio of offline and online data
                            batch_size_offline = int(
                                FLAGS.batch_size * FLAGS.offline_data_ratio
                            )
                            batch_size_online = FLAGS.batch_size - batch_size_offline
                            online_batch = replay_buffer.sample(batch_size_online)
                            offline_batch = subsample_batch(dataset, batch_size_offline)
                            # update with the combined batch
                            batch = concatenate_batches([online_batch, offline_batch])
                        elif FLAGS.online_sampling_method == "append":
                            # batch from online replay buffer, with is initialized with offline data
                            batch = replay_buffer.sample(FLAGS.batch_size)
                        else:
                            raise RuntimeError("Incorrect online sampling method")

                        # critic-only update (single step, no UTD to keep warmup simple)
                        agent, update_info = agent.update(
                            batch,
                            networks_to_update=frozenset({"critic"}),
                        )
                    else:
                        # no updates during warmup
                        pass
                else:
                    # do online updates, gather batch
                    if FLAGS.online_sampling_method == "mixed":
                        # batch from a mixing ratio of offline and online data
                        batch_size_offline = int(
                            FLAGS.batch_size * FLAGS.offline_data_ratio
                        )
                        batch_size_online = FLAGS.batch_size - batch_size_offline
                        online_batch = replay_buffer.sample(batch_size_online)
                        offline_batch = subsample_batch(dataset, batch_size_offline)
                        # update with the combined batch
                        batch = concatenate_batches([online_batch, offline_batch])
                    elif FLAGS.online_sampling_method == "append":
                        # batch from online replay buffer, with is initialized with offline data
                        batch = replay_buffer.sample(FLAGS.batch_size)
                    else:
                        raise RuntimeError("Incorrect online sampling method")

                    # update
                    if FLAGS.utd > 1:
                        agent, update_info = agent.update_high_utd(
                            batch,
                            utd_ratio=FLAGS.utd,
                        )
                    else:
                        agent, update_info = agent.update(
                            batch,
                        )

        """
        Advance Step
        """
        step += 1

        """
        Evals
        """
        eval_steps = (
            FLAGS.num_offline_steps,  # finish offline training
            FLAGS.num_offline_steps + 1,  # start of online training
            FLAGS.num_offline_steps + FLAGS.num_online_steps,  # end of online training
        )
        if step % FLAGS.eval_interval == 0 or step in eval_steps:
            logging.info("Evaluating...")
            with timer.context("evaluation"):
                policy_fn = partial(
                    agent.sample_actions, argmax=FLAGS.deterministic_eval
                )
                eval_func = partial(
                    evaluate_with_trajectories, clip_action=FLAGS.clip_action
                )

                evaluate_and_log_results(
                    eval_env=eval_env,
                    policy_fn=policy_fn,
                    eval_func=eval_func,
                    step_number=step,
                    wandb_logger=wandb_logger,
                )

        """
        Save Checkpoint
        """
        if step % FLAGS.save_interval == 0 or step == FLAGS.num_offline_steps:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=step, keep=30
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        """
        Logging
        """
        if step % FLAGS.log_interval == 0:
            # check if update_info is available (False during warmup)
            if "update_info" in locals():
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=step)

            wandb_logger.log({"timer": timer.get_average_times()}, step=step)


if __name__ == "__main__":
    app.run(main)