#!/usr/bin/env python3

import glob
import time
import os
import copy
import pickle as pkl

# Configure JAX memory management before importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from flax.core import frozen_dict
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.conrft_single_octo_cp import ConrftCPOctoAgentSingleArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from data_util import add_mc_returns_to_trajectory, add_next_embeddings_to_trajectory

from serl_launcher.utils.launcher import (
    make_conrft_octo_cp_pixel_agent_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

from octo.model.octo_model import OctoModel

import mujoco.viewer

from franka_sim.utils.viewer_utils import DualMujocoViewer


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0,
                     "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")

flags.DEFINE_float("gamma", 0.95, "return discount")
flags.DEFINE_float("reward_neg", 0.0, "reward_neg for spase reward envs")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_float("q_weight", 0.1, "q_weight ")
flags.DEFINE_float("bc_weight", 1.0, "bc_weight")

flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(tasks, agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []
        episode_length_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions, _ = agent.sample_actions(
                    observations=jax.device_put(obs),
                    tasks=jax.device_put(tasks),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward > 0:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        episode_length_list.append(info["episode"]["l"])
                        print(dt)
                        print(info["episode"]["l"])

                    success_counter += 1 if reward > 0 else 0
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average episode length: {np.mean(episode_length_list)}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(
            FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    trajectory = []

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    with  mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data, show_left_ui=False, show_right_ui=False) as viewer:
        for step in pbar:
            timer.tick("total")
            viewer.sync()

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions, action_embeddings = agent.sample_actions(
                        observations=jax.device_put(obs),
                        tasks=jax.device_put(tasks),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)
                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                    intervened=already_intervened,
                    embeddings=action_embeddings,
                )
                if 'grasp_penalty' in info:
                    transition['grasp_penalty'] = info['grasp_penalty']

                trajectory.append(transition)

                obs = next_obs
                if done or truncated:
                    trajectory = add_mc_returns_to_trajectory(trajectory, FLAGS.gamma,
                                                            FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.reward_neg, is_sparse_reward=False
                                                            )
                    trajectory = add_next_embeddings_to_trajectory(trajectory)
                    for transition in trajectory:
                        data_store.insert(transition)
                        transitions.append(copy.deepcopy(transition))
                        if transition['intervened']:
                            intvn_data_store.insert(transition)
                            demo_transitions.append(copy.deepcopy(transition))

                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    info["episode"]["succeed"] = int(info['succeed'])
                    info["episode"]["total_steps"] = step
                    # send stats to the learner to log
                    stats = {"environment": info}
                    client.request("send-stats", stats)
                    pbar.set_description(f"last return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    client.update()
                    trajectory = []
                    obs, _ = env.reset()

            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                # dump to pickle file
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(
                    FLAGS.checkpoint_path, "demo_buffer")
                if not os.path.exists(buffer_path):
                    os.makedirs(buffer_path)
                if not os.path.exists(demo_buffer_path):
                    os.makedirs(demo_buffer_path)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []
                with open(
                    os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
                ) as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)


##############################################################################


def learner(rng, tasks, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(
            FLAGS.checkpoint_path))[11:]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step
    online_start_step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(),
                           request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    train_critic_networks_to_update = frozenset({"critic"})
    train_actor_networks_to_update = frozenset({"actor"})
    train_networks_to_update = frozenset({"critic", "actor"})

    def create_batch_tasks(data_dict, batch_size):
        batch_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):  # Handling nested dictionary (e.g., language_instruction)
                batch_dict[key] = {k: np.tile(
                    v, (batch_size, *([1] * (v.ndim - 1)))) for k, v in value.items()}
            else:
                # For non-dictionary values, repeat along batch dimension (axis=0)
                batch_dict[key] = np.tile(
                    value, (batch_size, *([1] * (value.ndim - 1))))  # Repeat along axis 0

        return batch_dict

    # Pretrain the model with the demo data
    if step < FLAGS.pretrain_steps:
        print_green("Pretraining the model with demo data")
        for step in tqdm.tqdm(range(start_step, FLAGS.pretrain_steps + 1), desc="pretraining"):
            for _ in range(config.cta_ratio - 1):
                batch = next(demo_buffer.get_iterator(
                    sample_args={"batch_size": config.batch_size,
                                 "pack_obs": True, },
                    device=sharding.replicate(),
                ))

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
                batch = frozen_dict.freeze(batch)
                agent, critics_info = agent.update_calql(
                    batch, networks_to_update=train_critic_networks_to_update,)

            batch = next(demo_buffer.get_iterator(
                sample_args={"batch_size": config.batch_size,
                             "pack_obs": True, },
                device=sharding.replicate(),
            ))

            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)

            agent, update_info = agent.update_calql(
                batch, networks_to_update=train_networks_to_update,)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)

            if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state, step=step, keep=100)

        print_green("Pretraining done")
        return  # after pretraining, return and exit
    else:
        print_green(
            "Existing pretrained checkpoint model found. Skipping pretraining")

    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # Start online training after offline pretraining
    online_start_step = FLAGS.pretrain_steps + \
        1 if online_start_step < FLAGS.pretrain_steps else online_start_step
    for step in tqdm.tqdm(range(online_start_step, config.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
            batch = frozen_dict.freeze(batch)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_ql(
                    batch, networks_to_update=train_critic_networks_to_update,)

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)
            agent, update_info = agent.update_ql(
                batch, networks_to_update=train_networks_to_update,)
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=step, keep=100)


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner, save_video=FLAGS.eval_checkpoint_step, classifier=False, stack_obs_num=2)
    env = RecordEpisodeStatistics(env)

    FLAGS.reward_neg = config.reward_neg

    # Convert checkpoint path to absolute path to fix Orbax requirement
    if FLAGS.checkpoint_path is not None:
        FLAGS.checkpoint_path = os.path.abspath(FLAGS.checkpoint_path)

    rng, sampling_rng = jax.random.split(rng)

    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

    if config.setup_mode == 'single-arm-fixed-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            fix_gripper=True,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
        )
        include_grasp_penalty = False
        include_octo_embeddings = True
        include_mc_returns = True
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
        )
        include_grasp_penalty = True
        include_octo_embeddings = True
        include_mc_returns = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree.map(
        jnp.array, agent), sharding.replicate())

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        if not FLAGS.learner:
            input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path, agent.state,)
        # agent = agent.replace(state=ckpt)

        # Update params only, ignore the optimizer states
        new_params = ckpt.params
        new_target_params = ckpt.target_params

        agent = agent.replace(state=agent.state.replace(
            params=new_params, target_params=new_target_params))

        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(FLAGS.checkpoint_path))[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        # set up wandb and logging

        wandb_logger = make_wandb_logger(
            project="conrft",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        assert FLAGS.demo_path is not None

        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(sampling_rng,
                tasks,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                wandb_logger=wandb_logger,
                )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(tasks,
              agent,
              data_store,
              intvn_data_store,
              env,
              sampling_rng,
              )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
