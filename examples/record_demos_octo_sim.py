import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

import mujoco
import mujoco.viewer

from experiments.mappings import CONFIG_MAPPING
from data_util import add_mc_returns_to_trajectory, add_embeddings_to_trajectory, add_next_embeddings_to_trajectory

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20,
                     "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False, save_video=False, classifier=False, stack_obs_num=2)

    model = OctoModel.load_pretrained(config.octo_path)
    tasks = model.create_tasks(texts=[config.task_desc])
    # model = None
    # tasks = None

    obs, info = env.reset()
    print(obs.keys())
    print("Reset done")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    print("Press shift to start recording.\nIf your controller is not working check controller_type (default is xbox) is configured in examples/experiments/pick_cube_sim/config.py")
    with  mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)
            viewer.sync()
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)

            pbar.set_description(f"Return: {returns:.2f}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    trajectory = add_mc_returns_to_trajectory(
                        trajectory, config.discount, FLAGS.reward_scale, FLAGS.reward_bias, config.reward_neg, is_sparse_reward=True)
                    trajectory = add_embeddings_to_trajectory(
                        trajectory, model, tasks=tasks)
                    trajectory = add_next_embeddings_to_trajectory(trajectory)
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                obs, info = env.reset()
                time.sleep(2.0)
            if success_count >= success_needed:
                break

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")


if __name__ == "__main__":
    app.run(main)
