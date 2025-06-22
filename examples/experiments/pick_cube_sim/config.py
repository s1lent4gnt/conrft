import os
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    JoystickIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    ControllerType
)

from experiments.pick_cube_sim.wrapper import GripperPenaltyWrapper

from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from franka_sim.envs.panda_pick_gym_env_multi_size import PandaPickCubeGymEnvMultiSize


class TrainConfig(DefaultTrainingConfig):
    controller_type = ControllerType.XBOX
    image_keys = ["front", "wrist"]
    classifier_keys = ["front", "wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    buffer_period = 2000
    replay_buffer_capacity = 50000
    batch_size = 128
    random_steps = 0
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    fake_env = False
    classifier = False
    octo_path = "hf://rail-berkeley/octo-small-1.5"
    task_desc = "pick up the cube"
    reward_neg = False

    def get_environment(self, fake_env=False, save_video=False, classifier=False, stack_obs_num=1):
        # env = PandaPickCubeGymEnv(render_mode="human", image_obs=True, reward_type="sparse", time_limit=100.0, control_dt=0.1)
        env = PandaPickCubeGymEnvMultiSize(render_mode="human", image_obs=True, reward_type="sparse", time_limit=100.0, control_dt=0.1)
        if not fake_env:
            env = JoystickIntervention(env=env, controller_type=self.controller_type)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=stack_obs_num, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs))[0] > 0.95)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
