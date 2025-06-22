from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "arena.xml"
_PANDA_HOME = np.asarray((0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickCubeGymEnvMultiSize(MujocoGymEnv):
    """Enhanced PandaPickCubeGymEnv with support for different camera image sizes."""
    
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.05, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        # New parameters for different camera sizes
        front_camera_size: Tuple[int, int] = (256, 256),  # (height, width)
        wrist_camera_size: Tuple[int, int] = (128, 128),  # (height, width)
    ):
        self._action_scale = action_scale
        self.reward_type = reward_type
        self.front_camera_size = front_camera_size
        self.wrist_camera_size = wrist_camera_size

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self.render_mode = render_mode
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)
        self.image_obs = image_obs
        
        # Create separate renderers for different camera sizes
        self._front_renderer = mujoco.Renderer(
            self.model,
            height=front_camera_size[0],
            width=front_camera_size[1]
        )
        self._wrist_renderer = mujoco.Renderer(
            self.model,
            height=wrist_camera_size[0],
            width=wrist_camera_size[1]
        )
        
        # Caching.
        self._panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id
        self._block_z = self._model.geom("block").size[2]

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float32
                        ),
                        "tcp_vel": spaces.Box(
                            -np.inf, np.inf, shape=(6,), dtype=np.float32
                        ),
                        "gripper_pose": spaces.Box(
                            -1, 1, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {
                            "tcp_pose": spaces.Box(
                                -np.inf, np.inf, shape=(7,), dtype=np.float32
                            ),
                            "tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(6,), dtype=np.float32
                            ),
                            "gripper_pose": spaces.Box(
                                -1, 1, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": spaces.Dict(
                        {
                            "front": spaces.Box(
                                low=0,
                                high=255,
                                shape=(front_camera_size[0], front_camera_size[1], 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                low=0,
                                high=255,
                                shape=(wrist_camera_size[0], wrist_camera_size[1], 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        # block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
        block_xy = np.asarray([0.5, 0.0])
        self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        x, y, z, rx, ry, rz, grasp = action

        # Set the mocap position.
        pos = self._data.mocap_pos[0].copy()
        dpos = np.asarray([x, y, z]) * self._action_scale[0]
        npos = np.clip(pos + dpos, *_CARTESIAN_BOUNDS)
        self._data.mocap_pos[0] = npos

        # Set gripper grasp.
        g = self._data.ctrl[self._gripper_ctrl_id] / 255
        dg = grasp * self._action_scale[1]
        ng = np.clip(g + dg, 0.0, 1.0)
        self._data.ctrl[self._gripper_ctrl_id] = ng * 255

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self._model,
                data=self._data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self._data.mocap_pos[0],
                ori=self._data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self._data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self._model, self._data)
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()
        block_pos = self._data.sensor("block_pos").data
        outside_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05))
        terminated = self.time_limit_exceeded() or success or outside_bounds

        return obs, rew, terminated, False, {"succeed": success}

    def render(self):
        """Render both cameras with their respective sizes."""
        rendered_frames = []
        
        # Render front camera with front camera size
        self._front_renderer.update_scene(self.data, camera=self.camera_id[0])
        front_frame = self._front_renderer.render()
        rendered_frames.append(front_frame)
        
        # Render wrist camera with wrist camera size
        self._wrist_renderer.update_scene(self.data, camera=self.camera_id[1])
        wrist_frame = self._wrist_renderer.render()
        rendered_frames.append(wrist_frame)
        
        return rendered_frames

    # def close(self) -> None:
    #     """Close all renderers."""
    #     super().close()
    #     if hasattr(self, '_front_renderer'):
    #         self._front_renderer.close()
    #     if hasattr(self, '_wrist_renderer'):
    #         self._wrist_renderer.close()

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        tcp_quat = self._data.sensor("2f85/pinch_quat").data

        obs["state"]["tcp_pose"] = np.concatenate([tcp_pos, tcp_quat]).astype(np.float32)

        tcp_vel = self._data.sensor("2f85/pinch_vel").data
        tcp_angvel = self._data.sensor("2f85/pinch_angvel").data
        obs["state"]["tcp_vel"] = np.concatenate([tcp_vel, tcp_angvel]).astype(np.float32)

        gripper_pose = np.array(
            self._data.ctrl[self._gripper_ctrl_id] / 255, dtype=np.float32
        )
        obs["state"]["gripper_pose"] = gripper_pose

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        return obs

    def _compute_reward(self) -> float:
        if self.reward_type == "dense":
            block_pos = self._data.sensor("block_pos").data
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            rew = 0.3 * r_close + 0.7 * r_lift
            return rew
        else:
            block_pos = self._data.sensor("block_pos").data
            lift = block_pos[2] - self._z_init
            return float(lift > 0.2)

    def _is_success(self) -> bool:
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.2


if __name__ == "__main__":
    # Example usage with different camera sizes
    env = PandaPickCubeGymEnvMultiSize(
        render_mode="rgb_array",
        image_obs=True,
        front_camera_size=(256, 256),  # Higher resolution for front camera
        wrist_camera_size=(128, 128),  # Lower resolution for wrist camera
    )
    
    obs, _ = env.reset()
    print("Front camera shape:", obs["images"]["front"].shape)
    print("Wrist camera shape:", obs["images"]["wrist"].shape)
    
    for i in range(10):
        action = np.random.uniform(-1, 1, 7)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    env.close()
