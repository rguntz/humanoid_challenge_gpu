"""The pick-and-place env as ResFiT uses it: N simulator copies in parallel, with the
observations and actions of the LeRobot dataset the policy is trained on. Replaces
resfit/dexmg/environments/dexmg.py (robosuite / DexMimicGen) for this task.

PickPlaceGymEnv wraps one simulator (env.py, PickPlaceBottleEnv), which takes absolute
end-effector targets (step) or joint targets (step_joint):
  observations  observation.state (9, float32: end-effector pose and fingers, or the arm
                and finger joints for joint datasets, action_mode.STATE_MODES),
                observation.images.agentview and
                observation.images.wrist ((3, H, W) float32 in [0, 1]), the
                dataset's features in the format LeRobotDataset returns them
  actions       (7,) or (10,) in the dataset's action mode, turned into the
                simulator's absolute targets by action_mode.to_env: delta actions
                from the end-effector pose of the current observation (the
                definition the dataset was converted with), absolute_6d actions by
                turning their 6D rotation back into axis-angle; joint actions (8,)
                go to step_joint as they are.
  episodes      end on success (terminated, reward 1) or at the env's horizon
                (truncated); the last step's info has "success" and "episode_steps",
                as ResFiT's robosuite wrapper gives them.
  render()      both cameras side by side, (H, 2W, 3) uint8.

create_vectorized_env runs them in parallel with gymnasium's vector envs (one process
per env, finished episodes reset in the same step, the final observation in
info["final_obs"]) behind ResFiT's VectorizedEnvWrapper, which returns torch tensors.
It takes the LeRobot dataset folder: the action mode and the simulation scene come
from the dataset's metadata, so the env always matches the data the policy saw.
"""
import json
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from resfit.pick_place import task
from resfit.pick_place.action_mode import ACTION_DIMS, STATE_MODES, to_env
from resfit.pick_place.env import CAMERA_KEYS, PickPlaceBottleEnv
from resfit.pick_place.task import IMAGE_FEATURES

REPO = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def dataset_env_config(dataset_dir):
    """The action mode and the robot-base source dataset (its layout/ group and
    env_scene.xml define the simulation) of a LeRobot dataset written by
    resfit/lerobot/dataset/convert_pick_place_to_lerobot.py."""
    info = json.loads((Path(dataset_dir) / "meta" / "info.json").read_text())
    return info["action_mode"], REPO / info["source"]


class PickPlaceGymEnv(gym.Env):
    """One simulator with the dataset's observations and actions (see the module docstring)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, source_path, action_mode, horizon=None):
        self.env = PickPlaceBottleEnv(source_path, horizon=horizon, camera_names=list(CAMERA_KEYS))
        self.action_mode = action_mode
        self.render_mode = "rgb_array"
        # horizon: the episode length limit, read by the residual RL (train_residual_td3.py)
        self.metadata = {**self.metadata, "render_fps": self.env.control_freq, "horizon": self.env.horizon}
        h, w = self.env.image_size
        self.observation_space = gym.spaces.Dict({
            "observation.state": self.env.observation_space,
            **{IMAGE_FEATURES[k]: gym.spaces.Box(0.0, 1.0, (3, h, w), np.float32) for k in CAMERA_KEYS},
        })
        # Nominal, as in ResFiT's robosuite wrapper: only the shape is used; the values
        # are the dataset's actions, scaled by ResFiT from the dataset's statistics.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIMS[action_mode],), np.float32)
        self._state = None
        self.episode_steps = 0
        self.num_layouts = len(self.env.layouts["robot_qpos"])  # one per recorded demo
        # Demo layouts the next resets start from, in order (autoresets included); empty:
        # random layouts. Set per env with vec_env.set_attr("demo_queue", [[...], ...]).
        self.demo_queue = []

    def _observe(self, state):
        self._state = state  # end-effector pose and fingers: delta actions are relative to it
        if STATE_MODES[self.action_mode] == "joint":
            state = self.env.data.qpos[:task.ARM + 2]  # arm and finger joints
        obs = {"observation.state": np.asarray(state, np.float32)}
        for key, image in self.env.get_images(CAMERA_KEYS).items():
            obs[IMAGE_FEATURES[key]] = np.ascontiguousarray(image.transpose(2, 0, 1), np.float32) / 255.0
        return obs

    def reset(self, *, seed=None, options=None):
        """options={"demo": row} starts from the demo's layout (PickPlaceBottleEnv.reset);
        without options, the next layout of demo_queue if any."""
        super().reset(seed=seed)
        if not options and self.demo_queue:
            self.demo_queue = list(self.demo_queue)
            options = {"demo": self.demo_queue.pop(0)}
        state, info = self.env.reset(seed=seed, options=options)
        self.episode_steps = 0
        return self._observe(state), info

    def step(self, action):
        action = np.asarray(action.cpu() if hasattr(action, "cpu") else action, np.float32).reshape(-1)
        if self.action_mode == "joint":
            state, reward, terminated, truncated, info = self.env.step_joint(action)
        else:
            action = to_env(action, self.action_mode, self._state[0:3], self._state[3:7])
            state, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        if terminated or truncated:
            info = {**info, "episode_steps": self.episode_steps}
        return self._observe(state), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class VectorizedEnvWrapper:
    """ResFiT's wrapper (resfit/dexmg/environments/dexmg.py): returns torch tensors and
    renders every env."""

    def __init__(self, vec_env, video_key, device="cpu"):
        self.vec_env = vec_env
        self.video_key = video_key
        self._last_obs = None
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self._last_obs = obs
        obs = self._convert_obs_to_torch(obs, self.device)
        return obs, info

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            # On the CPU: a CUDA tensor sent to the env processes would start CUDA in each (~0.5 GB)
            actions = actions.detach().float().cpu().numpy()
        obs, rewards, terminated, truncated, info = self.vec_env.step(actions)
        self._last_obs = obs

        # Convert to torch tensors
        obs = self._convert_obs_to_torch(obs, self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)

        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return RGB frames from all environments (num_envs, H, W, 3, uint8) for video recording."""
        frames = self.vec_env.render()
        if frames is None:
            raise RuntimeError("No frames returned from vectorized environment")
        return np.stack(frames)

    @property
    def fps(self):
        return self.vec_env.metadata["render_fps"]

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)

    def _convert_obs_to_torch(self, obs, device):
        """Convert numpy observations from vectorized env to PyTorch tensors for policy."""
        if isinstance(obs, dict):
            torch_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    torch_obs[key] = torch.from_numpy(value).to(device)
                else:
                    torch_obs[key] = value
            return torch_obs
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).to(device)
        return obs


def create_vectorized_env(
    dataset_dir,
    num_envs: int,
    device: str = "cpu",
    debug: bool = False,
    horizon: int | None = None,
    video_key: str = "observation.images.agentview",
) -> VectorizedEnvWrapper:
    """num_envs simulators matching the LeRobot dataset in dataset_dir (see the module
    docstring); debug runs them in this process, one after the other."""
    action_mode, source_path = dataset_env_config(dataset_dir)
    env_fns = [lambda: PickPlaceGymEnv(source_path, action_mode, horizon) for _ in range(num_envs)]
    if debug:
        vec_env = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=True,
            copy=True,
            context="spawn",
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    wrapped_env = VectorizedEnvWrapper(vec_env, video_key, device)
    wrapped_env.env_name = "PickPlaceBottle"
    wrapped_env.action_mode = action_mode
    logger.debug(f"Created {num_envs} vectorized PickPlaceBottle environments ({action_mode} actions)")
    return wrapped_env
