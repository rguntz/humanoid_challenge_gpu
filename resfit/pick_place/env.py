"""Evaluation env for the pick-and-place task: the simulated Panda puts the bottle in
the container, driven by a policy's absolute end-effector targets.

Same simulation as the demos (sim/rollout_physics.py): the scene of one demo
(env_scene.xml, written next to the dataset by make_dataset.py), the arm's
position actuators tracking joint targets that are linearly interpolated over each
control step, the gripper fully closed or fully open, and the same success test
(bottle in the container, released). Observations and actions are in the
robot-base frame (task.py).

Control: each action is a base-frame target [position (3), axis-angle (3),
gripper (1)]. Inverse kinematics (sim/replay_hand.py's solve_ik, warm-started from
the current joints, as in the demos) turns it into joint targets; the gripper
closes when the last component is > 0. step_joint takes joint targets instead
[7 arm joints (rad), gripper (1)], clipped to the joint ranges, without IK: the
demos' own recorded control (rollout_physics.Recorder's actions_joint).

Reset (no demo state is replayed): each part of the start state is drawn
independently from what the demos cover, in the base frame (the dataset's layout/
group):
  robot     the start joint positions of a random demo
  bottle    a random point in the convex hull of the demos' bottle positions
  container a random point in the convex hull of the demos' container positions,
            at least min_distance from the bottle (default: the smallest
            bottle-container distance in the demos)
  table, tag  the placement of a random demo (they show in the agentview images)

Observations are read right after stepping, without a new mj_forward, exactly as
the demos were recorded (see make_dataset.py).
"""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering, unless the caller chose otherwise

import gymnasium as gym
import h5py
import mujoco
import numpy as np
from scipy.spatial import Delaunay

from resfit.pick_place import task
from replay_hand import solve_ik  # sim/ is on sys.path (task.py)
from rollout_physics import GRIPPER_CLOSED, GRIPPER_OPEN

PICK_PLACE_ALL = ["PickPlaceBottle"]
CAMERA_KEYS = tuple(task.CAMERAS)
# terminate_on_stuck: largest change of the finger joints over the window for the gripper to
# count as still (a full opening or closing changes them by ~0.04)
STUCK_FINGER_RANGE = 0.005


class PickPlaceBottleEnv(gym.Env):
    def __init__(self, dataset_path, horizon=None, render=False, camera_names=None, min_distance=None,
                 max_tries=1000, terminate_on_fall=False, robot_start="demo", robot_start_noise_deg=5.0,
                 box_position="random", terminate_on_stuck=False, stuck_steps=48, stuck_radius=0.02):
        """Options (defaults: the original behaviour):
          terminate_on_fall      end the episode (terminated, reward 0, info["fell"]) when the bottle
                                 has fallen: its cap touches the table while the robot does not touch
                                 the bottle (upright, the cap is ~7 cm above the table; a bottle
                                 falling into the container touches the container's floor, not the
                                 table), or it dropped below the table top (knocked off the table)
          robot_start            "demo": the start joint positions of a random demo; "near_home": the
                                 home pose (the medoid of the demos' starts, as regenerate_dataset.py
                                 --home) plus a uniform offset of up to robot_start_noise_deg per arm
                                 joint, clipped to the joint ranges
          box_position           "random": a random point in the demos' container region; "fixed":
                                 always the same one, the demos' container position that leaves the
                                 largest share of the bottle region at least min_distance away (the
                                 two regions overlap: at their centroid, almost no bottle position
                                 fits); the bottle is still random
          terminate_on_stuck     end the episode (terminated, reward 0, info["stuck"]) when the robot
                                 has not moved for stuck_steps steps: every end-effector position of
                                 those steps within stuck_radius (m) of their mean, and the fingers
                                 within STUCK_FINGER_RANGE (gripper neither opening nor closing).
                                 48 steps / 2 cm cut ~35 % of the base policy's failures and none of
                                 its successes (200 rollouts of the fp32 flow matching checkpoint)
        """
        if robot_start not in ("demo", "near_home"):
            raise ValueError(f"robot_start must be 'demo' or 'near_home', got {robot_start!r}")
        if box_position not in ("random", "fixed"):
            raise ValueError(f"box_position must be 'random' or 'fixed', got {box_position!r}")
        self.terminate_on_fall = terminate_on_fall
        self.robot_start = robot_start
        self.robot_start_noise = np.radians(robot_start_noise_deg)
        self.box_position = box_position
        self.terminate_on_stuck = terminate_on_stuck
        self.stuck_steps = stuck_steps
        self.stuck_radius = stuck_radius
        self._recent = []  # the last stuck_steps observations' end-effector position and fingers
        dataset_path = Path(dataset_path)
        with h5py.File(dataset_path, "r") as f:
            self.layouts = {k: f["layout"][k][:] for k in f["layout"]}
            demos = [f["data"][k] for k in f["data"]]
            control_freq = float(demos[0].attrs["control_freq"])
            longest = max(int(d.attrs["num_samples"]) for d in demos)
            self.image_size = demos[0]["obs"]["agentview_image"].shape[1:3]
        self.horizon = horizon or int(np.ceil(1.5 * longest))

        self.model = mujoco.MjModel.from_xml_path(str(dataset_path.parent / "env_scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.scene = task.Scene(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.pb, self.Rb = task.base_pose(self.model, self.data)
        if abs(task.base_yaw_deg(self.Rb) - 90.0) > 1e-3:
            raise ValueError("env_scene.xml must have the robot base at yaw 90 deg, like the demos")
        mujoco.mj_resetDataKeyframe(self.model, self.data, mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "panda/home"))
        self.q_home = self.data.qpos[:task.ARM].copy()
        self.control_freq = control_freq
        self.steps_per_control = int(round(1.0 / control_freq / self.model.opt.timestep))

        geom = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.table, self.tag = geom("table"), geom("tag")
        bottle_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bottle")
        self.bottle_adr = self.model.jnt_qposadr[bottle_jnt]
        self.bottle_z = self.model.qpos0[self.bottle_adr + 2]  # standing on the table
        self.bottle_hull = Delaunay(self.layouts["bottle_xy"])
        self.box_hull = Delaunay(self.layouts["box_xy"])
        dist = np.linalg.norm(self.layouts["box_xy"] - self.layouts["bottle_xy"], axis=1)
        self.min_distance = dist.min() if min_distance is None else min_distance
        self.max_tries = max_tries
        q_starts = self.layouts["robot_qpos"][:, :task.ARM]
        d_starts = np.linalg.norm(q_starts[:, None] - q_starts[None], axis=-1)
        self.home_row = int(np.argmin(d_starts.sum(1)))  # medoid of the demos' start poses
        self.box_fixed = self._roomiest_box()
        self.cap_geom = geom("bottle_cap")
        self.table_top_z = self.bottle_z - 0.05  # well below a standing bottle's centre: off the table

        # Vision policies never get the ground-truth object poses (same rule as xqc's
        # robomimic env and dataset loader).
        self.use_object_obs = camera_names is None
        self.camera_names = list(camera_names or [])
        self.renderer = None
        if render or self.camera_names:
            self.renderer = mujoco.Renderer(self.model, *self.image_size)
            self.render_opt = mujoco.MjvOption()
            self.render_opt.geomgroup[1] = False  # phone marker, as in the demos
            self.render_opt.sitegroup[:] = False

        n_obs = len(task.flat_obs(task.world_lowdim_obs(self.data, self.scene), self.use_object_obs))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.rng = np.random.default_rng()
        self.time_step = 0

    # ---------------------------------------------------------------- reset
    def _sample_in(self, hull, points):
        lo, hi = points.min(0), points.max(0)
        for _ in range(self.max_tries):
            p = self.rng.uniform(lo, hi)
            if hull.find_simplex(p) >= 0:
                return p
        raise RuntimeError("could not sample a point in the demos' region")

    def _roomiest_box(self):
        """Among the demos' container positions, the one that leaves the largest share of the bottle
        region at least min_distance away (estimated on a fixed grid of samples: deterministic)."""
        L = self.layouts
        lo, hi = L["bottle_xy"].min(0), L["bottle_xy"].max(0)
        pts = np.random.default_rng(0).uniform(lo, hi, (20000, 2))
        pts = pts[self.bottle_hull.find_simplex(pts) >= 0]
        room = [(np.linalg.norm(pts - box, axis=1) >= self.min_distance).mean() for box in L["box_xy"]]
        return L["box_xy"][int(np.argmax(room))].copy()

    def _to_world(self, p_base):
        return self.pb + self.Rb @ np.asarray(p_base, float)

    def _sample_layout(self):
        L, n = self.layouts, len(self.layouts["bottle_xy"])
        robot, backdrop = self.rng.integers(n, size=2)
        if self.robot_start == "near_home":
            q_robot = L["robot_qpos"][self.home_row].copy()
            lo, hi = self.model.jnt_range[:task.ARM].T
            q_robot[:task.ARM] = np.clip(
                q_robot[:task.ARM] + self.rng.uniform(-self.robot_start_noise, self.robot_start_noise, task.ARM), lo, hi)
        else:
            q_robot = L["robot_qpos"][robot]
        for _ in range(self.max_tries):
            bottle = self._sample_in(self.bottle_hull, L["bottle_xy"])
            box = self.box_fixed if self.box_position == "fixed" else self._sample_in(self.box_hull, L["box_xy"])
            if np.linalg.norm(box - bottle) >= self.min_distance:
                return q_robot, bottle, box, backdrop
        raise RuntimeError(f"no bottle / container placement {self.min_distance:.3f} m apart")

    def reset(self, seed=None, options=None):
        """options={"demo": i} starts from demo i's layout instead of a random one
        (to check the env against the demos)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        L = self.layouts
        if options and "demo" in options:
            i = options["demo"]
            q_robot, bottle, box, backdrop = L["robot_qpos"][i], L["bottle_xy"][i], L["box_xy"][i], i
        else:
            q_robot, bottle, box, backdrop = self._sample_layout()

        # Table and tag where a random demo had them (base frame -> this scene).
        size = np.abs(self.Rb) @ L["table_half"][backdrop]
        m.geom_pos[self.table] = self._to_world(L["table_center"][backdrop])
        m.geom_size[self.table] = size
        m.geom_rbound[self.table] = np.linalg.norm(size)
        m.geom_aabb[self.table] = np.concatenate([np.zeros(3), size])
        m.geom_pos[self.tag] = self._to_world(L["tag_pos"][backdrop])
        m.body_pos[self.scene.cont] = self._to_world([box[0], box[1], 0.0])

        mujoco.mj_resetData(m, d)
        d.qpos[:task.ARM + 2] = q_robot
        d.qpos[self.bottle_adr:self.bottle_adr + 3] = self._to_world([bottle[0], bottle[1], 0.0])[:2].tolist() \
            + [self.bottle_z]
        d.qpos[self.bottle_adr + 3:self.bottle_adr + 7] = [1, 0, 0, 0]
        d.ctrl[:task.ARM] = q_robot[:task.ARM]
        d.ctrl[task.ARM] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)
        self.q_target = q_robot[:task.ARM].copy()
        self.time_step = 0
        self._recent = []
        return self._obs(), {}

    # ----------------------------------------------------------------- step
    def step(self, action):
        """Absolute base-frame end-effector target (7), through IK."""
        action = np.asarray(action, np.float64).reshape(-1)
        p, R = task.from_base_pose(action[:3], action[3:6], self.pb, self.Rb)
        self.ik_data.qpos[:] = self.data.qpos
        solve_ik(self.model, self.ik_data, self.scene.site, p, R, self.q_home)
        return self._advance(self.ik_data.qpos[:task.ARM].copy(), action[6] > 0)

    def step_joint(self, action):
        """Joint targets (7, rad) and gripper (1), no IK."""
        action = np.asarray(action, np.float64).reshape(-1)
        lo, hi = self.model.jnt_range[:task.ARM].T
        return self._advance(np.clip(action[:task.ARM], lo, hi), action[task.ARM] > 0)

    def _advance(self, q_next, close):
        """One control step: the joint targets move linearly from the previous ones to q_next."""
        q_prev = self.q_target
        d = self.data
        d.ctrl[task.ARM] = GRIPPER_CLOSED if close else GRIPPER_OPEN
        for s in range(self.steps_per_control):
            a = s / self.steps_per_control
            d.ctrl[:task.ARM] = (1 - a) * q_prev + a * q_next
            mujoco.mj_step(self.model, d)
        self.q_target = q_next
        self.time_step += 1

        success, _, _, touch_robot = self.scene.in_box(d)
        success = bool(success)
        fell = False
        if self.terminate_on_fall and not success and not touch_robot:
            fell = self._bottle_fell()
        obs = self._obs()
        stuck = False
        if self.terminate_on_stuck and not success and not fell:
            stuck = self._robot_stuck(obs)
        truncated = self.time_step >= self.horizon
        return obs, float(success), success or fell or stuck, truncated, \
            {"success": success, "fell": fell, "stuck": stuck}

    def _robot_stuck(self, obs):
        """The last stuck_steps observations (this one included) are still: see terminate_on_stuck."""
        self._recent = (self._recent + [np.concatenate([obs[0:3], obs[7:9]])])[-self.stuck_steps:]
        if len(self._recent) < self.stuck_steps:
            return False
        w = np.asarray(self._recent)
        pos_still = np.linalg.norm(w[:, :3] - w[:, :3].mean(0), axis=1).max() < self.stuck_radius
        return bool(pos_still and np.ptp(w[:, 3:], 0).max() < STUCK_FINGER_RANGE)

    def _bottle_fell(self):
        """The bottle's cap touches the table, or the bottle dropped below the table top."""
        d = self.data
        if d.xpos[self.scene.bottle][2] < self.table_top_z:
            return True
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if {g1, g2} == {self.cap_geom, self.table}:
                return True
        return False

    # ---------------------------------------------------------- observation
    def _obs(self):
        obs = task.obs_to_base(task.world_lowdim_obs(self.data, self.scene), self.pb, self.Rb)
        return task.flat_obs(obs, self.use_object_obs)

    def get_images(self, camera_keys):
        """uint8 (H, W, 3) per key, rendered like the demos (no flip needed: MuJoCo's
        Renderer already returns upright images)."""
        out = {}
        for key in camera_keys:
            self.renderer.update_scene(self.data, camera=task.CAMERAS[key], scene_option=self.render_opt)
            out[key] = self.renderer.render().copy()
        return out

    def render(self):
        """All cameras side by side (agentview, then wrist)."""
        return np.hstack(list(self.get_images(CAMERA_KEYS).values()))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def make_pick_place_env(env_name, dataset_path, render=False, camera_names=None, horizon=None):
    assert env_name in PICK_PLACE_ALL, env_name
    if dataset_path is None:
        raise ValueError("PickPlaceBottle needs the dataset (agent.offline_data_path): its layout/ group "
                         "and env_scene.xml define the initial states and the scene")
    return PickPlaceBottleEnv(dataset_path, horizon=horizon, render=render, camera_names=camera_names)
