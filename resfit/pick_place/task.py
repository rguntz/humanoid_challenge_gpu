"""The pick-and-place task as the policy sees it, shared by the dataset conversion
(make_dataset.py) and the evaluation env (env.py) so both compute exactly the same
observations and actions.

Robot-base frame. Every episode's scene (sim/build_scene.py) is in the tag frame,
with the robot base placed behind that episode's workspace, so the same arm pose
has different tag-frame coordinates from one episode to the next. The policy works
in the frame of the robot base (body panda/link0) instead: positions, orientations
and actions are all expressed relative to it.

Observations (robomimic key names, the ones xqc's dataset loader reads):
  robot0_eef_pos       (3)   tool centre point position
  robot0_eef_quat      (4)   its orientation, quaternion x y z w (w >= 0)
  robot0_gripper_qpos  (2)   finger joint positions
  object               (13)  bottle position (3) and quaternion (4), bottle minus
                             tool centre point (3), container position (3)
  agentview_image      (H, W, 3) uint8, the fixed camera across the table
  robot0_eye_in_hand_image  (H, W, 3) uint8, the camera on the robot's wrist

Actions (absolute end-effector target, the recorded actions_abs):
  position (3), rotation axis-angle (3), gripper (1: -1 open, +1 close)
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
from rollout_physics import ARM, Scene, quat_xyzw  # noqa: E402

BASE_BODY = "panda/link0"
LOWDIM_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
# Image observation key -> MuJoCo camera, as the recorder (sim/rollout_physics.py) renders them.
CAMERAS = {"agentview_image": "agentview", "robot0_eye_in_hand_image": "panda/wrist"}


def base_pose(model, data):
    """Robot base position and rotation matrix in the scene (tag) frame."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def to_base_pos(p, pb, Rb):
    """Points (..., 3) from the scene frame to the base frame."""
    return (np.asarray(p) - pb) @ Rb


def to_base_vec(v, Rb):
    """Directions / differences (..., 3): rotate only."""
    return np.asarray(v) @ Rb


def to_base_quat(q_xyzw, Rb):
    """Quaternions (..., 4, x y z w) from the scene frame to the base frame, w >= 0."""
    R = Rotation.from_matrix(Rb.T) * Rotation.from_quat(np.asarray(q_xyzw))
    return R.as_quat(canonical=True).astype(np.float32)


def to_base_rotvec(rv, Rb):
    """Axis-angle rotations (..., 3) from the scene frame to the base frame."""
    return (Rotation.from_matrix(Rb.T) * Rotation.from_rotvec(np.asarray(rv))).as_rotvec().astype(np.float32)


def from_base_pose(pos, rotvec, pb, Rb):
    """Base-frame target (position, axis-angle) back to a scene-frame position and rotation matrix."""
    return pb + Rb @ np.asarray(pos), Rb @ Rotation.from_rotvec(rotvec).as_matrix()


def world_lowdim_obs(data, scene):
    """Low-dim observations in the scene frame, exactly as rollout_physics.Recorder.observe
    records them."""
    site, b = scene.site, scene.bottle
    eef_pos = data.site_xpos[site].copy()
    return {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": quat_xyzw(data.site_xmat[site]),
        "robot0_gripper_qpos": data.qpos[ARM:ARM + 2].copy(),
        "object": np.concatenate([data.xpos[b], quat_xyzw(data.xmat[b]), data.xpos[b] - eef_pos,
                                  data.xpos[scene.cont]]),
    }


def obs_to_base(obs, pb, Rb):
    """Low-dim observations (one step or a stack of steps) from the scene frame to the
    base frame. Used on the recorded demos and on the env's live observations."""
    obj = np.asarray(obs["object"], np.float64)
    return {
        "robot0_eef_pos": to_base_pos(obs["robot0_eef_pos"], pb, Rb).astype(np.float32),
        "robot0_eef_quat": to_base_quat(obs["robot0_eef_quat"], Rb),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float32),
        "object": np.concatenate([
            to_base_pos(obj[..., 0:3], pb, Rb),
            to_base_quat(obj[..., 3:7], Rb),
            to_base_vec(obj[..., 7:10], Rb),
            to_base_pos(obj[..., 10:13], pb, Rb),
        ], axis=-1).astype(np.float32),
    }


def actions_to_base(actions_abs, pb, Rb):
    """Recorded absolute actions (N, 7) from the scene frame to the base frame."""
    a = np.asarray(actions_abs, np.float64)
    return np.concatenate([to_base_pos(a[:, :3], pb, Rb), to_base_rotvec(a[:, 3:6], Rb), a[:, 6:]],
                          axis=-1).astype(np.float32)


def flat_obs(obs, use_object):
    """The state vector the policy gets: LOWDIM_KEYS concatenated in order (as xqc's
    load_robomimic_flat does), without "object" for vision policies."""
    keys = LOWDIM_KEYS if use_object else LOWDIM_KEYS[:-1]
    return np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in keys])


def layout(model, data, scene):
    """Where things are at the start of an episode, in the base frame: the robot's
    joint positions, the bottle and the container (x, y on the table), and the table
    and tag (which show in the agentview images)."""
    pb, Rb = base_pose(model, data)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    tag = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tag")
    return {
        "robot_qpos": data.qpos[:ARM + 2].copy(),
        "bottle_xy": to_base_pos(data.xpos[scene.bottle], pb, Rb)[:2],
        "box_xy": to_base_pos(data.xpos[scene.cont], pb, Rb)[:2],
        "table_center": to_base_pos(model.geom_pos[table], pb, Rb),
        "table_half": np.abs(Rb.T) @ model.geom_size[table],  # base yaw is a multiple of 90 deg
        "tag_pos": to_base_pos(model.geom_pos[tag], pb, Rb),
    }


def base_yaw_deg(Rb):
    return float(np.degrees(np.arctan2(Rb[1, 0], Rb[0, 0])))
