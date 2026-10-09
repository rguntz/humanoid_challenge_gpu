"""Absolute and delta end-effector actions for the pick-and-place task.

The env (env.py) takes absolute targets in the robot-base frame: position (3),
rotation axis-angle (3), gripper (1: > 0 closes). A delta action is the same target
expressed relative to the end-effector pose observed at the step it is sent from
(observation.state[:7]: position, quaternion x y z w), so the policy only outputs
the motion from where the gripper is:

    delta position  = target position - eef position          (base frame)
    delta rotation  = axis-angle of R_target @ R_eef.T          (base frame)
    gripper         = unchanged

The dataset converter writes delta actions with to_delta, from the recorded
observations; the env adapter turns the policy's delta actions back into absolute
targets with to_absolute, from the env's live observation. The demos' recorder
(sim/rollout_physics.py, Recorder.act) defines its delta actions the same way.
"""
import numpy as np
from scipy.spatial.transform import Rotation

ACTION_MODES = ("absolute", "delta")


def to_delta(action_abs, eef_pos, eef_quat_xyzw):
    """Absolute actions (..., 7) -> delta actions relative to the given eef poses."""
    a = np.asarray(action_abs, np.float64)
    R_eef = Rotation.from_quat(np.asarray(eef_quat_xyzw, np.float64))
    drot = (Rotation.from_rotvec(a[..., 3:6]) * R_eef.inv()).as_rotvec()
    return np.concatenate([a[..., :3] - eef_pos, drot, a[..., 6:]], axis=-1).astype(np.float32)


def to_absolute(action_delta, eef_pos, eef_quat_xyzw):
    """Delta actions (..., 7) relative to the given eef poses -> absolute actions."""
    a = np.asarray(action_delta, np.float64)
    R_eef = Rotation.from_quat(np.asarray(eef_quat_xyzw, np.float64))
    rot = (Rotation.from_rotvec(a[..., 3:6]) * R_eef).as_rotvec()
    return np.concatenate([a[..., :3] + eef_pos, rot, a[..., 6:]], axis=-1).astype(np.float32)
