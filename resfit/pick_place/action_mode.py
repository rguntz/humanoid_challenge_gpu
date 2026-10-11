"""Absolute and delta end-effector actions for the pick-and-place task.

The env (env.py) takes absolute targets in the robot-base frame: position (3),
rotation axis-angle (3), gripper (1: > 0 closes). A delta action is the same target
expressed relative to the end-effector pose observed at the step it is sent from
(observation.state[:7]: position, quaternion x y z w), so the policy only outputs
the motion from where the gripper is:

    delta position  = target position - eef position          (base frame)
    delta rotation  = axis-angle of R_target @ R_eef.T          (base frame)
    gripper         = unchanged

An absolute_6d action is the absolute target with its rotation as the first two rows
of the rotation matrix (10 values: position, 6D rotation, gripper). Axis-angle jumps
where the angle wraps around pi, and a network regressing it has to fit that jump;
the 6D representation is continuous (Zhou et al. 2019, "On the Continuity of
Rotation Representations in Neural Networks"), as in Diffusion Policy's absolute
actions. Any 6 values give back a rotation by Gram-Schmidt orthogonalization, so the
policy's outputs need not be exact rows of a rotation matrix. Same convention as
pytorch3d's matrix_to_rotation_6d / rotation_6d_to_matrix.

The dataset converter writes the actions with from_absolute, from the recorded
observations; the env adapter turns the policy's actions back into the env's absolute
targets with to_env, from the env's live observation. The demos' recorder
(sim/rollout_physics.py, Recorder.act) defines its delta actions the same way.
"""
import numpy as np
from scipy.spatial.transform import Rotation

ACTION_MODES = ("absolute", "delta", "absolute_6d")
ACTION_DIMS = {"absolute": 7, "delta": 7, "absolute_6d": 10}


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


def matrix_to_rotation_6d(matrix):
    """(..., 3, 3) rotation matrices -> (..., 6): the first two rows."""
    m = np.asarray(matrix)
    return m[..., :2, :].reshape(m.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6):
    """(..., 6) -> (..., 3, 3) rotation matrices, by Gram-Schmidt on the two rows."""
    d6 = np.asarray(d6, np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def to_absolute_6d(action_abs):
    """Absolute actions (..., 7) -> absolute actions with a 6D rotation (..., 10)."""
    a = np.asarray(action_abs, np.float64)
    d6 = matrix_to_rotation_6d(Rotation.from_rotvec(a[..., 3:6].reshape(-1, 3)).as_matrix())
    return np.concatenate([a[..., :3], d6.reshape(a.shape[:-1] + (6,)), a[..., 6:]], axis=-1).astype(np.float32)


def from_absolute_6d(action_6d):
    """Absolute actions with a 6D rotation (..., 10) -> absolute actions (..., 7)."""
    a = np.asarray(action_6d, np.float64)
    R = rotation_6d_to_matrix(a[..., 3:9].reshape(-1, 6))
    rot = Rotation.from_matrix(R).as_rotvec().reshape(a.shape[:-1] + (3,))
    return np.concatenate([a[..., :3], rot, a[..., 9:]], axis=-1).astype(np.float32)


def from_absolute(action_abs, mode, eef_pos, eef_quat_xyzw):
    """Absolute actions (..., 7) -> actions in the given mode (eef pose used by delta only)."""
    if mode == "delta":
        return to_delta(action_abs, eef_pos, eef_quat_xyzw)
    if mode == "absolute_6d":
        return to_absolute_6d(action_abs)
    return np.asarray(action_abs, np.float32)


def to_env(action, mode, eef_pos, eef_quat_xyzw):
    """Actions in the given mode -> the env's absolute actions (..., 7) (eef pose used by delta only)."""
    if mode == "delta":
        return to_absolute(action, eef_pos, eef_quat_xyzw)
    if mode == "absolute_6d":
        return from_absolute_6d(action)
    return np.asarray(action, np.float32)
