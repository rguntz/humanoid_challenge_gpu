"""Robomimic-convention helpers shared by chunked_qc_buffer.py: the low-dim
observation keys and the absolute-action conversion. (In xqcfd this module also
held the non-chunked OfflineReplayBuffer, which is not used here.)"""

from typing import List

import jax.numpy as jnp
import numpy as np

from xqc.agents.common.rotation_transformer import axis_angle_to_rotation_6d

ROBOSUITE_OBS_KEYS: List[str] = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


def robosuite_obs_keys_for_num_robots(num_robots: int, include_object: bool = True) -> List[str]:
    """Generalizes ROBOSUITE_OBS_KEYS to num_robots>1 (e.g. TwoArmTransport) -
    per-robot [robotN_eef_pos, robotN_eef_quat, robotN_gripper_qpos] blocks
    concatenated in robot order, "object" (shared, not per-robot) last -
    matches robomimic's own per-episode obs field naming convention and the
    concatenation order ROBOSUITE_OBS_KEYS already uses for num_robots=1.
    robosuite_obs_keys_for_num_robots(1, True) == ROBOSUITE_OBS_KEYS."""
    keys = []
    for i in range(num_robots):
        keys += [f"robot{i}_eef_pos", f"robot{i}_eef_quat", f"robot{i}_gripper_qpos"]
    if include_object:
        keys.append("object")
    return keys


def convert_abs_action_batch(actions: np.ndarray, num_robots: int) -> np.ndarray:
    """pos(3) + axis-angle-rot(3) + gripper(1), PER ROBOT -> pos(3) +
    rotation_6d(6) + gripper(1), per robot - batched over a leading
    (timestep) axis, one episode's worth of actions at once. Matches the
    reference Diffusion Policy implementation's own _data_to_obs
    (robomimic_replay_lowdim_dataset.py), which reshapes to (...,2,7) for
    dual-arm before the identical per-robot split/convert/concat, then back
    to (...,20). Lives here (not in xqc/replay_buffer/chunked_qc_buffer.py
    or xqc/agents/common/dataset.py, both of which had their own identical
    copy of this formula before this dedup) because those two files import
    FROM each other's shared pieces already (dataset.py's
    ChunkedNormalizationStats/compute_stats, imported into
    chunked_qc_buffer.py) - putting it in either would create a circular
    import; this module (xqc/replay_buffer/offline_replay_buffer.py) is
    already a shared dependency of both, imports nothing from either, and
    already hosts the analogous ROBOSUITE_OBS_KEYS/
    robosuite_obs_keys_for_num_robots robomimic-convention helpers - same
    home as _convert_abs_action, chunked_qc_buffer.py's own single-
    transition (online ReplayBuffer.insert) equivalent of this formula."""
    stacked = actions.reshape(*actions.shape[:-1], num_robots, 7)
    pos, rot, gripper = stacked[..., :3], stacked[..., 3:6], stacked[..., 6:]
    rot6d = np.asarray(axis_angle_to_rotation_6d(jnp.asarray(rot)))
    return np.concatenate([pos, rot6d, gripper], axis=-1).reshape(
        *actions.shape[:-1], num_robots * 10
    ).astype(np.float32)
