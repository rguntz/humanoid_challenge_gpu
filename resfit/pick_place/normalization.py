"""Per-dimension range normalization of the pick-and-place state and actions, the
formulation of the earlier JAX pipeline (policy/xqc/replay_buffer/chunked_qc_buffer.py:
normalizer="per_dim_range" for the state, abs_action for the actions), itself the
reference Diffusion Policy's robomimic_abs_action_only_normalizer_from_stat:

  observation.state  per dimension, range to [-1, 1]: offset = (min + max) / 2,
                     scale = max((max - min) / 2, 1e-3); the quaternion (3:7) is left
                     unchanged
  action, absolute   position (0:3), per dimension, range to [-1, 1] (a range below
  and absolute_6d    1e-7: offset = min, scale = 1); the rotation (axis-angle or 6D)
                     and the gripper are left unchanged
  action, delta      every dimension but the gripper, range to [-1, 1] as the
                     position above. The earlier pipeline left delta actions
                     unchanged because robosuite's are already in [-1, 1]; these are
                     in metres and radians (a few mm per step), so they are scaled.

LeRobot's MIN_MAX normalization computes (x - min) / (max - min) * 2 - 1, so each
formula is given to it as min = offset - scale, max = offset + scale (an unchanged
dimension: min = -1, max = 1). per_dim_range_stats returns the dataset's statistics
with these min/max; the policy stores them with its weights, so evaluation and
residual RL use the same normalization as training.
"""
import numpy as np

STATE_QUAT = slice(3, 7)  # observation.state: eef position (3), quaternion (4), fingers (2)
STATE_MIN_SCALE = 1e-3
ACTION_RANGE_EPS = 1e-7


def _min_max(offset, scale):
    return {"min": (offset - scale).astype(np.float32), "max": (offset + scale).astype(np.float32)}


def state_min_max(lo, hi):
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    offset = (lo + hi) / 2
    scale = np.maximum((hi - lo) / 2, STATE_MIN_SCALE)
    offset[STATE_QUAT], scale[STATE_QUAT] = 0.0, 1.0
    return _min_max(offset, scale)


def action_min_max(lo, hi, action_mode):
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    offset, scale = np.zeros_like(lo), np.ones_like(lo)
    ranged = slice(0, len(lo) - 1) if action_mode == "delta" else slice(0, 3)  # gripper: last
    r = hi[ranged] - lo[ranged]
    ignore = r < ACTION_RANGE_EPS
    offset[ranged] = np.where(ignore, lo[ranged], (lo[ranged] + hi[ranged]) / 2)
    scale[ranged] = np.where(ignore, 1.0, r / 2)
    return _min_max(offset, scale)


def per_dim_range_stats(stats, action_mode):
    """The dataset's statistics (LeRobotDatasetMetadata.stats) with the min/max of
    observation.state and action replaced by the formulation above."""
    out = dict(stats)
    out["observation.state"] = {**stats["observation.state"],
                                **state_min_max(stats["observation.state"]["min"], stats["observation.state"]["max"])}
    out["action"] = {**stats["action"], **action_min_max(stats["action"]["min"], stats["action"]["max"], action_mode)}
    return out
