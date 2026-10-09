"""Min-max action scaler for residual RL: normalises raw 7-D EEF actions to
[-1, 1] and back.  Mirrors resfit/rl_finetuning/utils/normalization.py's
ActionScaler but in numpy/JAX (no PyTorch dependency).

The action_scale parameter does double duty (same design as the original):
  - It expands the normalisation range by (1 + action_scale) on each side of
    the per-dimension midpoint, so the base policy's actions map to roughly
    [−0.91, 0.91] when action_scale=0.1, leaving ±0.1 headroom for the
    residual.
  - The residual network's last-layer output is multiplied by the same
    action_scale to constrain the delta magnitude.

Build once at startup via ActionScaler.from_hdf5(), then call:
    norm  = scaler.scale(raw_action)    # raw EEF → [−1, 1]
    raw   = scaler.unscale(norm_action) # [−1, 1] → raw EEF
"""
from __future__ import annotations

import numpy as np
import h5py


class ActionScaler:
    def __init__(
        self,
        action_min: np.ndarray,
        action_max: np.ndarray,
        action_scale: float = 0.1,
        min_range_per_dim: float = 0.1,
    ):
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)

        mid = (action_min + action_max) / 2.0
        half = (action_max - action_min) / 2.0
        half = np.maximum(half, min_range_per_dim / 2.0)
        half = half * (1.0 + action_scale)

        self._lo = (mid - half).astype(np.float32)
        self._hi = (mid + half).astype(np.float32)
        self._range = (self._hi - self._lo).astype(np.float32)
        self._range = np.maximum(self._range, 1e-8)

    # ------------------------------------------------------------------
    def scale(self, action: np.ndarray) -> np.ndarray:
        """Raw EEF action → [−1, 1]."""
        a = np.clip(action, self._lo, self._hi)
        return (2.0 * (a - self._lo) / self._range - 1.0).astype(np.float32)

    def unscale(self, norm_action: np.ndarray) -> np.ndarray:
        """[−1, 1] → raw EEF action."""
        n = np.clip(norm_action, -1.0, 1.0)
        return (self._lo + (n + 1.0) * self._range / 2.0).astype(np.float32)

    # ------------------------------------------------------------------
    @classmethod
    def from_hdf5(
        cls,
        dataset_path: str,
        action_scale: float = 0.1,
        min_range_per_dim: float = 0.1,
    ) -> ActionScaler:
        """Build from all demo actions stored in pick_place_bottle_base.hdf5."""
        with h5py.File(dataset_path, "r") as f:
            actions = np.concatenate(
                [f["data"][d]["actions"][:] for d in f["data"]], axis=0
            )
        return cls(
            action_min=actions.min(axis=0),
            action_max=actions.max(axis=0),
            action_scale=action_scale,
            min_range_per_dim=min_range_per_dim,
        )

    # ------------------------------------------------------------------
    @property
    def lo(self) -> np.ndarray:
        return self._lo

    @property
    def hi(self) -> np.ndarray:
        return self._hi
