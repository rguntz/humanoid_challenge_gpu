import numpy as np
from typing import NamedTuple
import jax

class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray
    discount: np.ndarray

    @staticmethod
    def concat(b1: "Batch", b2: "Batch") -> "Batch":
        return jax.tree_util.tree_map(
            lambda x, y: np.concatenate((x, y), axis=2), b1, b2 # (N_envs, N_batches, B, D)
        )