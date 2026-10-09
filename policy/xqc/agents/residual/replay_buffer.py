"""Step-level replay buffer for residual RL.

Stores flat (s, a, r, s') transitions — no chunking, no To-window assembly
needed at insertion time (caller inserts pre-assembled To-window obs arrays).

Each transition holds:
  observations        (To, obs_dim)     low-dim obs history ending at t
  images              (To, H, W, C)     image history (uint8, saves memory)
  norm_base_actions   (act_dim,)        normalised base-policy action at t
  norm_combined_acts  (act_dim,)        normalised combined action executed at t
  rewards             ()
  masks               ()                1-done (0 if terminal within n_step window)
  next_observations   (To, obs_dim)
  next_images         (To, H, W, C)
  next_norm_base_acts (act_dim,)        normalised base action at t+n (for TD)

N-step returns: when n_step>1, each stored reward is the n-step discounted sum
  r_t + γ*r_{t+1} + ... + γ^{k-1}*r_{t+k-1}  (k ≤ n_step, stops at episode end),
next_* fields point to the state at t+k, and mask=0 if the episode terminated
within [t, t+k).  The effective per-step discount in the TD target is γ^n_step
(since mask=0 kills the bootstrap when k<n anyway).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ResidualBatch:
    observations: jnp.ndarray          # (B, To, obs_dim)
    images: Dict[str, jnp.ndarray]     # cam → (B, To, H, W, C) uint8
    norm_base_actions: jnp.ndarray     # (B, act_dim)
    norm_combined_actions: jnp.ndarray # (B, act_dim)
    rewards: jnp.ndarray               # (B,)  n-step discounted return
    masks: jnp.ndarray                 # (B,)  1-done (0 if terminal in window)
    next_observations: jnp.ndarray     # (B, To, obs_dim)  at t+n
    next_images: Dict[str, jnp.ndarray]
    next_norm_base_actions: jnp.ndarray # (B, act_dim)
    discount: float = 0.99              # γ^n_step (TD bootstrap discount)


jax.tree_util.register_pytree_node(
    ResidualBatch,
    lambda b: (
        (b.observations, b.images, b.norm_base_actions, b.norm_combined_actions,
         b.rewards, b.masks, b.next_observations, b.next_images,
         b.next_norm_base_actions, b.discount),
        None,
    ),
    lambda _, leaves: ResidualBatch(*leaves),
)


class ResidualReplayBuffer:
    """Ring buffer for online transitions with n-step return accumulation."""

    def __init__(
        self,
        capacity: int,
        To: int,
        obs_dim: int,
        act_dim: int,
        image_shape: tuple,           # (H, W, C)
        camera_keys: tuple,
        discount: float = 0.99,
        n_step: int = 1,
    ):
        self.capacity = capacity
        self.To = To
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.camera_keys = camera_keys
        self.discount = discount
        self.n_step = n_step
        self._ptr = 0
        self._size = 0

        self._obs         = np.zeros((capacity, To, obs_dim), dtype=np.float32)
        self._next_obs    = np.zeros((capacity, To, obs_dim), dtype=np.float32)
        self._images      = {k: np.zeros((capacity, To, *image_shape), dtype=np.uint8) for k in camera_keys}
        self._next_images = {k: np.zeros((capacity, To, *image_shape), dtype=np.uint8) for k in camera_keys}
        self._norm_base      = np.zeros((capacity, act_dim), dtype=np.float32)
        self._norm_combined  = np.zeros((capacity, act_dim), dtype=np.float32)
        self._next_norm_base = np.zeros((capacity, act_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._masks   = np.zeros(capacity, dtype=np.float32)

        # Per-env staging deques for n-step accumulation (keyed by env_id)
        self._staging: dict[int, collections.deque] = {}

    # ------------------------------------------------------------------
    # Public add interface
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,               # (To, obs_dim)
        images: Dict[str, np.ndarray], # cam → (To, H, W, C) uint8
        norm_base: np.ndarray,         # (act_dim,)
        norm_combined: np.ndarray,     # (act_dim,)
        reward: float,
        next_obs: np.ndarray,
        next_images: Dict[str, np.ndarray],
        next_norm_base: np.ndarray,
        done: bool,
        env_id: int = 0,
    ):
        staging = self._staging.setdefault(env_id, collections.deque())
        staging.append({
            "obs": obs, "images": images,
            "norm_base": norm_base, "norm_combined": norm_combined,
            "reward": reward,
            "next_obs": next_obs, "next_images": next_images,
            "next_norm_base": next_norm_base,
            "done": done,
        })

        if done:
            # Episode ended — flush all remaining staged entries
            while staging:
                self._flush_one(staging)
        elif len(staging) >= self.n_step:
            # Staging is full — flush the oldest entry
            self._flush_one(staging)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_one(self, staging: collections.deque):
        """Compute n-step return for the oldest staged entry and write to ring buffer."""
        base = staging[0]

        # Accumulate discounted return; stop at first done in staging
        r_nstep = 0.0
        done_idx = None
        for k in range(len(staging)):
            entry = staging[k]
            r_nstep += (self.discount ** k) * entry["reward"]
            if entry["done"]:
                done_idx = k
                break

        last_idx = done_idx if done_idx is not None else len(staging) - 1
        last = staging[last_idx]
        mask = 0.0 if done_idx is not None else 1.0

        self._write(
            obs=base["obs"], images=base["images"],
            norm_base=base["norm_base"], norm_combined=base["norm_combined"],
            reward=r_nstep,
            next_obs=last["next_obs"], next_images=last["next_images"],
            next_norm_base=last["next_norm_base"],
            mask=mask,
        )
        staging.popleft()

    def _write(
        self,
        obs: np.ndarray,
        images: Dict[str, np.ndarray],
        norm_base: np.ndarray,
        norm_combined: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_images: Dict[str, np.ndarray],
        next_norm_base: np.ndarray,
        mask: float,
    ):
        i = self._ptr
        self._obs[i]          = obs
        self._next_obs[i]     = next_obs
        for k in self.camera_keys:
            self._images[k][i]      = images[k]
            self._next_images[k][i] = next_images[k]
        self._norm_base[i]       = norm_base
        self._norm_combined[i]   = norm_combined
        self._next_norm_base[i]  = next_norm_base
        self._rewards[i]         = reward
        self._masks[i]           = mask

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, rng: np.random.Generator) -> ResidualBatch:
        idx = rng.integers(0, self._size, size=batch_size)
        return self._make_batch(idx)

    def _make_batch(self, idx: np.ndarray) -> ResidualBatch:
        return ResidualBatch(
            observations=jnp.asarray(self._obs[idx]),
            images={k: jnp.asarray(self._images[k][idx]) for k in self.camera_keys},
            norm_base_actions=jnp.asarray(self._norm_base[idx]),
            norm_combined_actions=jnp.asarray(self._norm_combined[idx]),
            rewards=jnp.asarray(self._rewards[idx]),
            masks=jnp.asarray(self._masks[idx]),
            next_observations=jnp.asarray(self._next_obs[idx]),
            next_images={k: jnp.asarray(self._next_images[k][idx]) for k in self.camera_keys},
            next_norm_base_actions=jnp.asarray(self._next_norm_base[idx]),
            discount=self.discount ** self.n_step,
        )

    def __len__(self) -> int:
        return self._size


class OfflineResidualBuffer(ResidualReplayBuffer):
    """Fixed-size buffer filled once from demonstrations; never written to
    after populate().  Shares the same add() / sample() interface."""

    def __init__(self, capacity: int, **kwargs):
        super().__init__(capacity=capacity, **kwargs)

    def finalize(self):
        """Call after all add() calls to lock the filled size."""
        self.capacity = self._size  # sample only from filled entries
        self._ptr = 0               # prevent accidental further writes


def sample_mixed(
    online: ResidualReplayBuffer,
    offline: Optional[OfflineResidualBuffer],
    batch_size: int,
    offline_fraction: float,
    rng: np.random.Generator,
) -> ResidualBatch:
    """Sample a mixed batch: offline_fraction from offline, rest from online."""
    if offline is None or len(offline) == 0 or offline_fraction <= 0.0:
        return online.sample(batch_size, rng)

    n_offline = int(batch_size * offline_fraction)
    n_online  = batch_size - n_offline

    b_on  = online.sample(n_online, rng)
    b_off = offline.sample(n_offline, rng)

    def cat(a, b):
        if isinstance(a, dict):
            return {k: jnp.concatenate([a[k], b[k]], axis=0) for k in a}
        return jnp.concatenate([a, b], axis=0)

    return ResidualBatch(
        observations=cat(b_on.observations, b_off.observations),
        images=cat(b_on.images, b_off.images),
        norm_base_actions=cat(b_on.norm_base_actions, b_off.norm_base_actions),
        norm_combined_actions=cat(b_on.norm_combined_actions, b_off.norm_combined_actions),
        rewards=cat(b_on.rewards, b_off.rewards),
        masks=cat(b_on.masks, b_off.masks),
        next_observations=cat(b_on.next_observations, b_off.next_observations),
        next_images=cat(b_on.next_images, b_off.next_images),
        next_norm_base_actions=cat(b_on.next_norm_base_actions, b_off.next_norm_base_actions),
        discount=b_on.discount,
    )
