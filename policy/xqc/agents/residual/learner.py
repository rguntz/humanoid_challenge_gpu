"""ResidualAgent — frozen base policy + trainable ResidualMLP.

Exposes the same sample_actions / reset_policy / reset_policy_idx / camera_keys
interface as ChunkedPolicyLearner so that ParallelEnv.evaluate() can drive it
directly without a custom eval loop.

Also exports:
  ObsHistoryWindow  — sliding To-frame obs/image window (used by the training
                      loop in train_residual.py for the online rollout buffer).
  jit_encode        — JIT-compiled vision+obs encoder forward pass.
  jit_residual      — JIT-compiled ResidualMLP forward pass.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from xqc.agents.residual.action_scaler import ActionScaler
from xqc.agents.residual.critic import _encode_frozen
from xqc.networks.common import Model


# ---------------------------------------------------------------------------
# Sliding obs/image history window
# ---------------------------------------------------------------------------

class ObsHistoryWindow:
    """Per-env sliding window of the last To observation frames and images.

    Maintains (num_envs, To, obs_dim) obs and {cam: (num_envs, To, H, W, C)}
    images.  Used to assemble the To-frame context required by the critic and
    residual actor (the base ChunkExecutor tracks its own window internally).
    """

    def __init__(self, num_envs: int, To: int, obs_dim: int,
                 image_shape: Tuple, camera_keys: tuple):
        self.To = To
        self.camera_keys = camera_keys
        self._obs  = np.zeros((num_envs, To, obs_dim), dtype=np.float32)
        self._imgs = {
            cam: np.zeros((num_envs, To, *image_shape), dtype=np.uint8)
            for cam in camera_keys
        }

    def _fill(self, dest_obs, dest_imgs, src_obs, src_imgs, idx):
        for t in range(self.To):
            dest_obs[idx, t] = src_obs[idx]
            for cam in self.camera_keys:
                dest_imgs[cam][idx, t] = src_imgs[cam][idx]

    def reset_all(self, obs: np.ndarray, images: Dict[str, np.ndarray]):
        for t in range(self.To):
            self._obs[:, t] = obs
            for cam in self.camera_keys:
                self._imgs[cam][:, t] = images[cam]

    def reset_envs(self, env_indices, obs: np.ndarray, images: Dict[str, np.ndarray]):
        for i in env_indices:
            self._fill(self._obs, self._imgs, obs, images, i)

    def step(self, obs: np.ndarray, images: Dict[str, np.ndarray]):
        """Shift left and append new frame."""
        self._obs = np.roll(self._obs, shift=-1, axis=1)
        self._obs[:, -1] = obs
        for cam in self.camera_keys:
            self._imgs[cam] = np.roll(self._imgs[cam], shift=-1, axis=1)
            self._imgs[cam][:, -1] = images[cam]

    @property
    def obs(self) -> np.ndarray:
        return self._obs.copy()

    @property
    def images(self) -> Dict[str, np.ndarray]:
        return {cam: self._imgs[cam].copy() for cam in self.camera_keys}


# ---------------------------------------------------------------------------
# JIT-compiled inference helpers
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("camera_keys",))
def jit_encode(vision_models, obs_enc_model, obs, images,
               obs_offset, obs_scale, camera_keys):
    return _encode_frozen(vision_models, obs_enc_model, obs, images,
                          obs_offset, obs_scale, camera_keys)


@partial(jax.jit, static_argnames=("apply_fn",))
def jit_residual(params, apply_fn, obs_flat, norm_base):
    return apply_fn.apply({"params": params}, obs_flat, norm_base)


# ---------------------------------------------------------------------------
# ResidualAgent
# ---------------------------------------------------------------------------

class ResidualAgent:
    """Wraps a frozen base ChunkedPolicyLearner and a trainable ResidualMLP.

    The combined action is clamp(norm_base + norm_delta, -1, 1), unscaled back
    to raw EEF space before being returned.

    Attributes
    ----------
    camera_keys : tuple[str]
        Consumed by ParallelEnv.evaluate() to know which images to fetch.
    residual_model : Model
        Reassigned from outside after each actor gradient update so that eval
        always reflects the latest weights without recreating the agent.
    """

    def __init__(
        self,
        base_agent,
        residual_model: Model,
        action_scaler: ActionScaler,
        obs_offset: jnp.ndarray,
        obs_scale: jnp.ndarray,
        num_envs: int,
        obs_dim: int,
        img_shape: tuple,
        cfg: DictConfig,
    ):
        self.camera_keys = tuple(cfg.agent.camera_keys)
        self._base = base_agent
        self.residual_model = residual_model
        self._scaler = action_scaler
        self._obs_offset = obs_offset
        self._obs_scale = obs_scale
        self._cfg = cfg
        self._num_envs = num_envs
        self._history = ObsHistoryWindow(num_envs, base_agent.To, obs_dim, img_shape, self.camera_keys)
        self._just_reset = np.ones(num_envs, dtype=bool)

    def reset_policy(self):
        self._base.reset_policy()
        self._just_reset[:] = True

    def reset_policy_idx(self, just_finished: np.ndarray):
        self._base.reset_policy_idx(just_finished)
        self._just_reset |= just_finished

    def sample_actions(self, obs: np.ndarray, temperature: float = 0.0, images=None):
        """obs: (num_envs, obs_dim).  images: {cam: (num_envs, H, W, C)}."""
        # Advance history; fill entire window for envs that just reset.
        self._history.step(obs, images)
        if self._just_reset.any():
            self._history.reset_envs(np.where(self._just_reset)[0], obs, images)
        self._just_reset[:] = False

        raw_base = self._base.sample_actions(obs, images=images)
        norm_base = np.stack([self._scaler.scale(raw_base[i]) for i in range(self._num_envs)])

        if self._cfg.get("debug_freeze_residual", False):
            norm_delta = np.zeros_like(norm_base)
        else:
            obs_flat = np.array(jit_encode(
                self._base.vision, self._base.obs_encoder,
                jnp.asarray(self._history.obs),
                {cam: jnp.asarray(self._history.images[cam]) for cam in self.camera_keys},
                self._obs_offset, self._obs_scale, self.camera_keys,
            ))
            norm_delta = np.array(jit_residual(
                self.residual_model.params, self.residual_model.apply_fn,
                jnp.asarray(obs_flat), jnp.asarray(norm_base),
            ))

        norm_combined = np.clip(norm_base + norm_delta, -1.0, 1.0)
        return np.stack([self._scaler.unscale(norm_combined[i]) for i in range(self._num_envs)])
