"""Residual policy network: small MLP that predicts a normalised delta action
on top of a frozen base policy.

Input:  concat(frozen_vision_features, flat_obs_history, norm_base_action)
Output: delta ∈ (act_dim,), scaled by action_scale so the raw network output
        lives in [-1, 1] and the effective residual is ≤ action_scale per dim.

Critical design choices (matching resfit):
  - Last Dense layer kernel AND bias initialised to ZERO → network starts
    predicting zero residuals, so training begins from the base policy.
  - LayerNorm after each hidden layer (no BatchNorm: residual runs at
    eval-time step frequency where batch statistics are unreliable).
  - action_scale=0.1 constrains the residual magnitude.
"""
from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class ResidualMLP(nn.Module):
    hidden_dims: Sequence[int]
    act_dim: int
    action_scale: float = 0.1

    @nn.compact
    def __call__(
        self,
        obs_flat: jnp.ndarray,           # (B, To*(obs_dim+vision_dim)) fused, stop_gradient by caller
        norm_base_action: jnp.ndarray,   # (B, act_dim)
    ) -> jnp.ndarray:
        x = jnp.concatenate([obs_flat, norm_base_action], axis=-1)
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        delta = nn.Dense(
            self.act_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        return delta * self.action_scale
