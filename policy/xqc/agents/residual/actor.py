"""Step-level actor (residual network) update for residual RL.

Gradient flows ONLY through residual_model params — vision and base policy
are frozen via stop_gradient.

Loss = -min_ensemble Q(obs, clamp(norm_base + delta, -1, 1))
       + l2_reg_weight * ||delta||^2

The L2 regularisation on the raw delta (before action_scale multiplication)
encourages the residual to stay small, matching resfit's action_l2_reg_weight.
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp

from xqc.agents.chunking.actor import apply_model_gradient
from xqc.agents.common.obs_encoder import encode_observation_from_models
from xqc.agents.critics.vision_utils import rescale_images
from xqc.agents.residual.replay_buffer import ResidualBatch
from xqc.networks.common import InfoDict, Model, Params


def _encode_frozen(
    vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    obs: jnp.ndarray,
    images: Dict[str, jnp.ndarray],
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
) -> jnp.ndarray:
    vision_params = {cam: jax.lax.stop_gradient(m.params) for cam, m in vision_models.items()}
    enc_obs, _ = encode_observation_from_models(
        vision_models, vision_params,
        obs_encoder_model, jax.lax.stop_gradient(obs_encoder_model.params),
        obs, rescale_images(images), obs_offset, obs_scale, camera_keys,
        train=False, random_crop=False, crop_shape=(0, 0), crop_key=None,
    )
    return enc_obs.reshape(enc_obs.shape[0], -1)


def residual_actor_loss_fn(
    residual_params: Params,
    residual_model: Model,
    critic: Model,
    frozen_vision_models: Dict[str, Model],
    frozen_obs_encoder_model: Model,
    batch: ResidualBatch,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    l2_reg_weight: float,
) -> Tuple[jnp.ndarray, InfoDict]:
    obs_flat = _encode_frozen(
        frozen_vision_models, frozen_obs_encoder_model,
        batch.observations, batch.images,
        obs_offset, obs_scale, camera_keys,
    )

    # Residual forward pass (gradient flows through residual_params)
    norm_delta = residual_model.apply_fn.apply(
        {"params": residual_params},
        obs_flat,
        batch.norm_base_actions,
    )
    norm_combined = jnp.clip(batch.norm_base_actions + norm_delta, -1.0, 1.0)

    # Q-values from frozen critic (stop_gradient on critic params).
    # training=False: uses running batch-stats, no mutable update needed.
    q_values, _ = critic.apply(
        jax.lax.stop_gradient(critic.params),
        obs_flat, norm_combined,
        batch_stats=jax.lax.stop_gradient(critic.batch_stats),
        training=False,
    )
    # q_values: (n_critics, B) expected Q — ensemble mean for policy gradient
    q_mean = q_values.mean(axis=0)

    # L2 regularisation on the raw delta (before action_scale scaling)
    l2_penalty = jnp.mean(jnp.sum(norm_delta ** 2, axis=-1))

    actor_loss = -q_mean.mean() + l2_reg_weight * l2_penalty

    return actor_loss, {
        "actor_loss": actor_loss,
        "q_actor": q_mean.mean(),
        "residual_l2": l2_penalty,
        "residual_mean_abs": jnp.abs(norm_delta).mean(),
        "residual_max_abs": jnp.abs(norm_delta).max(),
    }


@partial(jax.jit, static_argnames=("camera_keys",))
def update_actor(
    residual_model: Model,
    critic: Model,
    frozen_vision_models: Dict[str, Model],
    frozen_obs_encoder_model: Model,
    batch: ResidualBatch,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    l2_reg_weight: float,
) -> Tuple[Model, InfoDict]:
    loss_fn = partial(
        residual_actor_loss_fn,
        residual_model=residual_model,
        critic=critic,
        frozen_vision_models=frozen_vision_models,
        frozen_obs_encoder_model=frozen_obs_encoder_model,
        batch=batch,
        obs_offset=obs_offset,
        obs_scale=obs_scale,
        camera_keys=camera_keys,
        l2_reg_weight=l2_reg_weight,
    )

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(residual_model.params)
    new_residual_model = apply_model_gradient(residual_model, grads)
    return new_residual_model, info
