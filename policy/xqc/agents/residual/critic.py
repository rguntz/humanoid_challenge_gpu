"""Step-level critic update for residual RL.

Uses the xqc-style VMapCritic (C51 categorical loss) operating on flat
(obs_features, 7D_action) inputs instead of chunked action windows.

Vision is ALWAYS frozen (stop_gradient): the base policy's ResNet features
are reused without training a separate critic vision encoder.  Gradients
flow only through VMapCritic params.

TD target bootstrap:
    next_combined = clamp(next_norm_base + residual(next_obs, next_norm_base), -1, 1)
    Q_target = categorical_td_loss(Q_target_critic(next_obs, next_combined))
"""
from __future__ import annotations

from functools import partial
from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from xqc.agents.chunking.actor import apply_model_gradient
from xqc.agents.common.obs_encoder import encode_observation_from_models
from xqc.agents.critics.vision_utils import rescale_images
from xqc.agents.residual.network import ResidualMLP
from xqc.agents.residual.replay_buffer import ResidualBatch
from xqc.agents.xqc.critic import categorical_td_loss
from xqc.networks.common import InfoDict, Model, Params, norm_network


def _encode_frozen(
    vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    obs: jnp.ndarray,
    images: Dict[str, jnp.ndarray],
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
) -> jnp.ndarray:
    """Encode observations with frozen vision (stop_gradient on all params).
    Returns (B, To * condition_dim) flat features."""
    vision_params = {cam: jax.lax.stop_gradient(m.params) for cam, m in vision_models.items()}
    enc_obs, _ = encode_observation_from_models(
        vision_models, vision_params,
        obs_encoder_model, jax.lax.stop_gradient(obs_encoder_model.params),
        obs, rescale_images(images), obs_offset, obs_scale, camera_keys,
        train=False, random_crop=False, crop_shape=(0, 0), crop_key=None,
    )
    # enc_obs: (B, To, condition_dim) → flatten to (B, To*condition_dim)
    return enc_obs.reshape(enc_obs.shape[0], -1)


def residual_critic_loss_fn(
    critic_params: Params,
    critic_batch_stats: Params,
    critic: Model,
    target_critic: Model,
    frozen_vision_models: Dict[str, Model],
    frozen_obs_encoder_model: Model,
    residual_model: Model,
    batch: ResidualBatch,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
) -> Tuple[jnp.ndarray, InfoDict]:
    # ---- encode current and next observations (frozen vision) ----
    obs_flat = _encode_frozen(
        frozen_vision_models, frozen_obs_encoder_model,
        batch.observations, batch.images,
        obs_offset, obs_scale, camera_keys,
    )
    next_obs_flat = _encode_frozen(
        frozen_vision_models, frozen_obs_encoder_model,
        batch.next_observations, batch.next_images,
        obs_offset, obs_scale, camera_keys,
    )

    # ---- next combined action for bootstrap ----
    # Split obs_flat into vision and lowdim parts to feed ResidualMLP
    # ResidualMLP takes (vision_features, obs_flat, norm_base) separately;
    # here we pass the full flattened encoding as obs_flat and an empty
    # vision tensor since features are already fused — simpler to just
    # concatenate everything and use a single-input variant.
    # We call residual_model directly on the fused next_obs_flat by
    # treating the full encoding as "obs_flat" with empty vision dim.
    next_norm_delta = jax.lax.stop_gradient(
        residual_model.apply_fn.apply(
            {"params": residual_model.params},
            next_obs_flat,
            batch.next_norm_base_actions,
        )
    )
    next_norm_combined = jnp.clip(
        batch.next_norm_base_actions + next_norm_delta, -1.0, 1.0
    )

    # ---- forward pass through target + live critic ----
    concat_obs     = jnp.concatenate([obs_flat, next_obs_flat], axis=0)
    concat_actions = jnp.concatenate([batch.norm_combined_actions, next_norm_combined], axis=0)

    (_, target_infos), _ = critic.apply(
        target_critic.params,
        concat_obs, concat_actions,
        batch_stats=target_critic.batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, _: isinstance(mdl, nn.Dense),
        training=True,
    )
    _, next_q_values = jnp.split(target_infos["log_probs"], 2, axis=1)
    # min-over-ensemble target selection
    # next_q_values: (n_critics, B, num_bins) → pick ensemble member with min Q
    next_q_scalar = jnp.sum(
        jnp.exp(next_q_values) *
        jnp.linspace(critic.apply_fn.min_v, critic.apply_fn.max_v,
                     critic.apply_fn.n_outputs)[None, None, :],
        axis=-1,
    )  # (n_critics, B)
    min_indices = next_q_scalar.argmin(axis=0)
    target_log_probs = jax.vmap(lambda x, i: x[i], in_axes=[1, 0])(
        next_q_values, min_indices
    )
    target_log_probs = jnp.repeat(target_log_probs[None], repeats=2, axis=0)

    (_, live_infos), state_updates = critic.apply(
        critic_params,
        concat_obs, concat_actions,
        batch_stats=critic_batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, _: isinstance(mdl, nn.Dense),
        training=True,
    )
    pred_log_probs, _ = jnp.split(live_infos["log_probs"], 2, axis=1)

    # ---- C51 categorical TD loss ----
    actor_entropy = jnp.zeros_like(batch.rewards)
    loss, loss_info = jax.vmap(
        partial(
            categorical_td_loss,
            reward=batch.rewards,
            done=batch.masks,
            actor_entropy=actor_entropy,
            gamma=jnp.full_like(batch.rewards, batch.discount),
            num_bins=critic.apply_fn.n_outputs,
            max_v=critic.apply_fn.max_v,
            min_v=critic.apply_fn.min_v,
        ),
        in_axes=0,
    )(pred_log_probs=pred_log_probs, target_log_probs=target_log_probs)

    loss = loss.sum()

    # scalar Q for logging
    current_q = jnp.sum(
        jnp.exp(pred_log_probs) *
        jnp.linspace(critic.apply_fn.min_v, critic.apply_fn.max_v,
                     critic.apply_fn.n_outputs)[None, None, :],
        axis=-1,
    ).mean()
    target_q_scalar = (batch.rewards +
                       batch.discount * batch.masks *
                       next_q_scalar.min(axis=0)).mean()

    return loss, {
        "critic_loss": loss,
        "q_values": current_q,
        "target_q": target_q_scalar,
        "r": batch.rewards.mean(),
        "new_batch_stats": state_updates.get("batch_stats", None),
        **jax.tree.map(partial(jnp.mean, axis=0), loss_info),
    }


@partial(jax.jit, static_argnames=("camera_keys", "use_weight_norm"))
def update_critic(
    critic: Model,
    target_critic: Model,
    frozen_vision_models: Dict[str, Model],
    frozen_obs_encoder_model: Model,
    residual_model: Model,
    batch: ResidualBatch,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    use_weight_norm: bool,
    tau: float,
) -> Tuple[Model, Model, InfoDict]:
    """Returns (new_critic, new_target_critic, info)."""
    loss_fn = partial(
        residual_critic_loss_fn,
        critic_batch_stats=critic.batch_stats,
        critic=critic,
        target_critic=target_critic,
        frozen_vision_models=frozen_vision_models,
        frozen_obs_encoder_model=frozen_obs_encoder_model,
        residual_model=residual_model,
        batch=batch,
        obs_offset=obs_offset,
        obs_scale=obs_scale,
        camera_keys=camera_keys,
    )

    # critic.batch_stats is already bound in loss_fn via partial; differentiate
    # only through critic_params (batch_stats are running stats, not learnable).
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic.params)

    new_critic = apply_model_gradient(critic, grads)
    if use_weight_norm:
        new_critic = norm_network(new_critic)
    new_critic = new_critic.replace(batch_stats=info.pop("new_batch_stats"))

    # Polyak target update
    new_target_params = jax.tree.map(
        lambda tp, p: tau * p + (1.0 - tau) * tp,
        target_critic.params, new_critic.params,
    )
    new_target_batch_stats = jax.tree.map(
        lambda tp, p: tau * p + (1.0 - tau) * tp,
        target_critic.batch_stats, new_critic.batch_stats,
    )
    new_target_critic = target_critic.replace(
        params=new_target_params, batch_stats=new_target_batch_stats
    )

    return new_critic, new_target_critic, info


@partial(jax.jit, static_argnames=("camera_keys",))
def compute_q(
    critic: Model,
    frozen_vision_models: Dict[str, Model],
    frozen_obs_encoder_model: Model,
    obs: jnp.ndarray,
    images: Dict[str, jnp.ndarray],
    norm_combined_action: jnp.ndarray,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
) -> jnp.ndarray:
    """Returns per-ensemble-member Q values, shape (n_critics, B).
    Used by the actor update to compute the policy gradient."""
    obs_flat = _encode_frozen(
        frozen_vision_models, frozen_obs_encoder_model,
        obs, images, obs_offset, obs_scale, camera_keys,
    )
    (q_values, _), _ = critic(obs_flat, norm_combined_action,
                              training=False, mutable=["batch_stats"])
    return q_values
