"""XQC-style critic for the chunked-SARSA diagnostic: the UNMODIFIED
production critic architecture and loss math (xqc/agents/xqc/critic.py's
VMapCritic + categorical_td_loss, same CrossQ/BatchNorm/weight-norm/
categorical-head options as conf/agent/xqc.yaml + conf/agent/base_agent.yaml
- see xqc/agents/chunking/chunked_learner_sarsa.py for where those defaults
are set) - reproducing
the supervisor's existing XQC critic setup exactly, per conversation
history. `categorical_td_loss` (the C51 projection math) is imported
UNCHANGED from xqc/agents/xqc/critic.py rather than re-derived here, since
it's pure per-sample math with no actor/chunking dependency at all.

Only two things differ from xqc/agents/xqc/critic.py's own
categorical_critic_loss_fn/mse_critic_loss_fn:
  1. The bootstrap action is the REAL next h-action chunk straight from the
     dataset (batch.next_actions) - never resampled from an actor (there is
     no actor anywhere in this module), so there's no entropy/KL term
     either (actor_entropy is always exactly zero here).
  2. observations/actions are chunked - (B, To, obs_dim) / (B, h, act_dim) -
     flattened to (B, To*obs_dim) / (B, h*act_dim) right before hitting the
     critic (Critic.__call__ just concatenates+MLPs, so this needs no
     architecture change - see conversation history).

Vision support (camera_keys non-empty): see xqc/agents/critics/mlp/critic.py's
own module docstring for the full design (joint gradient across critic+
vision+obs_encoder, target_vision/target_obs_encoder Polyak-averaged same
as target_critic) - identical here, just applied to both loss variants.
"""
import functools
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from xqc.agents.chunking.actor import apply_model_gradient
from xqc.agents.common.chunked_critic_target import (
    bootstrap_next_actions,
    entropy_from_log_probs,
    fraction_at_boundary,
    sarsa_td_target,
)
from xqc.agents.common.obs_encoder import encode_observation_from_models
from xqc.agents.critics.vision_utils import rescale_images
from xqc.agents.xqc.critic import categorical_td_loss
from xqc.networks.common import InfoDict, Model, Params, norm_network
from xqc.replay_buffer.chunked_qc_buffer import ChunkedBatch


def _flatten(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape(x.shape[0], -1)


def _encode_current_and_next(
    vision_models, vision_params, obs_encoder_model, obs_encoder_params,
    target_vision_models, target_obs_encoder_model,
    obs_offset, obs_scale, camera_keys, random_crop, crop_shape, crop_key, train, batch,
):
    """Shared by categorical_critic_loss_fn/mse_critic_loss_fn below - the
    current half (differentiated vision/obs_encoder params) and the next/
    bootstrap half (target_vision/target_obs_encoder, never differentiated,
    always train=False) - see mlp/critic.py's own module docstring for why
    the bootstrap half specifically must not use the live/training vision
    weights."""
    obs, new_vision_batch_stats = encode_observation_from_models(
        vision_models, vision_params, obs_encoder_model, obs_encoder_params,
        batch.observations, rescale_images(batch.images), obs_offset, obs_scale, camera_keys,
        train=train, random_crop=random_crop, crop_shape=crop_shape, crop_key=crop_key,
    )
    target_vision_params = {cam: m.params for cam, m in target_vision_models.items()}
    next_obs, _ = encode_observation_from_models(
        target_vision_models, target_vision_params, target_obs_encoder_model, target_obs_encoder_model.params,
        batch.next_observations, rescale_images(batch.next_images), obs_offset, obs_scale, camera_keys,
        train=False, random_crop=False, crop_shape=crop_shape, crop_key=None,
    )
    return _flatten(obs), _flatten(next_obs), new_vision_batch_stats


@functools.partial(jax.jit, static_argnames=("camera_keys",))
def compute_q(
    critic_model: Model,
    vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    obs_window: jnp.ndarray,
    images: Dict[str, jnp.ndarray],
    action: jnp.ndarray,
) -> jnp.ndarray:
    """Q(s, a) for the CURRENT observation only - see xqc/agents/critics/
    mlp/critic.py's own compute_q for the full rationale (identical here,
    a plain duplicate - each critic variant's critic.py stays
    self-contained). critic_model's own __call__ always returns a decoded
    scalar Q as its first output regardless of critic_loss mode
    ("categorical" or "mse" - see categorical_critic_loss_fn's own
    "Decoded scalar Q" comment), so this needs no critic_loss-specific
    branching at all.

    Returns the RAW per-ensemble-member Q values, shape (n_critics, B) -
    NOT already min-aggregated. See mlp/critic.py's own compute_q
    docstring for the full rationale (matches xqc/agents/xqc/actor.py's
    own update_actor, which does `jnp.min(q_values, axis=0)` itself right
    after its own bare `critic(...)` call)."""
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    obs, _ = encode_observation_from_models(
        vision_models, vision_params, obs_encoder_model, obs_encoder_model.params,
        obs_window, rescale_images(images), obs_offset, obs_scale, camera_keys,
        train=False, random_crop=False, crop_shape=(0, 0), crop_key=None,
    )
    obs_flat = _flatten(obs)
    action_flat = _flatten(action)
    (catted_q, _), _ = critic_model(obs_flat, action_flat, training=False, mutable=["batch_stats"])
    return catted_q


def categorical_critic_loss_fn(
    critic_params: Params,
    vision_params: Dict[str, Params],
    obs_encoder_params: Params,
    critic_batch_stats: Params,
    critic: Model,
    target_critic: Model,
    vision_models: Dict[str, Model],
    target_vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    target_obs_encoder_model: Model,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    random_crop: bool,
    crop_shape: Tuple[int, int],
    crop_key: Optional[jnp.ndarray],
    train: bool,
    batch: ChunkedBatch,
    actor_network_model: Optional[Model] = None,
    actor_vision_models: Optional[Dict[str, Model]] = None,
    actor_obs_encoder_model: Optional[Model] = None,
    actor_obs_offset: Optional[jnp.ndarray] = None,
    actor_obs_scale: Optional[jnp.ndarray] = None,
    actor_camera_keys: Sequence[str] = (),
    Ta: int = 0,
    execution_horizon: int = 0,
    act_dim: int = 0,
    actor_rng: Optional[jnp.ndarray] = None,
    offline_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, InfoDict]:
    obs, next_obs, new_vision_batch_stats = _encode_current_and_next(
        vision_models, vision_params, obs_encoder_model, obs_encoder_params,
        target_vision_models, target_obs_encoder_model,
        obs_offset, obs_scale, camera_keys, random_crop, crop_shape, crop_key, train, batch,
    )
    actions = _flatten(batch.actions)
    next_actions = bootstrap_next_actions(
        batch, actor_network_model, actor_vision_models, actor_obs_encoder_model,
        actor_obs_offset, actor_obs_scale, actor_camera_keys, Ta, execution_horizon, act_dim, actor_rng,
        flatten=True,
    )

    concat_obs = jnp.concatenate((obs, next_obs), axis=0)
    concat_actions = jnp.concatenate((actions, next_actions), axis=0)

    # Target critic, evaluated on the REAL next chunk (SARSA) by default, or
    # on the actor's own predicted next action when actor_network_model is
    # given - see xqc/agents/critics/vision_utils.py's own bootstrap_next_actions
    # docstring (matches xqc/agents/xqc/critic.py's own version, which always samples from the
    # actor - this variant makes that optional instead).
    (catted_qs, target_critic_infos), _ = critic.apply(
        target_critic.params,
        concat_obs,
        concat_actions,
        batch_stats=target_critic.batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )
    _, next_q_values = jnp.split(catted_qs, 2, axis=1)
    _, target_log_probs = jnp.split(target_critic_infos["log_probs"], 2, axis=1)

    # Min-over-ensemble target selection, identical to the production critic.
    min_indices = next_q_values.argmin(axis=0)
    target_log_probs = jax.vmap(lambda x, i: x[i], in_axes=[1, 0])(
        target_log_probs, min_indices
    )
    target_log_probs = jnp.repeat(target_log_probs[None], repeats=2, axis=0)

    (catted_current_qs, critic_infos), state_updates = critic.apply(
        critic_params,
        concat_obs,
        concat_actions,
        batch_stats=critic_batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )
    pred_log_probs, _ = jnp.split(critic_infos["log_probs"], 2, axis=1)
    # Decoded scalar Q (mean of the categorical distribution) for the
    # CURRENT half of the batch - logged as "q_values" purely for parity
    # with mse_critic_loss_fn's own key, not used in the loss itself (the
    # categorical loss trains on the full projected distribution above).
    current_q_values, _ = jnp.split(catted_current_qs, 2, axis=1)

    # No actor -> no entropy/KL term, ever (this is the one substantive
    # difference from the production loss's kl_regularization branch).
    actor_entropy = jnp.zeros_like(batch.rewards)

    loss, loss_info = jax.vmap(
        partial(
            categorical_td_loss,
            reward=batch.rewards,
            done=batch.masks,
            actor_entropy=actor_entropy,
            gamma=batch.discount,
            num_bins=critic.apply_fn.n_outputs,
            max_v=critic.apply_fn.max_v,
            min_v=critic.apply_fn.min_v,
            valid=batch.valid,
        ),
        in_axes=0,
    )(
        pred_log_probs=pred_log_probs,
        target_log_probs=target_log_probs,
    )

    loss = loss.sum()

    # Diagnostic-only scalar target (the C51 loss above trains on the full
    # projected distribution, not this) - lets "target_q" be logged in the
    # same real-value units across every critic variant, for a comparison
    # that isn't just eyeballing differently-scaled loss curves.
    next_q_min = jnp.min(next_q_values, axis=0)
    target_q = sarsa_td_target(batch.rewards, batch.masks, batch.discount, next_q_min)

    info = {
        "critic_loss": loss,
        "q_values": current_q_values.mean(),
        "q_min": current_q_values.min(),
        "q_max": current_q_values.max(),
        "target_q": target_q.mean(),
        "target_q_min": target_q.min(),
        "target_q_max": target_q.max(),
        "td_error_mean": jnp.abs(current_q_values.mean(axis=0) - target_q).mean(),
        "td_error_max": jnp.abs(current_q_values.mean(axis=0) - target_q).max(),
        # Distributional-head-only health checks - a collapsed or
        # boundary-saturated distribution can hide behind a fine-looking
        # decoded Q mean (see chunked_critic_target.py's own docstring).
        "q_entropy": entropy_from_log_probs(pred_log_probs).mean(),
        "q_frac_at_boundary": fraction_at_boundary(
            current_q_values, critic.apply_fn.min_v, critic.apply_fn.max_v
        ),
        "r": batch.rewards.mean(),
        "new_batch_stats": state_updates.get("batch_stats", None),
        "critic_activations": state_updates.get("activations", None),
        "new_vision_batch_stats": new_vision_batch_stats,
        **jax.tree.map(partial(jnp.mean, axis=0), loss_info),
    }
    # q_max_offline/q_max_online (explicit request - see conversation
    # history): current_q_values is (n_critics, B), one column per CURRENT-
    # half batch row (matches offline_mask's own (B,) shape 1:1, same
    # convention as actor.py's own offline_prior_mse). jnp.where(..., -inf)
    # masks OUT the other group before max - the standard trick for a
    # masked max inside a jitted function (Python-level branching on
    # offline_mask's presence is fine here since None-vs-array is a
    # structural/pytree difference resolved at trace time, not a runtime
    # value - same reasoning as offline_prior_mse's own guard). Guards the
    # degenerate all-one-group-empty case (e.g. offline_data_ratio so small
    # a whole batch could in principle be one group) with NaN rather than
    # a bogus -inf leaking into the logged value.
    if offline_mask is not None:
        is_offline = offline_mask[None, :] > 0.5
        offline_present = offline_mask.sum() > 0.5
        online_present = (1.0 - offline_mask).sum() > 0.5
        info["q_max_offline"] = jnp.where(
            offline_present, jnp.max(jnp.where(is_offline, current_q_values, -jnp.inf)), jnp.nan
        )
        info["q_max_online"] = jnp.where(
            online_present, jnp.max(jnp.where(~is_offline, current_q_values, -jnp.inf)), jnp.nan
        )
    return loss, info


def mse_critic_loss_fn(
    critic_params: Params,
    vision_params: Dict[str, Params],
    obs_encoder_params: Params,
    critic_batch_stats: Params,
    critic: Model,
    target_critic: Model,
    vision_models: Dict[str, Model],
    target_vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    target_obs_encoder_model: Model,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    random_crop: bool,
    crop_shape: Tuple[int, int],
    crop_key: Optional[jnp.ndarray],
    train: bool,
    batch: ChunkedBatch,
    actor_network_model: Optional[Model] = None,
    actor_vision_models: Optional[Dict[str, Model]] = None,
    actor_obs_encoder_model: Optional[Model] = None,
    actor_obs_offset: Optional[jnp.ndarray] = None,
    actor_obs_scale: Optional[jnp.ndarray] = None,
    actor_camera_keys: Sequence[str] = (),
    Ta: int = 0,
    execution_horizon: int = 0,
    act_dim: int = 0,
    actor_rng: Optional[jnp.ndarray] = None,
    offline_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, InfoDict]:
    obs, next_obs, new_vision_batch_stats = _encode_current_and_next(
        vision_models, vision_params, obs_encoder_model, obs_encoder_params,
        target_vision_models, target_obs_encoder_model,
        obs_offset, obs_scale, camera_keys, random_crop, crop_shape, crop_key, train, batch,
    )
    actions = _flatten(batch.actions)
    next_actions = bootstrap_next_actions(
        batch, actor_network_model, actor_vision_models, actor_obs_encoder_model,
        actor_obs_offset, actor_obs_scale, actor_camera_keys, Ta, execution_horizon, act_dim, actor_rng,
        flatten=True,
    )

    concat_obs = jnp.concatenate((obs, next_obs), axis=0)
    concat_actions = jnp.concatenate((actions, next_actions), axis=0)

    (catted_q, _), state_updates = critic.apply(
        critic_params,
        concat_obs,
        concat_actions,
        batch_stats=critic_batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )
    current_q_values, _ = jnp.split(catted_q, 2, axis=1)

    (catted_q_target, _), _ = critic.apply(
        target_critic.params,
        concat_obs,
        concat_actions,
        batch_stats=target_critic.batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )
    _, next_q_values = jnp.split(catted_q_target, 2, axis=1)
    next_q_values = jnp.min(next_q_values, axis=0)

    target_q = sarsa_td_target(batch.rewards, batch.masks, batch.discount, next_q_values)
    target_q = jax.lax.stop_gradient(target_q)
    # batch.valid weighting - see mlp/critic.py's own mse_critic_loss_fn
    # comment for the full rationale (same formula, same reasoning).
    critic_loss = (0.5 * (current_q_values - target_q) ** 2 * batch.valid).mean(1).sum()

    td_error = jnp.abs(current_q_values.mean(axis=0) - target_q)

    info = {
        "critic_loss": critic_loss,
        "q_values": current_q_values.mean(),
        "q_min": current_q_values.min(),
        "q_max": current_q_values.max(),
        "target_q": target_q.mean(),
        "target_q_min": target_q.min(),
        "target_q_max": target_q.max(),
        "td_error_mean": td_error.mean(),
        "td_error_max": td_error.max(),
        "r": batch.rewards.mean(),
        "new_batch_stats": state_updates.get("batch_stats", None),
        "critic_activations": state_updates.get("activations", None),
        "new_vision_batch_stats": new_vision_batch_stats,
    }
    # q_max_offline/q_max_online - see categorical_critic_loss_fn's own
    # identical block for the full rationale.
    if offline_mask is not None:
        is_offline = offline_mask[None, :] > 0.5
        offline_present = offline_mask.sum() > 0.5
        online_present = (1.0 - offline_mask).sum() > 0.5
        info["q_max_offline"] = jnp.where(
            offline_present, jnp.max(jnp.where(is_offline, current_q_values, -jnp.inf)), jnp.nan
        )
        info["q_max_online"] = jnp.where(
            online_present, jnp.max(jnp.where(~is_offline, current_q_values, -jnp.inf)), jnp.nan
        )
    return critic_loss, info


def update_critic(
    critic: Model,
    target_critic: Model,
    vision_models: Dict[str, Model],
    target_vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    target_obs_encoder_model: Model,
    batch: ChunkedBatch,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    random_crop: bool,
    crop_shape: Tuple[int, int],
    crop_key: Optional[jnp.ndarray],
    use_weight_norm: bool,
    critic_loss: str,
    actor_network_model: Optional[Model] = None,
    actor_vision_models: Optional[Dict[str, Model]] = None,
    actor_obs_encoder_model: Optional[Model] = None,
    actor_obs_offset: Optional[jnp.ndarray] = None,
    actor_obs_scale: Optional[jnp.ndarray] = None,
    actor_camera_keys: Sequence[str] = (),
    Ta: int = 0,
    execution_horizon: int = 0,
    act_dim: int = 0,
    actor_rng: Optional[jnp.ndarray] = None,
    offline_mask: Optional[jnp.ndarray] = None,
) -> Tuple[Model, Dict[str, Model], Model, InfoDict]:
    """actor_network_model, when given, switches the bootstrap from
    chunked-SARSA (default - batch.next_actions, the real recorded next
    action) to standard off-policy TD-target bootstrapping (the actor's
    own predicted next action) - see xqc/agents/critics/vision_utils.py's
    own bootstrap_next_actions docstring for the full rationale.

    offline_mask (explicit request - see conversation history): per-sample
    (B,) array, 1.0 for a row drawn from the offline buffer / 0.0 for
    online - same convention and same array as actor.py's own
    update_actor/offline_prior_mse (built once by train_parallel_chunking.py
    at its own online/offline concat site, threaded through unchanged).
    Passed straight through to categorical_critic_loss_fn/mse_critic_loss_fn,
    which use it to additionally report q_max split by buffer origin
    (q_max_offline/q_max_online) - purely a diagnostic, never affects the
    loss itself. None (default) omits those two keys entirely."""
    critic_loss_fn_cls = {"categorical": categorical_critic_loss_fn, "mse": mse_critic_loss_fn}[critic_loss]
    critic_loss_fn = partial(
        critic_loss_fn_cls,
        critic_batch_stats=critic.batch_stats,
        critic=critic,
        target_critic=target_critic,
        vision_models=vision_models,
        target_vision_models=target_vision_models,
        obs_encoder_model=obs_encoder_model,
        target_obs_encoder_model=target_obs_encoder_model,
        obs_offset=obs_offset,
        obs_scale=obs_scale,
        camera_keys=camera_keys,
        random_crop=random_crop,
        crop_shape=crop_shape,
        crop_key=crop_key,
        train=True,
        batch=batch,
        offline_mask=offline_mask,
        actor_network_model=actor_network_model,
        actor_vision_models=actor_vision_models,
        actor_obs_encoder_model=actor_obs_encoder_model,
        actor_obs_offset=actor_obs_offset,
        actor_obs_scale=actor_obs_scale,
        actor_camera_keys=actor_camera_keys,
        Ta=Ta, execution_horizon=execution_horizon, act_dim=act_dim,
        actor_rng=actor_rng,
    )
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    (loss, info), (critic_grads, vision_grads, obs_encoder_grads) = jax.value_and_grad(
        critic_loss_fn, argnums=(0, 1, 2), has_aux=True
    )(critic.params, vision_params, obs_encoder_model.params)

    new_critic = apply_model_gradient(critic, critic_grads)
    new_obs_encoder = apply_model_gradient(obs_encoder_model, obs_encoder_grads)
    new_vision_batch_stats = info.pop("new_vision_batch_stats")
    new_vision_models = {
        cam: apply_model_gradient(vision_models[cam], vision_grads[cam]).replace(batch_stats=new_vision_batch_stats[cam])
        for cam in vision_models
    }
    info["critic_grads"] = critic_grads

    if use_weight_norm:
        new_critic = norm_network(model=new_critic)
    new_critic = new_critic.replace(batch_stats=info.pop("new_batch_stats"))

    return new_critic, new_vision_models, new_obs_encoder, info


@functools.partial(jax.jit, static_argnames=("critic_loss", "camera_keys", "random_crop", "crop_shape"))
def eval_critic(
    critic: Model,
    target_critic: Model,
    vision_models: Dict[str, Model],
    target_vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    target_obs_encoder_model: Model,
    batch: ChunkedBatch,
    obs_offset: jnp.ndarray,
    obs_scale: jnp.ndarray,
    camera_keys: Sequence[str],
    random_crop: bool,
    crop_shape: Tuple[int, int],
    crop_key: Optional[jnp.ndarray],
    critic_loss: str,
) -> InfoDict:
    """No-gradient pass over a held-out batch (chunked_qc_buffer.py's Dataset.
    get_val_batch) - same loss/target math as training, just never calls
    apply_gradient, so it tells you whether the critic generalizes to
    transitions it was never fit to, not just whether training loss went
    down."""
    critic_loss_fn_cls = {"categorical": categorical_critic_loss_fn, "mse": mse_critic_loss_fn}[critic_loss]
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    _, info = critic_loss_fn_cls(
        critic.params, vision_params, obs_encoder_model.params,
        critic_batch_stats=critic.batch_stats, critic=critic, target_critic=target_critic,
        vision_models=vision_models, target_vision_models=target_vision_models,
        obs_encoder_model=obs_encoder_model, target_obs_encoder_model=target_obs_encoder_model,
        obs_offset=obs_offset, obs_scale=obs_scale, camera_keys=camera_keys,
        random_crop=random_crop, crop_shape=crop_shape, crop_key=crop_key, train=False, batch=batch,
    )
    info.pop("new_batch_stats", None)
    info.pop("critic_activations", None)
    info.pop("new_vision_batch_stats", None)
    return info
