"""Vanilla MLP critic baseline for the chunked-SARSA critic comparison -
see xqc/agents/critics/xqc_style/critic.py for the production-XQC-recipe
counterpart this is compared against.

Reuses the SAME xqc/networks/critic_net.py VMapCritic/Critic class - it
already supports a plain-MLP mode as a config flag (use_batch_norm=False,
use_layer_norm=False -> IdentityEmbedder+DenseBlock, no CrossQ; n_outputs=1
-> plain scalar MSE head, no categorical/C51 projection), so no new network
code is needed here, only this simpler loss function. Every other design
choice - chunked-SARSA target from the shared helper, no actor anywhere,
same flattening of the (obs_window, action_chunk) inputs - is identical to
the xqc_style variant; see that file for the more heavily annotated version
of the shared parts.

Vision support (camera_keys non-empty): the critic trains its own vision
encoder + obs_encoder from scratch, independent of any actor (see
conversation history). Since critic/vision/obs_encoder params all feed one
connected computational graph into one TD loss, they get ONE joint gradient
(jax.value_and_grad over all three params trees at once), mirroring
xqc/agents/chunking/actor.py's own update_actor_bc - not three separate
Model.apply_gradient calls. The bootstrap half (next_observations/
next_images, used only to compute the target Q(s',a')) is encoded through
target_vision/target_obs_encoder (Polyak-averaged, same tau/
target_net_update_freq as target_critic) rather than the live/training
vision weights - same reasoning as target_critic itself: an encoder that
keeps moving while it also defines its own bootstrap target reintroduces
the instability target networks exist to prevent (see conversation
history - this is a known consideration in visual RL, not specific to this
codebase)."""
import functools
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from xqc.agents.chunking.actor import apply_model_gradient
from xqc.agents.common.chunked_critic_target import bootstrap_next_actions, sarsa_td_target
from xqc.agents.common.obs_encoder import encode_observation_from_models
from xqc.agents.critics.vision_utils import rescale_images
from xqc.networks.common import InfoDict, Model, Params, norm_network
from xqc.replay_buffer.chunked_qc_buffer import ChunkedBatch


def _flatten(x: jnp.ndarray) -> jnp.ndarray:
    return x.reshape(x.shape[0], -1)


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
    """Q(s, a) for the CURRENT observation only - no TD target, no next-obs
    - this is what xqc/agents/chunking/actor.py's own update_actor needs
    (the critic's induced value at the actor's own freshly-sampled action),
    not a bootstrap. Live critic/vision/obs_encoder params (whatever was
    passed in - the caller's job to pass live, not target), train=False
    (deterministic vision, no random-crop/batch_stats mutation) - matches
    xqc/agents/xqc/actor.py's own update_actor convention exactly
    (`critic(obs, actions, training=False, ...)`).

    Returns the RAW per-ensemble-member Q values, shape (n_critics, B) -
    NOT already min-aggregated. Matches xqc/agents/xqc/actor.py's own
    update_actor exactly: that function's own bare `critic(...)` call
    likewise returns the raw ensemble output, and update_actor itself does
    `jnp.min(q_values, axis=0)` right there in its own actor_loss_fn - the
    aggregation strategy (min-over-ensemble, TD3-style pessimism) is an
    actor-loss/algorithm-level choice, not a fact about how this specific
    critic architecture decodes its own output, so it belongs at the
    caller, not baked into every compute_q variant (see conversation
    history). Any caller wanting a single aggregated scalar per sample
    (not going through update_actor) must call `.min(axis=0)` itself.

    A standalone, jit-decorated FREE function (not a method on a stateful
    learner object) so update_actor can take it as a plain, static-jit-
    compatible callable - see conversation history: xqc/agents/xqc/
    actor.py's own update_actor takes `critic` as a direct Model argument
    (no stateful Python wrapper involved at all, since xqc has only one
    critic architecture); passing a Python method bound to a stateful
    object (e.g. ChunkedPolicyLearner/ChunkedCriticSARSALearner) into a
    jitted function risks the object's own params going stale across
    calls, since the object itself isn't a JAX pytree. This function
    sidesteps that entirely by taking every piece it needs as an explicit
    pytree argument, exactly like xqc's own critic does - the per-variant
    swappability (the one thing xqc's reference never needed) is handled
    by the CALLER choosing which variant's own compute_q to pass in, not
    by this function knowing about other variants.

    `action` must already be normalized (the same units the caller's own
    action normalization produces) - the caller's job."""
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
) -> Tuple[jnp.ndarray, InfoDict]:
    # train=False (eval_critic) disables random-crop AND vision batch_stats
    # mutation (see encode_condition's own `if random_crop and train`/`if
    # train: ... mutable=["batch_stats"] else: ...` branches) - a held-out
    # validation metric must be deterministic, not contaminated by crop
    # noise, matching xqc/agents/chunking/actor.py's own val_loss convention
    # (always train=False there). Only the CURRENT half's train-ness is
    # configurable here - the next/bootstrap half below always uses
    # train=False regardless, since it goes through target_vision either way.
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
    obs = _flatten(obs)
    next_obs = _flatten(next_obs)
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

    # Target critic, evaluated on the REAL next chunk (SARSA) - no actor.
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
    # batch.valid (B,) - 0 wherever batch.rewards itself is contaminated by a
    # reward from PAST the episode's own terminal (the chunk window ran
    # past the boundary, not just reached it - see
    # xqc/replay_buffer/chunked_qc_buffer.py's own ChunkedBatch docstring
    # and Dataset.sample_sequence). Zeroing here rather than excluding from
    # the mean's denominator matches qc's own critic_loss = ((q -
    # target_q)**2 * batch['valid']).mean() exactly (see conversation
    # history) - broadcasts fine against the (n_critics, B) squared-error
    # array below, same as batch.masks/batch.discount already do inside
    # sarsa_td_target above.
    critic_loss = (0.5 * (current_q_values - target_q) ** 2 * batch.valid).mean(1).sum()

    # Mean-over-ensemble prediction, used only for the diagnostics below
    # (the loss itself trains every ensemble member independently above).
    current_q_mean = current_q_values.mean(axis=0)
    td_error = jnp.abs(current_q_mean - target_q)

    return critic_loss, {
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
) -> Tuple[Model, Dict[str, Model], Model, InfoDict]:
    """actor_network_model, when given, switches the bootstrap from
    chunked-SARSA (default - batch.next_actions, the real recorded next
    action) to standard off-policy TD-target bootstrapping (the actor's
    own predicted next action) - see mse_critic_loss_fn's own docstring
    for the full rationale."""
    critic_loss_fn = partial(
        mse_critic_loss_fn,
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


@functools.partial(jax.jit, static_argnames=("camera_keys", "random_crop", "crop_shape"))
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
) -> InfoDict:
    """No-gradient pass over a held-out batch (chunked_qc_buffer.py's Dataset.
    get_val_batch) - same loss/target math as training, just never calls
    apply_gradient, so it tells you whether the critic generalizes to
    transitions it was never fit to, not just whether training loss went
    down."""
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    _, info = mse_critic_loss_fn(
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
