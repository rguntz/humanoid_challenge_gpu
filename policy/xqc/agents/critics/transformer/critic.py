"""HL-Gauss chunked-SARSA loss for the transformer critic - a JAX port of
srl_il_dev's HLGaussLoss (srl_il/algo/q_chunking.py:20-44), matched exactly
except the value-support bounds/sigma: those are task-specific in the
reference (calibrated to its own adroit reward/return scale via
adroit_task.q_min/q_max/q_sigma) and have no literal equivalent to copy, so
they're re-derived here for our own reward scale instead - min_v/max_v
reuse xqc_style's own C51 bounds (-5, 5), and the default sigma matches the
~bin_width/4 ratio the reference's own task configs consistently use
(door/hammer/pen/relocate, dense and sparse variants alike - see
conversation history for the actual numbers) rather than an arbitrary
guess: sigma = (max_v - min_v) / (4 * num_bins).

Unlike xqc_style/mlp, the target and the trained critic get two genuinely
separate forward passes here (not one shared concatenated batch): the
target critic is evaluated deterministically (training=False, dropout off
- a regression target should not be stochastic), while the trained critic
uses its normal training-mode dropout and needs its own dropout RNG.

Vision support (camera_keys non-empty): see xqc/agents/critics/mlp/critic.py's
own module docstring for the full design (joint gradient across critic+
vision+obs_encoder, target_vision/target_obs_encoder Polyak-averaged same
as target_critic). Unlike mlp/xqc_style, encode_condition's own output
(B, To, condition_dim) is fed to the critic network UNFLATTENED - this
critic's own VMapTransformerCritic already consumes the (B, To, obs_dim)
sequence shape directly (it's a transformer, not an MLP - see this file's
own module docstring), so no _flatten() step is needed or wanted here."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

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
from xqc.networks.common import InfoDict, Model, Params, PRNGKey
from xqc.replay_buffer.chunked_qc_buffer import ChunkedBatch


def hlgauss_transform_to_probs(
    target: jnp.ndarray, support: jnp.ndarray, sigma: float
) -> jnp.ndarray:
    """target: (B,). support: (num_bins+1,) bin EDGES. Returns (B, num_bins)
    soft bin probabilities - matches HLGaussLoss.transform_to_probs exactly
    (Gaussian-CDF difference across each bin, normalized by the CDF mass
    actually inside [min_v, max_v])."""
    cdf_evals = jax.scipy.special.erf(
        (support[None, :] - target[:, None]) / (jnp.sqrt(2.0) * sigma)
    )
    z = cdf_evals[:, -1] - cdf_evals[:, 0]
    bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]
    return bin_probs / (z[:, None] + 1e-6)


def hlgauss_transform_from_probs(probs: jnp.ndarray, support: jnp.ndarray) -> jnp.ndarray:
    """probs: (..., num_bins). support: (num_bins+1,) edges. Returns (...,)
    scalar expectation over bin centers - matches
    HLGaussLoss.transform_from_probs exactly."""
    centers = (support[:-1] + support[1:]) / 2.0
    return jnp.sum(probs * centers, axis=-1)


@partial(jax.jit, static_argnames=("camera_keys",))
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
    support: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Q(s, a) for the CURRENT observation only - see xqc/agents/critics/
    mlp/critic.py's own compute_q for the full rationale. Two differences
    from mlp/xqc_style's own version, both from this critic's own
    architecture (see this file's module docstring): (1) no _flatten() -
    VMapTransformerCritic consumes the (B, To, condition_dim) sequence
    shape directly; (2) the raw network output is logits over value bins,
    not an already-decoded scalar, so this decodes via
    hlgauss_transform_from_probs(softmax(logits), support) - matching
    hlgauss_critic_loss_fn's own q_values computation exactly. `sigma` is
    accepted for signature symmetry with the loss function but unused here
    (only needed to build a soft target distribution during training, not
    to decode an existing one). training=False, so `rngs={"dropout":
    ...}` is passed for call-signature consistency but never actually
    drawn from (see hlgauss_critic_loss_fn's own training=False branch) -
    a fixed dummy key is fine.

    `support`/`sigma` are placed LAST, after the same 9-argument prefix
    mlp/xqc_style's own compute_q uses (critic_model...action) - this
    matters: any caller's own compute_q_fn (e.g. chunked_learner.py's own
    _build_critic_bound_fns, or chunked_learner_sarsa.py's own equivalent)
    binds them via functools.partial(compute_q, support=..., sigma=...)
    once at construction time, and partial's bound keyword
    arguments must not collide with positions filled by update_actor's own
    fixed 9 positional arguments - putting them anywhere in the shared
    prefix would break that. A standalone, jit-decorated FREE function
    (not a method) so update_actor can take a uniform, static-jit-
    compatible callable regardless of critic variant - see
    mlp/critic.py's own compute_q docstring for the full rationale.

    Returns the RAW per-ensemble-member Q values, shape (n_critics, B) -
    NOT already min-aggregated. See mlp/critic.py's own compute_q
    docstring for the full rationale (matches xqc/agents/xqc/actor.py's
    own update_actor, which does `jnp.min(q_values, axis=0)` itself right
    after its own bare `critic(...)` call)."""
    del sigma  # unused - see docstring
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    obs, _ = encode_observation_from_models(
        vision_models, vision_params, obs_encoder_model, obs_encoder_model.params,
        obs_window, rescale_images(images), obs_offset, obs_scale, camera_keys,
        train=False, random_crop=False, crop_shape=(0, 0), crop_key=None,
    )
    logits = critic_model(obs, action, training=False, rngs={"dropout": jax.random.PRNGKey(0)})
    q_per_critic = hlgauss_transform_from_probs(jax.nn.softmax(logits, axis=-1), support)
    return q_per_critic


def hlgauss_critic_loss_fn(
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
    batch: ChunkedBatch,
    support: jnp.ndarray,
    sigma: float,
    dropout_key: PRNGKey,
    training: bool = True,
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
    # `training` gates both the CURRENT half's vision encode (random-crop/
    # batch_stats mutation off at eval, same reasoning as mlp/xqc_style's
    # own train=False) and the critic's own dropout - one flag serves both,
    # matching this file's own pre-existing convention (a real gradient
    # step vs. eval_critic's no-grad, deterministic held-out pass).
    obs, new_vision_batch_stats = encode_observation_from_models(
        vision_models, vision_params, obs_encoder_model, obs_encoder_params,
        batch.observations, rescale_images(batch.images), obs_offset, obs_scale, camera_keys,
        train=training, random_crop=random_crop, crop_shape=crop_shape, crop_key=crop_key,
    )
    # Bootstrap half always goes through target_vision/target_obs_encoder,
    # never the live/training vision weights - see this module's own
    # docstring for why.
    target_vision_params = {cam: m.params for cam, m in target_vision_models.items()}
    next_obs, _ = encode_observation_from_models(
        target_vision_models, target_vision_params, target_obs_encoder_model, target_obs_encoder_model.params,
        batch.next_observations, rescale_images(batch.next_images), obs_offset, obs_scale, camera_keys,
        train=False, random_crop=False, crop_shape=crop_shape, crop_key=None,
    )

    # Target critic: deterministic (no dropout), no gradient. Min-aggregated
    # over the ensemble - matches the reference's own aggr="min" (clipped
    # double-Q / TD3-style target selection).
    next_actions = bootstrap_next_actions(
        batch, actor_network_model, actor_vision_models, actor_obs_encoder_model,
        actor_obs_offset, actor_obs_scale, actor_camera_keys, Ta, execution_horizon, act_dim, actor_rng,
    )
    next_logits = critic.apply(
        target_critic.params, next_obs, next_actions, training=False
    )
    next_probs = jax.nn.softmax(next_logits, axis=-1)
    next_q_per_critic = hlgauss_transform_from_probs(next_probs, support)
    next_q = jnp.min(next_q_per_critic, axis=0)
    next_q = jax.lax.stop_gradient(next_q)

    # SARSA target (default) - or, when actor_network_model is given, the
    # actor's own predicted next action - see xqc/agents/critics/vision_utils.py's
    # own bootstrap_next_actions docstring.
    target_scalar = sarsa_td_target(batch.rewards, batch.masks, batch.discount, next_q)
    target_scalar = jax.lax.stop_gradient(target_scalar)
    target_probs = hlgauss_transform_to_probs(target_scalar, support, sigma)

    logits = critic.apply(
        critic_params,
        obs,
        batch.actions,
        training=training,
        rngs={"dropout": dropout_key},
    )
    n_critics, bs, num_bins = logits.shape
    logits_flat = logits.reshape(n_critics * bs, num_bins)
    targets_flat = jnp.tile(target_probs, (n_critics, 1))

    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    loss_per_sample = -jnp.sum(targets_flat * log_probs, axis=-1)
    # batch.valid (bs,) weighting - see mlp/critic.py's own
    # mse_critic_loss_fn comment for the full rationale. Tiled n_critics
    # times to match loss_per_sample's own (n_critics*bs,) flattening above.
    valid_flat = jnp.tile(batch.valid, n_critics)
    loss = (loss_per_sample * valid_flat).mean()

    q_values = hlgauss_transform_from_probs(jax.nn.softmax(logits, axis=-1), support)
    td_error = jnp.abs(q_values.mean(axis=0) - target_scalar)

    return loss, {
        "critic_loss": loss,
        "q_values": q_values.mean(),
        "q_min": q_values.min(),
        "q_max": q_values.max(),
        "target_q": target_scalar.mean(),
        "target_q_min": target_scalar.min(),
        "target_q_max": target_scalar.max(),
        "td_error_mean": td_error.mean(),
        "td_error_max": td_error.max(),
        # Distributional-head-only health checks - a collapsed or
        # boundary-saturated distribution can hide behind a fine-looking
        # decoded Q mean (see chunked_critic_target.py's own docstring).
        "q_entropy": entropy_from_log_probs(log_probs).mean(),
        "q_frac_at_boundary": fraction_at_boundary(q_values, support[0], support[-1]),
        "r": batch.rewards.mean(),
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
    support: jnp.ndarray,
    sigma: float,
    dropout_key: PRNGKey,
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
    own predicted next action) - see xqc/agents/critics/vision_utils.py's
    own bootstrap_next_actions docstring for the full rationale."""
    critic_loss_fn = partial(
        hlgauss_critic_loss_fn,
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
        batch=batch,
        support=support,
        sigma=sigma,
        dropout_key=dropout_key,
        training=True,
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

    return new_critic, new_vision_models, new_obs_encoder, info


@partial(jax.jit, static_argnames=("camera_keys", "random_crop", "crop_shape"))
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
    support: jnp.ndarray,
    sigma: float,
    dropout_key: PRNGKey,
) -> InfoDict:
    """No-gradient pass over a held-out batch (chunked_qc_buffer.py's Dataset.
    get_val_batch), dropout OFF (training=False) - a generalization check
    should be deterministic. `dropout_key` is still threaded through for a
    consistent call signature even though it's unused whenever
    training=False (Flax never actually draws from the "dropout" rng
    stream in that branch)."""
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    _, info = hlgauss_critic_loss_fn(
        critic.params,
        vision_params,
        obs_encoder_model.params,
        critic.batch_stats,
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
        batch=batch,
        support=support,
        sigma=sigma,
        dropout_key=dropout_key,
        training=False,
    )
    info.pop("new_vision_batch_stats", None)
    return info
