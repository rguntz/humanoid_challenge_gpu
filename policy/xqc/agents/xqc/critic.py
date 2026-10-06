from functools import partial

from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from xqc.replay_buffer import Batch
from xqc.networks.common import InfoDict, Model, Params, PRNGKey, norm_network


def categorical_critic_loss_fn(
    critic_params: Params,
    critic_batch_stats: Params,
    key: PRNGKey,
    actor: Model,
    critic: Model,
    temperature: Model,
    target_critic: Model,
    batch: Batch,
    kl_regularization: str,
    
) -> Tuple[jnp.ndarray, InfoDict]:

    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_actions_log_probs = dist.log_prob(next_actions)

    concat_obs = jnp.concatenate((batch.observations, batch.next_observations), axis=0)
    concat_actions = jnp.concatenate((batch.actions, next_actions), axis=0)

    # Target critic
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

    min_indices = next_q_values.argmin(axis=0)
    target_log_probs = jax.vmap(lambda x, i: x[i], in_axes=[1, 0])(
        target_log_probs, min_indices
    )
    target_log_probs = jnp.repeat(target_log_probs[None], repeats=2, axis=0)

    (_, critic_infos), state_updates = critic.apply(
        critic_params,
        concat_obs,
        concat_actions,
        batch_stats=critic_batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )
    catted_log_probs = critic_infos["log_probs"]
    pred_log_probs, _ = jnp.split(catted_log_probs, 2, axis=1)

    if kl_regularization == 'prior':
        actor_entropy = jnp.zeros_like(next_actions_log_probs)
    elif kl_regularization == 'uniform':
        actor_entropy = temperature() * next_actions_log_probs

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
        ),
        in_axes=0,
    )(
        pred_log_probs=pred_log_probs,
        target_log_probs=target_log_probs,
    )

    loss = loss.sum()
    return loss, {
        "critic_loss": loss,
        "r": (batch.rewards).mean(),
        "new_batch_stats": state_updates.get("batch_stats", None),
        "critic_activations": state_updates.get("activations", None),
        **jax.tree.map(partial(jnp.mean, axis=0), loss_info),
    }


def categorical_td_loss(
    pred_log_probs: jnp.ndarray,  # (n, num_bins)
    target_log_probs: jnp.ndarray,  # (n, num_bins)
    reward: jnp.ndarray,  # (n,)
    done: jnp.ndarray,  # (n,)
    actor_entropy: jnp.ndarray,  # (n,)
    gamma: jnp.ndarray,
    num_bins: int,
    min_v: float,
    max_v: float,
    valid: Optional[jnp.ndarray] = None,  # (n,) - chunked callers only (xqc_style/critic.py); None (default) preserves this function's original unweighted behavior for the production XQCLearner critic below, which has no concept of `valid` at all.
) -> Tuple[jnp.ndarray, InfoDict]:
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)
    gamma = gamma.reshape(-1, 1)

    # compute target value buckets
    # target_bin_values: (n, num_bins)
    bin_values = jnp.linspace(start=min_v, stop=max_v, num=num_bins).reshape(1, -1)
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * done
    target_bin_values = jnp.clip(target_bin_values, min_v, max_v)  # (B, num_bins)

    # for logging
    clipped_mask = (target_bin_values == min_v) | (target_bin_values == max_v)
    clip_percentage = jnp.mean(clipped_mask)

    # update indices
    b = (target_bin_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    l = jnp.floor(b)
    l_mask = jax.nn.one_hot(l.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    u = jnp.ceil(b)
    u_mask = jax.nn.one_hot(u.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    # target label
    _target_probs = jnp.exp(target_log_probs)
    m_l = (_target_probs * (u + (l == u).astype(jnp.float32) - b)).reshape(
        -1, num_bins, 1
    )
    m_u = (_target_probs * (b - l)).reshape((-1, num_bins, 1))
    target_probs = jax.lax.stop_gradient(jnp.sum(m_l * l_mask + m_u * u_mask, axis=1))

    # cross entropy loss
    per_sample_loss = -jnp.sum(target_probs * pred_log_probs, axis=1)  # (n,)
    # valid weighting (chunked callers only - see this arg's own docstring
    # above): multiply-then-plain-mean, matching qc's own critic_loss =
    # ((q - target_q)**2 * batch['valid']).mean() exactly (see
    # xqc/replay_buffer/chunked_qc_buffer.py's own ChunkedBatch docstring) -
    # not a masked/renormalized mean.
    loss = jnp.mean(per_sample_loss if valid is None else per_sample_loss * valid)

    return loss, {"clip_percentage": clip_percentage}


def mse_critic_loss_fn(
    critic_params: Params,
    critic_batch_stats: Params,
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temperature: Model,
    batch: Batch,
    **kwargs: str,
) -> Tuple[jnp.ndarray, InfoDict]:

    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    (catted_q, _), state_updates = critic.apply(
        critic_params,
        jnp.concatenate((batch.observations, batch.next_observations), axis=0),
        jnp.concatenate((batch.actions, next_actions), axis=0),
        batch_stats=critic_batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )

    current_q_values, _ = jnp.split(catted_q, 2, axis=1)
    (catted_q, _), _ = critic.apply(
        target_critic.params,
        jnp.concatenate((batch.observations, batch.next_observations), axis=0),
        jnp.concatenate((batch.actions, next_actions), axis=0),
        batch_stats=target_critic.batch_stats,
        mutable=["batch_stats", "attributes", "activations"],
        capture_intermediates=lambda mdl, method_name: isinstance(mdl, nn.Dense),
        training=True,
    )
    _, next_q_values = jnp.split(catted_q, 2, axis=1)
    next_q_values = jnp.min(next_q_values, axis=0)
    target_q = next_q_values - temperature() * next_log_probs
    target_q = batch.rewards + batch.discount.squeeze() * batch.masks * target_q
    target_q = jax.lax.stop_gradient(target_q)
    critic_loss = (0.5 * (current_q_values - target_q) ** 2).mean(1).sum()

    return critic_loss, {
        "critic_loss": critic_loss,
        "q_values": current_q_values.mean(),
        "r": batch.rewards.mean(),
        "new_batch_stats": state_updates.get("batch_stats", None),
        "critic_activations": state_updates.get("activations", None),
        "max_target_q": jnp.max(target_q),
        "min_target_q": jnp.min(target_q),
        "target_q": target_q.mean(),
    }


def update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temperature: Model,
    batch: Batch,
    use_weight_norm: bool,
    kl_regularization: str,
) -> Tuple[Model, InfoDict]:
    """Unified critic update function that uses critic.loss_fn."""

    new_critic, grads, info = critic.apply_gradient(
        partial(
            critic.loss_fn,
            key=key,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temperature=temperature,
            batch=batch,
            kl_regularization=kl_regularization
        )
    )
    info["critic_grads"] = grads

    if use_weight_norm:
        new_critic = norm_network(model=new_critic)
    new_critic = new_critic.replace(batch_stats=info.pop("new_batch_stats"))

    return new_critic, info
