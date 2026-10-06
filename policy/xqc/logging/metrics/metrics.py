"""Main metrics computation and orchestration module."""

import jax
import jax.numpy as jnp
import flax.traverse_util

from .norm_computation import (
    get_num_parameters_dict,
    get_gnorm,
    get_pnorm,
    get_normalization_pnorm,
)
from .learning_metrics import get_effective_lr
from .plasticity_metrics import get_feature_norm, compute_plasticity_metrics


def compute_metrics(params, grads, activations, prefix=""):
    """Compute parameter, gradient, and basic activation metrics.

    Args:
        params: Parameter values
        grads: Gradients
        activations: Layer activations

    Returns:
        Dictionary with metrics (NO plasticity - computed separately)
    """

    def separate_params(d):
        dense_dict = {}
        bn_dict = {}
        d_flat = flax.traverse_util.flatten_dict(d)
        for path, param in d_flat.items():
            if "BatchNorm" in path[-2] or "LayerNorm" in path[-2]:
                bn_dict[path] = param
            else:
                dense_dict[path] = param
        return flax.traverse_util.unflatten_dict(
            dense_dict
        ), flax.traverse_util.unflatten_dict(bn_dict)

    params, bn_params = separate_params(params)
    grads, bn_grads = separate_params(grads)

    pcount = jax.vmap(get_num_parameters_dict, in_axes=[0])(params)
    gnorm = jax.vmap(get_gnorm, in_axes=[0, 0])(grads, pcount)
    pnorm = jax.vmap(get_pnorm, in_axes=[0, 0])(params, pcount)
    elr = jax.vmap(get_effective_lr, in_axes=[0, 0, 0])(gnorm, pnorm, pcount)
    feat_norm = jax.vmap(get_feature_norm, in_axes=[0])(activations)

    infos = {**pcount, **gnorm, **pnorm, **elr, **feat_norm}

    if bn_params:
        infos.update(**jax.vmap(get_normalization_pnorm, in_axes=[0])(bn_params))

    # Average over additional critic / actor dimension.
    return {prefix + k: v.mean(0) for k, v in infos.items()}


def compute_logging_metrics(agent, metrics):
    """Compute all logging metrics for agent training.

    Clean, unified computation - no scattered logic!

    Args:
        agent: Agent with actor and critic
        metrics: Dict with grads and activations

    Returns:
        Updated metrics with all computed values
    """
    # Compute base metrics vmapped over seeds (params, grads, features)
    metrics_fn = jax.vmap(compute_metrics, in_axes=[0, 0, 0, None])
    critic_base = metrics_fn(
        agent.critic.params,
        metrics["critic_grads"],
        metrics["critic_activations"],
        "critic_",  # prefix
    )

    # expand actor dim because critic is vmapped
    def tree_expand(params):
        return jax.tree.map(lambda p: jnp.expand_dims(p, axis=1), params)

    actor_base = metrics_fn(
        tree_expand(agent.actor.params),
        tree_expand(metrics["actor_grads"]),
        tree_expand(metrics["actor_activations"]),
        "actor_",  # prefix
    )

    # Compute plasticity separately (vmapped over seeds)
    plasticity_fn = jax.vmap(compute_plasticity_metrics, in_axes=(0, None))
    critic_plasticity = plasticity_fn(
        metrics["critic_activations"], "critic_"  # prefix
    )
    actor_plasticity = plasticity_fn(
        tree_expand(metrics["actor_activations"]), "actor_"  # prefix
    )

    # Merge everything
    metrics.update(critic_base)
    metrics.update(critic_plasticity)
    metrics.update(actor_base)
    metrics.update(actor_plasticity)

    # Remove raw activations and gradients
    metrics.pop("critic_activations", None)
    metrics.pop("actor_activations", None)
    metrics.pop("critic_grads", None)
    metrics.pop("actor_grads", None)

    return metrics
