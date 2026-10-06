"""Activation-based metrics computation: norms, plasticity, and dormancy.

Based on: https://github.com/awjuliani/deep-rl-plasticity
"""

from typing import Dict, List
import jax.numpy as jnp
import flax.traverse_util


def get_feature_norm(
    activations: Dict[str, List[jnp.ndarray]],
) -> Dict[str, jnp.ndarray]:
    """Compute feature norm statistics for network activations.

    Args:
        activations: Dictionary mapping layer names to activation tensors

    Returns:
        Dictionary containing:
        - featnorm_<layer>: Mean L2 norm across batch
        - featmagnitude_<layer>: Mean normalized magnitude
        - featnorm_std_<layer>: Standard deviation of norms
        - featnorm_total: Sum of all layer norms
    """
    norms = {}
    total_norm = 0.0
    activations_flat = flax.traverse_util.flatten_dict(activations, sep="_")

    for layer_name, activs in list(activations_flat.items()):
        if isinstance(activs, tuple):
            if len(activs) > 1:
                # If sow is called multiple times in the same module, the activations are stored as a tuple
                raise ValueError("Only one activation per module is supported.")
            activs = activs[0]

        # Compute the L2 norm for all examples in the batch at once
        batch_norms = jnp.linalg.norm(activs, ord=2, axis=-1)

        # Compute the expected (mean) L2 norm across the batch
        expected_norm = jnp.mean(batch_norms)
        expected_std = jnp.std(batch_norms)
        expected_magnitude = jnp.mean((batch_norms * (1 / jnp.sqrt(activs.shape[-1]))))

        norms[f"featnorm_{layer_name}"] = expected_norm
        norms[f"featmagnitude_{layer_name}"] = expected_magnitude
        norms[f"featnorm_std_{layer_name}"] = expected_std
        total_norm += expected_norm

    norms["featnorm_total"] = total_norm

    return norms


def compute_dormant_units(
    activations: Dict[str, jnp.ndarray],
    tau: float,
) -> Dict[str, jnp.ndarray]:
    """Compute the ratio of dormant units per layer.

    A unit is considered dormant if its activation score s <= tau, where
    s is the activation normalized by the mean activation across features.

    Args:
        activations: Dictionary mapping layer names to activation tensors
        tau: Threshold for dormancy (typically 0.0)

    Returns:
        Dictionary with dormant unit ratios per layer and overall total
    """
    s_scores_dict = {}
    activations_flat = flax.traverse_util.flatten_dict(activations)

    # Calculate the s scores for each layer
    for layer_name, layer_activations in activations_flat.items():
        # if multiple layers are logged as one name, e.g., encoder
        if isinstance(layer_activations, tuple):
            layer_activations = jnp.concatenate(layer_activations, axis=0)
        s_scores = layer_activations / (
            jnp.mean(layer_activations, axis=2, keepdims=True) + 1e-6
        )
        s_scores = jnp.mean(s_scores, axis=1)
        # remove the architecture name, e.g., VmapCritic_0 and MLP_0
        layer_key = "_".join(layer_name[-2:])
        s_scores_dict[layer_key] = s_scores

    rdu = {}
    rdu["total"] = []
    # Calculate the ratio of dormant units
    for name, s_scores in s_scores_dict.items():
        reset_mask = s_scores <= tau
        rdu[name] = jnp.mean(reset_mask)
        rdu["total"].append(rdu[name])

    rdu["total"] = jnp.mean(jnp.array(rdu["total"]))
    return {f"dormant_neurons_{k}": v for k, v in rdu.items()}


def compute_stable_rank(
    features: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Compute stable rank of activations using singular value cumulative sum.

    Stable rank measures the effective dimensionality of the representation
    by counting how many singular values are needed to explain 99% of variance.

    Args:
        features: Dictionary mapping layer names to activation tensors

    Returns:
        Dictionary with stable rank per encoder layer and overall total
    """
    features_flat = flax.traverse_util.flatten_dict(features, sep="_")

    penultimate_key = list(filter(lambda k: "encoder" in k, features_flat.keys()))[-1]
    features = features_flat[penultimate_key]
    if isinstance(features, tuple):
        if len(features) > 1:
            raise ValueError("Only one activation per module is supported.")
        features = features[0]

    svals = jnp.linalg.svdvals(features)
    svals_sum = jnp.sum(svals, axis=-1, keepdims=True)
    svals_cumsum = jnp.cumsum(svals, axis=-1)
    stable_rank = jnp.sum(svals_cumsum / svals_sum < 0.99, axis=-1) + 1
    stable_rank = jnp.mean(stable_rank)
    return {"stable_rank_total": stable_rank}


def compute_plasticity_metrics(
    features: Dict[str, jnp.ndarray],
    prefix: str = "",
) -> Dict[str, jnp.ndarray]:
    """Compute all plasticity-related metrics for network activations.

    Args:
        features: Dictionary mapping layer names to activation tensors
        log_prefix: Prefix to add to metric names (e.g., "critic" or "actor")

    Returns:
        Dictionary containing:
        - <prefix>_dormant_neurons_<layer>: Ratio of dormant neurons
        - <prefix>_dormant_neurons_total: Overall dormant neuron ratio
        - <prefix>_stable_rank_<layer>: Stable rank per layer
        - <prefix>_stable_rank_total: Overall stable rank
    """
    dormant_neurons = compute_dormant_units(features, tau=0.0)
    stable_rank = compute_stable_rank(features)
    infos = {**dormant_neurons, **stable_rank}
    return {prefix + k: v for k, v in infos.items()}
