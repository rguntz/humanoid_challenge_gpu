"""Dictionary and parameter manipulation utilities for metrics computation."""

import jax
import jax.numpy as jnp
import flax
import flax.traverse_util
from flax.core import FrozenDict
from xqc.networks.common import Params


def add_all_key(d):
    """Recursively compute norms for nested parameter dictionaries.

    For each layer with 'kernel' and 'bias', computes:
    - Combined kernel+bias norm
    - Separate kernel norm
    - Separate bias norm
    """
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict) or isinstance(value, FrozenDict):
            new_dict[key] = add_all_key(value)
            if "kernel" in new_dict[key] and "bias" in new_dict[key]:
                kernel_norm = jnp.square(new_dict[key]["kernel"])
                bias_norm = jnp.square(new_dict[key]["bias"])
                # Integrated Norm
                new_dict[key + "_kernel+bias"] = jnp.sqrt(kernel_norm + bias_norm)
                # Separated Norm
                new_dict[key + "_kernel"] = jnp.sqrt(kernel_norm)
                new_dict[key + "_bias"] = jnp.sqrt(bias_norm)
        else:
            new_dict[key] = jnp.linalg.norm(value)
    return new_dict


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten nested dictionary with separator."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or isinstance(v, FrozenDict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def add_prefix_to_dict(d: dict, prefix: str = None, sep="/") -> dict:
    """Add prefix to all keys in dictionary."""
    new_dict = {}
    for key, value in d.items():
        new_dict[prefix + sep + key] = value
    return new_dict


def merge_critics_dict(combined_critic_params_dict):
    """Merge two critic parameter dictionaries by averaging."""
    tree = list(combined_critic_params_dict.keys())[0].split("_")[0]
    return jax.tree_util.tree_map(
        lambda x, y: (x + y) / 2,
        combined_critic_params_dict[f"{tree}_0"],
        combined_critic_params_dict[f"{tree}_1"],
    )


def merge_two_dicts(d1, d2):
    """Merge two dictionaries by averaging their values."""
    return jax.tree_util.tree_map(lambda x, y: (x + y) / 2, d1, d2)


def extract_dense_layers(d: Params):
    """Extract only Dense layer parameters from parameter dictionary."""
    flattened_dict = flax.traverse_util.flatten_dict(d)
    filtered_flattened_dict = {
        k: v for k, v in flattened_dict.items() if "Dense" in k[-2]
    }
    filtered_dict = flax.traverse_util.unflatten_dict(filtered_flattened_dict)
    return filtered_dict


def get_last_dense_layer(d: Params):
    """Get the name of the last Dense layer in the parameter dictionary."""
    flattened_dict = flax.traverse_util.flatten_dict(d)
    filtered_flattened_dict = {
        k: v for k, v in flattened_dict.items() if "Dense" in k[-2]
    }
    sorted_layers = sorted([k[-2] for k in list(filtered_flattened_dict.keys())])
    last_layer = sorted_layers[-1]
    return last_layer
