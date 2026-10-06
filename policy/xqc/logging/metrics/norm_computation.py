"""Computation of parameter and gradient norms for metrics logging."""

from typing import Dict
import jax
import jax.numpy as jnp
import numpy as np
from xqc.networks.common import Params
from .utils import add_all_key, flatten_dict, add_prefix_to_dict


def get_num_parameters_dict(
    param_dict: Params,
) -> Dict[str, int]:
    """Return dictionary that contains number of trainable parameters for each layer."""
    _pcount_dict = jax.tree_util.tree_map(lambda x: np.prod(x.shape), param_dict)
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_pcount = add_all_key(_pcount_dict)

    # Construct a pcount dict
    pcount_dict = {}
    for module, layer_dict in _updated_pcount.items():
        # e.g. module = "encoder" or "predictor"
        pcount_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                "pcount",
                sep="_",
            )
        )

    return pcount_dict


def get_normalization_pnorm(
    param_dict: Params,
) -> Dict[str, jnp.ndarray]:
    """Compute average parameter values (for normalization layers)."""
    # we take the mean here instead of the norm
    _pnorm_dict = jax.tree_util.tree_map(lambda x: jnp.mean(x), param_dict)
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_pnorm = add_all_key(_pnorm_dict)

    # Construct a pnorm dict
    pnorm_dict = {}

    for module, layer_dict in _updated_pnorm.items():
        # e.g. module = "encoder" or "predictor"
        pnorm_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                "avg_pnorm",
                sep="_",
            )
        )

    return pnorm_dict


def get_pnorm(
    param_dict: Params,
    pcount_dict: Params,
) -> Dict[str, jnp.ndarray]:
    """Compute parameter norm dictionary with aggregated statistics.

    Args:
        param_dict: Frozen dictionary containing parameter values
        pcount_dict: Dictionary containing parameter counts per layer

    Returns:
        Dictionary with parameter norms, including:
        - Per-layer norms (kernel, bias, kernel+bias)
        - Total norms (encoder, predictor, overall)
        - Effective norms (weighted by parameter count)

    Note:
        Norm values for vmapped functions (multi-head Q-networks) are summed.
    """
    _pnorm_dict = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), param_dict)
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_pnorm = add_all_key(_pnorm_dict)

    # Construct a pnorm dict
    pnorm_dict = {}
    last_layer_names = ["value", "mean", "log_std"]

    eff_pnorm_numer, eff_pnorm_denom = 0.0, 0.0
    eff_pnorm_total_numer, eff_pnorm_total_denom = 0.0, 0.0
    eff_pnorm_encoder_numer, eff_pnorm_encoder_denom = 0.0, 0.0
    eff_pnorm_predictor_numer, eff_pnorm_predictor_denom = 0.0, 0.0
    pnorm_total = 0.0
    pnorm_encoder_total = 0.0
    pnorm_predictor_total = 0.0
    for module, layer_dict in _updated_pnorm.items():
        # e.g. module = "encoder" or "predictor"
        pnorm_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                "pnorm",
                sep="_",
            )
        )

    # Aggregation
    for _layer_name, _pnorm in pnorm_dict.items():
        _layer = _layer_name.replace("pnorm_", "")
        if ("kernel+bias" in _layer) or ("total" in _layer):
            continue
        pnorm_total += jnp.square(_pnorm)

        # Compute effective parameter norms
        eff_pnorm_total_numer += pcount_dict["pcount_" + _layer] * jnp.square(_pnorm)
        eff_pnorm_total_denom += pcount_dict["pcount_" + _layer]

        eff_pnorm_numer += pcount_dict["pcount_" + _layer] * jnp.square(_pnorm)
        eff_pnorm_denom += pcount_dict["pcount_" + _layer]
        if any(last_layer_name in _layer_name for last_layer_name in last_layer_names):
            pnorm_predictor_total += jnp.square(_pnorm)
            eff_pnorm_predictor_numer += pcount_dict["pcount_" + _layer] * jnp.square(
                _pnorm
            )
            eff_pnorm_predictor_denom += pcount_dict["pcount_" + _layer]
        else:
            pnorm_encoder_total += jnp.square(_pnorm)
            eff_pnorm_encoder_numer += pcount_dict["pcount_" + _layer] * jnp.square(
                _pnorm
            )
            eff_pnorm_encoder_denom += pcount_dict["pcount_" + _layer]

    pnorm_dict["pnorm_total"] = jnp.sqrt(pnorm_total)
    pnorm_dict["encoder/pnorm_total"] = jnp.sqrt(pnorm_encoder_total)
    pnorm_dict["predictor/pnorm_total"] = jnp.sqrt(pnorm_predictor_total)

    pnorm_dict["effective_pnorm_total"] = jnp.sqrt(
        eff_pnorm_numer / (eff_pnorm_denom + 1e-5)
    )
    pnorm_dict["encoder/effective_pnorm_total"] = jnp.sqrt(
        eff_pnorm_encoder_numer / (eff_pnorm_encoder_denom + 1e-5)
    )
    pnorm_dict["predictor/effective_pnorm_total"] = jnp.sqrt(
        eff_pnorm_predictor_numer / (eff_pnorm_predictor_denom + 1e-5)
    )

    return pnorm_dict


def get_gnorm(
    grad_dict: Params,
    pcount_dict: Params,
) -> Dict[str, jnp.ndarray]:
    """Compute gradient norm dictionary with aggregated statistics.

    Args:
        grad_dict: Frozen dictionary containing gradients for each parameter
        pcount_dict: Dictionary containing parameter counts per layer

    Returns:
        Dictionary with gradient norms, including:
        - Per-layer gradient norms (kernel, bias, kernel+bias)
        - Total gradient norms (encoder, predictor, overall)
        - Effective gradient norms (weighted by parameter count)

    Note:
        Norm values for vmapped functions (multi-head Q-networks) are summed.
    """
    _gnorm_dict = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), grad_dict)
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_gnorm = add_all_key(_gnorm_dict)

    # Construct a gnorm dict
    gnorm_dict = {}
    last_layer_names = ["value", "mean", "log_std"]
    eff_gnorm_numer, eff_gnorm_denom = 0.0, 0.0
    eff_gnorm_total_numer, eff_gnorm_total_denom = 0.0, 0.0
    eff_gnorm_encoder_numer, eff_gnorm_encoder_denom = 0.0, 0.0
    eff_gnorm_predictor_numer, eff_gnorm_predictor_denom = 0.0, 0.0
    gnorm_total = 0.0
    gnorm_encoder_total = 0.0
    gnorm_predictor_total = 0.0
    for module, layer_dict in _updated_gnorm.items():
        # e.g. module = "encoder" or "predictor"
        gnorm_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                "gnorm",
                sep="_",
            )
        )

    # Aggregation
    for _layer_name, _gnorm in gnorm_dict.items():
        _layer = _layer_name.replace("gnorm_", "")
        if ("kernel+bias" in _layer) or ("total" in _layer):
            continue
        gnorm_total += jnp.square(_gnorm)

        # Compute effective parameter norms
        eff_gnorm_total_numer += pcount_dict["pcount_" + _layer] * jnp.square(_gnorm)
        eff_gnorm_total_denom += pcount_dict["pcount_" + _layer]

        eff_gnorm_numer += pcount_dict["pcount_" + _layer] * jnp.square(_gnorm)
        eff_gnorm_denom += pcount_dict["pcount_" + _layer]
        if any(last_layer_name in _layer_name for last_layer_name in last_layer_names):
            gnorm_predictor_total += jnp.square(_gnorm)
            eff_gnorm_predictor_numer += pcount_dict["pcount_" + _layer] * jnp.square(
                _gnorm
            )
            eff_gnorm_predictor_denom += pcount_dict["pcount_" + _layer]
        else:
            gnorm_encoder_total += jnp.square(_gnorm)
            eff_gnorm_encoder_numer += pcount_dict["pcount_" + _layer] * jnp.square(
                _gnorm
            )
            eff_gnorm_encoder_denom += pcount_dict["pcount_" + _layer]

    gnorm_dict["gnorm_total"] = jnp.sqrt(gnorm_total)
    gnorm_dict["encoder/gnorm_total"] = jnp.sqrt(gnorm_encoder_total)
    gnorm_dict["predictor/gnorm_total"] = jnp.sqrt(gnorm_predictor_total)

    gnorm_dict["effective_gnorm_total"] = jnp.sqrt(eff_gnorm_numer / eff_gnorm_denom)
    gnorm_dict["encoder/effective_gnorm_total"] = jnp.sqrt(
        eff_gnorm_encoder_numer / (eff_gnorm_encoder_denom + 1e-5)
    )
    gnorm_dict["predictor/effective_gnorm_total"] = jnp.sqrt(
        eff_gnorm_predictor_numer / (eff_gnorm_predictor_denom + 1e-5)
    )

    return gnorm_dict
