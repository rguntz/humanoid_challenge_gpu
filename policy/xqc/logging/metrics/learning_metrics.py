"""Learning-related metrics: effective learning rate and rank computation."""

from typing import Dict
import jax.numpy as jnp
from xqc.networks.common import Params


def get_effective_lr(
    gnorm_dict: Params,
    pnorm_dict: Params,
    pcount_dict: Params,
) -> Dict[str, jnp.ndarray]:
    """Compute effective learning rate from gradient and parameter norms.

    Taken from: https://github.com/dojeon-ai/SimbaV2/blob/d1d446798b83a9ec23c844d134645f9608ec8750/scale_rl/agents/jax_utils/metrics.py#L222

    Args:
        gnorm_dict: Dictionary containing gradient norms for each parameter
        pnorm_dict: Dictionary containing parameter norms for each parameter
        pcount_dict: Dictionary containing parameter counts for each parameter

    Returns:
        Dictionary with effective learning rates:
        - Per-layer effective LR
        - Total effective LR (encoder, predictor, overall)

    Note:
        Norm values for vmapped functions (multi-head Q-networks) are summed to a single value.
    """
    eff_lr_dict = {}
    last_layer_names = ["value", "mean", "log_std"]
    eff_lr_encoder_numer, eff_lr_encoder_denom = 0.0, 0.0
    eff_lr_predictor_numer, eff_lr_predictor_denom = 0.0, 0.0
    eff_lr_total_numer, eff_lr_total_denom = 0.0, 0.0

    for _layer_name, _gnorm in gnorm_dict.items():
        # e.g. module = actor_encoder, _layer_name = gnorm_Dense_0_bias
        # _module, _layer_name = _gnorm_layer.split("/", 1)
        _layer = _layer_name.replace("gnorm_", "")
        if ("kernel+bias" in _layer) or ("total" in _layer) or ("effective" in _layer):
            continue
        eff_lr = _gnorm / pnorm_dict["pnorm_" + _layer]
        eff_lr_dict["effective_lr_" + _layer] = eff_lr

        # Aggregation
        eff_lr_total_numer += pcount_dict["pcount_" + _layer] * eff_lr
        eff_lr_total_denom += pcount_dict["pcount_" + _layer]
        if any(last_layer_name in _layer_name for last_layer_name in last_layer_names):
            eff_lr_predictor_numer += pcount_dict["pcount_" + _layer] * eff_lr
            eff_lr_predictor_denom += pcount_dict["pcount_" + _layer]
        else:
            eff_lr_encoder_numer += pcount_dict["pcount_" + _layer] * eff_lr
            eff_lr_encoder_denom += pcount_dict["pcount_" + _layer]

    eff_lr_dict["encoder/effective_lr_total"] = eff_lr_encoder_numer / (
        eff_lr_encoder_denom + 1e-5
    )
    eff_lr_dict["predictor/effective_lr_total"] = eff_lr_predictor_numer / (
        eff_lr_predictor_denom + 1e-5
    )
    eff_lr_dict["effective_lr_total"] = eff_lr_total_numer / (eff_lr_total_denom + 1e-5)

    return eff_lr_dict
