"""Shared tanh-squashing helper for IL network `.loss()`/`.sample()` methods
that don't already bound their own output (xqc/agents/actors/mf/mean_flow.py
first, others - dp/fm/mip - opt in later per-file the same way) - explicit
request, see conversation history: xqc/networks/hetstat.py's own
TanhTransformedDistribution (XQCFD) and xqc/agents/actors/nf/flow.py's own
_tanh_transform/_arctanh_transform (NF) already solve this exact problem for
their own actors, each locally in its own file rather than through a shared
orchestrator - this module follows that same convention, factoring out only
the tiny numerically-safe atanh math those two already needed too (not the
decision of whether/where to apply it, which stays with each actor).

Root cause this addresses: an actor whose .sample() is a raw, unconstrained
map (an ODE/denoising solve, unlike XQCFD's own tanh-bijected distribution
or NF's own bijective flow) has nothing architecturally stopping its output
from drifting arbitrarily far from the training data's own (-1, 1) range -
soft imitation-learning pressure keeps it bounded during offline BC, but
online RL's Q-maximization gradient has no such pull once
wasserstein_offline_only leaves online-buffer rows unconstrained (see
conversation history: confirmed empirically via the offline_action_{mean,max,min}/
online_action_{mean,max,min} diagnostic - online action magnitude reached
~4x the valid range within ~9000 online steps for the single-action MF
sweep). Training in atanh-space and squashing only the final sample() output
(the SAC-style squashed-policy trick) makes this a hard architectural bound
instead of an optimization-pressure one.
"""
import jax.numpy as jnp


def safe_arctanh(x: jnp.ndarray, threshold: float = 0.999) -> jnp.ndarray:
    """Inverse of jnp.tanh, clipped away from +-1 first - matches
    xqc/networks/hetstat.py's own TanhTransformedDistribution threshold
    (0.999) and xqc/agents/actors/nf/flow.py's own _ARCTANH_EPS-clipped
    _arctanh_transform exactly (same problem, same fix, same margin).
    Not just a numerical nicety here: the delta-action Square dataset's own
    gripper dimension sits at EXACTLY +-1.0 for roughly half of all recorded
    transitions (checked directly against low_dim_v15.hdf5 - a binary
    open/close control, not a rare edge case) - arctanh(+-1) is +-inf, and
    even a zero-weight/masked-out sample would still propagate NaN through
    the batch via 0 * inf, so this clip is load-bearing for training at all,
    not an edge-case guard."""
    return jnp.arctanh(jnp.clip(x, -threshold, threshold))
