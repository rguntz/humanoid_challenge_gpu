"""Small, dependency-free tfp.distributions helpers shared by
xqc/agents/actors/xqc/xqcfd.py and xqc/agents/chunking/actor.py's
kl_regularization branch (see that function's own docstring). Lives under
xqc/networks/ (which has zero dependency on xqc.agents) rather than beside
either caller specifically so BOTH can import it with no risk of a circular
import: xqc/agents/actors/xqc/xqcfd.py sits behind xqc.agents.actors.xqc's
own package __init__, which xqc/agents/chunking/chunked_learner.py's own
import chain reaches first (chunked_learner imports xqc/agents/chunking/
actor.py, which would need this same helper) - if it lived on either
agents-side file, that import chain would circle back on itself. See
conversation history."""
from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp

tfd = tfp.distributions


def get_base_distribution(dist: tfd.Distribution) -> tfd.Distribution:
    """Unwraps a TransformedDistribution/Independent wrapper chain down to
    the innermost base distribution (for xqc/networks/hetstat.py's own
    StationaryHeteroskedasticNormalTanhDistribution, always the tfd.Normal
    underneath Independent(TanhTransformedDistribution(Normal))) - matches
    xqc/agents/xqc/actor.py's own identically-named helper exactly."""
    if hasattr(dist, "distribution"):
        return get_base_distribution(dist.distribution)
    return dist


def kl_divergence(p: tfd.Distribution, q: tfd.Distribution) -> jnp.ndarray:
    """Analytic (closed-form, no sampling needed) KL(p || q) between two
    StationaryHeteroskedasticNormalTanhDistribution outputs' own base
    Normals, summed over the action dimension - matches xqc/agents/xqc/
    actor.py's own kl_divergence exactly (reimplemented here rather than
    imported - see xqc/agents/actors/xqc/xqcfd.py's own module docstring:
    this whole integration deliberately never imports from xqc/agents/xqc/).
    Unlike MeanFlow's own wasserstein_regularization (xqc/agents/chunking/
    actor.py), which needs a shared-noise Monte Carlo trick because its
    networks have no tractable density at all, this distribution family
    gives an exact closed form here - no rng needed."""
    p = get_base_distribution(p)
    q = get_base_distribution(q)
    return tfd.kl_divergence(p, q).sum(axis=-1)
