"""Plain MLP behavior-cloning policy over action chunks (or single actions):
a minimal, deliberately non-generative baseline, meant for isolating
pipeline bugs from flow-technique bugs. DP/FM/MF/MIP/MIP-Flow all share
ChiUNet plus noise/time conditioning plus iterative sampling; this shares
only the data pipeline with them (xqc/agents/chunking/chunked_learner.py's
ChunkedPolicyLearner: dataset, obs_encoder, vision, normalization) and none
of that machinery. Direct regression: action = MLP(obs_embedding), trained
with plain MSE. Deterministic sampling (same observation always gives the
same action, like MIP) - no internal randomness anywhere.

Naturally supports both single-step (Ta=1) and chunked (Ta>1) actions with no
special-casing: unlike ChiUNet, a plain MLP has no power-of-2/downsampling
constraint, so Ta=1 isn't even a special case here (contrast with the
ChiUNet-backed agents, which need dim_mult=(1,) to support Ta=1).
"""
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from xqc.agents.common.loss_masking import masked_mean


class MLPPolicy(nn.Module):
    """act_dim, Ta, cond_dim, To: same meaning/shapes as the other agents'
    own network fields, for interface parity - cond_dim/To aren't actually
    used here (the MLP infers its input width from whatever `cond` shape
    it's called with), kept only so this class can be constructed via
    ChunkedPolicyLearner exactly like the ChiUNet-backed networks."""

    act_dim: int
    Ta: int
    cond_dim: int
    To: int
    hidden_dims: Sequence[int] = (256, 256)

    def setup(self):
        self.layers = [nn.Dense(h) for h in self.hidden_dims]
        self.out = nn.Dense(self.Ta * self.act_dim)

    def predict(self, cond: jnp.ndarray) -> jnp.ndarray:
        """cond: (B, To, cond_dim). Returns (B, Ta, act_dim)."""
        b = cond.shape[0]
        x = cond.reshape(b, -1)
        for layer in self.layers:
            x = nn.relu(layer(x))
        x = self.out(x)
        return x.reshape(b, self.Ta, self.act_dim)

    def loss(
        self, act: jnp.ndarray, cond: jnp.ndarray, rng: jax.random.PRNGKey, mask: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, dict]:
        """act: (B, Ta, act_dim) demonstration actions, normalized to
        (-1, 1). cond: (B, To, cond_dim). mask: optional (B,) per-sample
        weight (see xqc/agents/common/loss_masking.py) - None (default)
        reproduces plain-mean behavior exactly. Returns (loss, infos). Plain
        MSE - no noise, no time conditioning, nothing stochastic. `rng` is
        unused (accepted only so this matches ChunkedPolicyLearner's uniform
        network.loss(act, cond, rng) call site - see module docstring)."""
        del rng
        pred = self.predict(cond)
        per_sample_loss = jnp.mean((pred - act) ** 2, axis=(1, 2))
        loss = masked_mean(per_sample_loss, mask)
        return loss, {"bc_loss": loss}

    def sample(self, rng: jax.random.PRNGKey, shape, cond: jnp.ndarray) -> jnp.ndarray:
        """Deterministic: identical to predict(), exposed under this name/
        signature to match ChunkedPolicyLearner's uniform
        network.sample(rng, act_shape, cond) call site - `rng`/`shape` are
        both unused (the output shape is already fixed by Ta/act_dim)."""
        del rng, shape
        return self.predict(cond)
