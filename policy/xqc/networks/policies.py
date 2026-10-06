import functools
from typing import Any, Sequence, Tuple

import flax
import flax.linen
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]

tfd = tfp.distributions
tfb = tfp.bijectors

from xqc.networks.mlp import (
    MLP,
    BatchNormEmbedder,
    CrossQBlock,
    DenseBlock,
    TanhGaussPredictor,
    LayerNormEmbedder,
    LNBlock,
    XQCBlock,
)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    pre_activation_bn: bool
    use_layer_norm: bool
    use_batch_norm: bool
    skip_connections: bool

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:

        if self.use_batch_norm:
            embedder = BatchNormEmbedder()
            block_class = XQCBlock if self.pre_activation_bn else CrossQBlock
        elif self.use_layer_norm:
            embedder = LayerNormEmbedder()
            block_class = LNBlock
        else:
            embedder = None
            block_class = DenseBlock

        mlp = MLP(
            embedder=embedder,
            predictor=TanhGaussPredictor(
                action_dim=self.action_dim,
                temperature=temperature,
                name="predictor_tanh_gauss",
            ),
            hidden_dims=self.hidden_dims,
            block_class=block_class,
            skip_connections=self.skip_connections,
        )

        return mlp(observations, training=training)


@functools.partial(jax.jit, static_argnames=("actor_def", "temperature"))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, 0, None))
def sample_actions_with_log_probs(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    actor_batch_stats: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    variables = {"params": actor_params}
    if actor_batch_stats is not None:
        variables["batch_stats"] = actor_batch_stats
    dist = actor_def.apply(
        variables,
        observations,
        temperature,
    )
    rng, key = jax.random.split(rng)
    action = dist.sample(seed=key)
    log_probs = dist.log_prob(action)
    return rng, action, log_probs
