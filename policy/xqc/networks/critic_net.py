"""Implementations of algorithms for continuous control."""

from typing import Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp

from xqc.networks.mlp import (
    MLP,
    BatchNormEmbedder,
    CrossQBlock,
    DenseBlock,
    IdentityEmbedder,
    LayerNormEmbedder,
    LNBlock,
    ScalarPredictor,
    XQCBlock,
)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    n_outputs: int
    pre_activation_bn: bool
    use_layer_norm: bool
    use_batch_norm: bool
    skip_connections: bool
    min_v: float = None
    max_v: float = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate((observations, actions), axis=-1)

        # Determine Embedder
        if self.use_batch_norm:
            embedder = BatchNormEmbedder()
            block_class = XQCBlock if self.pre_activation_bn else CrossQBlock
        elif self.use_layer_norm:
            embedder = LayerNormEmbedder()
            block_class = LNBlock
        else:
            embedder = IdentityEmbedder()
            block_class = DenseBlock

        mlp = MLP(
            embedder=embedder,
            predictor=ScalarPredictor(
                n_outputs=self.n_outputs, name="predictor_scalar"
            ),
            hidden_dims=self.hidden_dims,
            block_class=block_class,
            skip_connections=self.skip_connections,
        )
        values = mlp(x, training=training)

        # Scalar critic for MSE loss
        if self.n_outputs == 1:
            return jnp.squeeze(values, -1), None

        # Distributional critic
        else:
            bin_values = jnp.linspace(
                self.min_v, self.max_v, values.shape[1], dtype=jnp.float32
            )
            log_probs = nn.log_softmax(values, axis=1)
            values = jnp.sum(jnp.exp(log_probs) * bin_values, axis=1)
            return values, log_probs


class VMapCritic(nn.Module):
    max_v: float
    min_v: float
    hidden_dims: Sequence[int]
    n_outputs: int
    n_critics: int
    pre_activation_bn: bool
    use_layer_norm: bool
    use_batch_norm: bool
    skip_connections: bool

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        training: bool,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q_values, log_probs = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0, "activations": 0},
            split_rngs={"params": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )(
            self.hidden_dims,
            n_outputs=self.n_outputs,
            max_v=self.max_v,
            min_v=self.min_v,
            pre_activation_bn=self.pre_activation_bn,
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            skip_connections=self.skip_connections,
        )(
            observations, actions, training
        )

        return q_values, {"log_probs": log_probs}
