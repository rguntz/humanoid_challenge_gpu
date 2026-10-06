from typing import Any, Callable, Optional, Sequence, Union

import flax.linen as nn

import jax.numpy as jnp
from flax.linen import BatchNorm, LayerNorm
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
EPS = 1e-8


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


# -----------------------------------------------------------------------------
# Activation functions
# -----------------------------------------------------------------------------


class ReLU(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.relu(x)


# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------


class BatchNormEmbedder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return BatchNorm(
            use_running_average=not training,
            momentum=0.99,
            epsilon=0.001,
        )(x)


class LayerNormEmbedder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return LayerNorm()(x)


class IdentityEmbedder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        return x


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------


class Block(nn.Module):
    feature_dim: int
    activation_fn: Callable = ReLU
    skip_connections: bool = False

    def get_layer(self, x, training):
        raise NotImplementedError

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        residual = x
        x = self.get_layer(x, training)
        if self.skip_connections and residual.shape == x.shape:
            x = x + residual
        return x


class XQCBlock(Block):
    """Dense -> BN -> Act (Pre-Activation BN style)"""

    def get_layer(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.feature_dim, use_bias=False, kernel_init=default_init())(x)
        x = BatchNorm(use_running_average=not training, momentum=0.99, epsilon=0.001)(x)
        x = self.activation_fn()(x)
        self.sow("activations", "encoder", x)
        return x


class CrossQBlock(Block):
    """Dense -> Act -> BN (Post-Activation BN style)"""

    def get_layer(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.feature_dim, use_bias=True, kernel_init=default_init())(x)
        x = self.activation_fn()(x)
        self.sow("activations", "encoder", x)
        x = BatchNorm(use_running_average=not training, momentum=0.99, epsilon=0.001)(x)
        return x


class LNBlock(Block):
    def get_layer(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.feature_dim, use_bias=True, kernel_init=default_init())(x)
        x = LayerNorm()(x)
        x = self.activation_fn()(x)
        self.sow("activations", "encoder", x)
        return x


class DenseBlock(Block):
    def get_layer(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.feature_dim, use_bias=True, kernel_init=default_init())(x)
        x = self.activation_fn()(x)
        self.sow("activations", "encoder", x)
        return x


# -----------------------------------------------------------------------------
# Predictors
# -----------------------------------------------------------------------------


class ScalarPredictor(nn.Module):
    n_outputs: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(
            features=self.n_outputs,
            use_bias=True,
            kernel_init=default_init(),
            name="value", # this means that it's not going to get normalized by the function norm_network. 
        )(x)
        self.sow("activations", "predictor", x)
        return x


class TanhGaussPredictor(nn.Module):
    action_dim: int
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> tfd.Distribution:
        means = nn.Dense(
            self.action_dim, use_bias=True, kernel_init=default_init(), name="mean"
        )(x)
        log_stds = nn.Dense(
            self.action_dim,
            use_bias=True,
            kernel_init=default_init(scale=1.0),
            name="log_std",
        )(x)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        self.sow(
            "activations", "predictor", jnp.concatenate([means, log_stds], axis=-1)
        )

        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * self.temperature
        )
        return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())


# -----------------------------------------------------------------------------
# MLP
# -----------------------------------------------------------------------------


class MLP(nn.Module):
    predictor: nn.Module
    hidden_dims: Sequence[int]
    block_class: Any  # Class type of the block to use
    embedder: Optional[nn.Module] = None
    skip_connections: bool = False

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, training: bool = False
    ) -> Union[jnp.ndarray, tfd.Distribution]:

        # Embed
        if self.embedder:
            x = self.embedder(x, training=training)

        # Blocks
        for i, size in enumerate(self.hidden_dims):
            x = self.block_class(
                feature_dim=size, skip_connections=self.skip_connections
            )(x, training=training)

        # Predictor
        if self.predictor:
            x = self.predictor(x, training=training)

        return x
