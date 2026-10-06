from typing import Callable, Sequence, Tuple, Union, Optional, Any
import jax
from tensorflow_probability.substrates import jax as tfp
import flax.linen as nn
import jax.numpy as jnp
import math
from xqc.networks.mlp import MLP, BatchNormEmbedder, XQCBlock, CrossQBlock


tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
EPS = 1e-8


@jax.jit
def sin_cos_activation(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(x) + jnp.cos(x)


def _triangle_activation(x: jnp.ndarray) -> jnp.ndarray:
    z = jnp.floor(x / jnp.pi + 0.5)
    return (x - jnp.pi * z) * (-1) ** z


def periodic_relu_activation(x: jnp.ndarray) -> jnp.ndarray:
    """Piecewise-linear (JVP-friendly) approximation to a cos/sin basis
    function - see StationaryFeatures_mf's own docstring for why this
    matters (MeanFlow differentiates through this activation via jax.jvp;
    sin_cos_activation above was never validated for that, only for
    StationaryHeteroskedasticNormalTanhDistribution's non-jvp use below).
    Copied from xqc/agents/common/network.py's own former local copy
    (moved here so both files' StationaryFeatures* classes can share one
    definition - network.py now imports it from here instead) and
    stationarity/models.py's toy copy (kept separate on purpose - see that
    folder's own module docstring: no cross-repo import into stationarity/)."""
    pdiv4 = 0.785398163
    pdiv2 = 1.570796326
    return (_triangle_activation(x) + _triangle_activation(x + pdiv2)) * pdiv4


gaussian_init = nn.initializers.normal(stddev=1.0)


class StationaryFeatures(nn.Module):
    feature_dimension: int = 512
    stationary_activation: Callable[[jnp.ndarray], jnp.ndarray] = sin_cos_activation
    stationary_init: Callable = gaussian_init
    # log_ls_init: the trainable per-input-dim log-lengthscale's own init
    # value (StationaryLogLengthscales below) - was hardcoded -5.0, now a
    # field so callers can sweep it (matches stationarity/models.py's own
    # StationaryMLP.log_ls_init from the toy comparison this was originally
    # validated against - see xqc/agents/common/network.py's ChiUNet.log_ls_init
    # field comment for the production wiring). -5.0 default reproduces
    # today's behavior exactly for every existing caller.
    log_ls_init: float = -5.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_dimension = inputs.shape[-1]

        # While the theory says that these random weights should be fixed, it's
        # crucial in practice to let them be trained. The distribution does not
        # actually change much, so they still contribute to the stationary
        # behaviour, and letting them be trained alleviates potential underfitting.
        random_weights = self.param(
            "StationaryRandomWeights",
            self.stationary_init,
            (input_dimension, self.feature_dimension // 2),
        )

        log_lengthscales = self.param(
            "StationaryLogLengthscales",
            nn.initializers.constant(self.log_ls_init),
            (input_dimension,),
        )

        ls = jnp.diag(jnp.exp(log_lengthscales))
        wx = inputs @ ls @ random_weights
        pdiv4 = 0.785398163  # π/4 - the shift that decomposes sin_cos_activation
        # (sin(x)+cos(x) = sqrt(2)*sin(x+pi/4)) back into clean sin/cos
        # components: sin_cos(wx+pi/4) = sqrt(2)*cos(wx), sin_cos(wx-pi/4) =
        # sqrt(2)*sin(wx). This constant is tuned for sin_cos_activation
        # SPECIFICALLY, not activation-agnostic - see StationaryFeatures_mf
        # just below, which uses a different shift for periodic_relu_activation's
        # own natural (different) decomposition. Swapping stationary_activation
        # here without also reconsidering this constant is not a like-for-like
        # comparison - see conversation history.
        f = jnp.concatenate(
            (
                self.stationary_activation(wx + pdiv4),
                self.stationary_activation(wx - pdiv4),
            ),
            axis=-1,
        )
        return f / math.sqrt(self.feature_dimension)


class StationaryFeatures_mf(nn.Module):
    """StationaryFeatures variant matching stationarity/models.py's own
    StationaryMLP construction (the toy's actual validated MeanFlow
    stationary-readout row, compare_stationarity_mf.py) - explicit request,
    see conversation history. Two real differences from StationaryFeatures
    above, both about the phase-shift constant, not just a rename:

    1. stationary_activation defaults to periodic_relu_activation, not
       sin_cos_activation - MeanFlow differentiates through this activation
       via jax.jvp (see xqc/agents/actors/mf/mean_flow.py's loss()), and
       periodic_relu_activation is specifically piecewise-linear so that
       derivative is exact rather than a smooth-curvature approximation
       (see its own docstring). sin_cos_activation was validated for
       StationaryHeteroskedasticNormalTanhDistribution below instead, which
       never takes a jax.jvp through this at all - a genuinely different
       requirement, never actually tested for MeanFlow.
    2. The phase shift is pi/2, not pi/4 - matching periodic_relu_activation's
       own internal construction (triangle(x) + triangle(x+pi/2), see its
       docstring) and stationarity/models.py's own outer usage
       (periodic_relu_activation(ph - pi/2), periodic_relu_activation(ph +
       pi/2)) exactly. StationaryFeatures' own pi/4 is tuned specifically for
       sin_cos_activation's decomposition (see its own comment) - reusing it
       here with a different activation would not reproduce what the toy
       validated.

    stationary_activation stays swappable (same field pattern as
    StationaryFeatures) - so xqc/agents/common/network.py's ChiUNet.norm-style
    "pick the class" switch (stationary_features_variant) and "pick the
    activation" switch (stationary_activation) are independent knobs. To
    reproduce exactly what the toy validated, set BOTH
    stationary_features_variant="mf" AND stationary_activation="periodic_relu"
    together (conf/actor/base_actor.yaml) - this class's own Python-level
    default (periodic_relu_activation) only applies if a caller constructs it
    with no explicit stationary_activation at all, and every existing caller
    in this codebase always passes one explicitly (MeanFlow's own
    stationary_activation field, itself still defaulting to "sin_cos" for
    backward compatibility) - so picking stationary_features_variant="mf"
    alone, without also changing stationary_activation, gives sin_cos with
    THIS class's pi/2 shift: a combination nobody has validated (not the
    toy, which pairs periodic_relu with pi/2; not today's production, which
    pairs sin_cos with pi/4)."""
    feature_dimension: int = 512
    stationary_activation: Callable[[jnp.ndarray], jnp.ndarray] = periodic_relu_activation
    stationary_init: Callable = gaussian_init
    log_ls_init: float = -5.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_dimension = inputs.shape[-1]

        random_weights = self.param(
            "StationaryRandomWeights",
            self.stationary_init,
            (input_dimension, self.feature_dimension // 2),
        )

        log_lengthscales = self.param(
            "StationaryLogLengthscales",
            nn.initializers.constant(self.log_ls_init),
            (input_dimension,),
        )

        ls = jnp.diag(jnp.exp(log_lengthscales))
        wx = inputs @ ls @ random_weights
        pdiv2 = 1.570796326  # pi/2 - periodic_relu_activation's own natural shift, see class docstring
        f = jnp.concatenate(
            (
                self.stationary_activation(wx + pdiv2),
                self.stationary_activation(wx - pdiv2),
            ),
            axis=-1,
        )
        return f / math.sqrt(self.feature_dimension)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Distribution followed by tanh."""

    def __init__(self, distribution, threshold=0.999, validate_args=False):
        """Initialize the distribution.

        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
          validate_args: Passed to super class.
        """
        super().__init__(
            distribution=distribution,
            bijector=tfp.bijectors.Tanh(),
            validate_args=validate_args,
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = (
            self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        )
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event):
        # Without this clip there would be NaNs in the inner tf.where and that
        # causes issues for some reasons.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(
                event >= self._threshold, self._log_prob_right, super().log_prob(event)
            ),
        )

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed=None):
        # We return an estimation using a single sample of the log_det_jacobian.
        # We can still do some backpropagation with this estimate.
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class HetStatMeanHead(nn.Module):
    feature_dimension: int
    num_dimensions: int

    @nn.compact
    def __call__(self):
        return self.param(
            "HetStatMean",
            nn.initializers.constant(0.0),
            (self.feature_dimension, self.num_dimensions),
        )


class HetStatMeanHead_mf(nn.Module):
    """HetStatMeanHead variant for xqc/agents/common/chiunet_stationary.py's
    StationaryChiUNet (explicit request - see conversation history): the
    same (feature_dimension, num_dimensions) "HetStatMean" readout param,
    but initialized with stationarity/models.py's own StationaryMLP
    convention - `nn.initializers.normal(stddev=w_mean_stddev)`, its own
    "w_mean" param - instead of HetStatMeanHead's exact-zero init above.
    HetStatMeanHead's own zero-init matches THIS file's own hetstat actor
    (XQCFD, a non-MeanFlow reference); StationaryMLP's small-random init is
    what was actually validated for the MeanFlow row of the toy
    stationarity comparison (compare_stationarity.py's own run_mean_flow) -
    this class exists so StationaryChiUNet can match THAT reference instead
    without touching HetStatMeanHead itself (still used elsewhere,
    unchanged)."""
    feature_dimension: int
    num_dimensions: int
    w_mean_stddev: float = 1e-3

    @nn.compact
    def __call__(self):
        return self.param(
            "HetStatMean",
            nn.initializers.normal(stddev=self.w_mean_stddev),
            (self.feature_dimension, self.num_dimensions),
        )


class HetStatCovarianceHead(nn.Module):
    feature_dimension: int
    num_dimensions: int

    @nn.compact
    def __call__(self):
        n_sqrt = self.feature_dimension * (self.feature_dimension + 1) // 2

        return self.param(
            "HetStatCovSqrt",
            nn.initializers.constant(0.0),
            (self.num_dimensions, n_sqrt),
        )


class StationaryHeteroskedasticNormalTanhDistribution(nn.Module):
    """Module that produces a stationary TanhTransformedDistribution."""

    hidden_dims: Sequence[int]
    action_dim: int
    use_batch_norm: bool
    use_layer_norm: bool
    pre_activation_bn: bool
    prior_variance: float = 1.0
    stationary_activation: Callable[[jnp.ndarray], jnp.ndarray] = sin_cos_activation
    feature_dimension: int = 512
    skip_connections: bool = False

    def setup(self):
        super().setup()
        self.prior_var = self.prior_variance
        self.prior_stddev = math.sqrt(self.prior_variance)

    @nn.compact # just means you can define variables and add a return at the end of the function. 
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
        return_cut_dist: bool = False,
    ) -> Union[tfd.Distribution, Tuple[tfd.Distribution, tfd.Distribution]]:
        # Determine Embedder
        embedder = BatchNormEmbedder()
        block_class = XQCBlock if self.pre_activation_bn else CrossQBlock

        z = MLP(
            embedder=embedder,
            predictor=None,  # We want to keep the hidden dimension
            hidden_dims=self.hidden_dims,
            block_class=block_class,
            skip_connections=self.skip_connections,
        )(inputs, training)
        features = StationaryFeatures(stationary_activation=self.stationary_activation)(
            z
        )

        # sometimes we vmap the features, so the batch dim will miss. This is usually not an issue since nn.modules take care of that,
        # but here we use einsum, which requires a batch dim
        features = jnp.atleast_2d(features)

        if return_cut_dist: # stop the gradient or not
            features_ = jax.lax.stop_gradient(features)
        else:
            features_ = features

        loc_weights = HetStatMeanHead(
            self.feature_dimension, self.action_dim, name="HetStatMean"
        )()

        scale_cross_weights_sqrt_raw = HetStatCovarianceHead(
            self.feature_dimension, self.action_dim, name="HetStatCovSqrt"
        )()
        # Convert vector into a lower triangular matrix with exponentiated diagonal,
        # so a vector of zeros becomes the identity matrix.
        b = tfb.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=None)
        scale_cross_weights_sqrt = jax.vmap(b.forward)(scale_cross_weights_sqrt_raw)

        loc = features @ loc_weights

        # Cholesky decomposition: A = LL^T where L is lower triangular
        # Variance is diagonal of x @ A @ x.T = x @ L @ L.T @ x.T
        # so first compute x @ L per output d
        var_sqrt = jnp.einsum("dij,bi->bdj", scale_cross_weights_sqrt, features_)
        var = jnp.einsum("bdi,bdi->bd", var_sqrt, var_sqrt)
        
        scale = self.prior_stddev * jnp.tanh(jnp.sqrt(math.exp(LOG_STD_MIN) ** 2 + var)) 
        

        # undo the previous manual dim expansion
        scale = scale.squeeze()
        loc = loc.squeeze()

        distribution = tfd.Normal(loc=loc, scale=scale * temperature)

        transformed_distribution = tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )

        if return_cut_dist:
            cut_distribution = tfd.Normal(loc=jax.lax.stop_gradient(loc), scale=scale)
            cut_transformed_distribution = tfd.Independent(
                TanhTransformedDistribution(cut_distribution),
                reinterpreted_batch_ndims=1,
            )
            return transformed_distribution, cut_transformed_distribution
        else:
            return transformed_distribution
