"""Flax port of the ChiUNet flow-map backbone, originally built for the MIP
(Minimum Iterative Policy) agent and now shared by MIPLearner, NFLearner
(as the coupling subnetwork), and MFLearner (as the velocity net u_fn).

Ported from xqc/networks/mip/networks/chiunet.py (PyTorch, channel-first NCW).
This port keeps Flax's native channel-last (N, T, C) convention throughout, so
no permutes are needed at the boundaries (unlike the original, which permutes
NCW<->NWC because PyTorch's Conv1d is channel-first).

Multi-level dim_mult (real down/upsampling, e.g. [1, 2, 2]) has been verified
via a direct weight-transplant numerical parity test against the PyTorch
reference ChiUNet (both dim_mult=[1,2] with Ta=8 and dim_mult=[1,2,2] with
Ta=16, obs_as_global_cond=True), including Upsample1d's ConvTranspose, which
needed both padding='SAME' and transpose_kernel=True to reproduce the
PyTorch ConvTranspose1d's exact values (not just matching output shape).

obs_as_global_cond=False (local conditioning) mirrors the original exactly:
`condition` must have length Ta (not To) - it's a per-action-timestep signal,
not pooled into the global embedding. Two ChiResidualBlocks map it to
model_dim channels (h_local[0] = full resolution, h_local[1] = downsampled
once), injected additively at two fixed points: right after the first
down-block's first ChiResidualBlock (h_local[0]), and right before the last
up-block's second ChiResidualBlock (h_local[1]). When dim_mult has only one
level, the ups loop never runs, so h_local[1]'s injection point never fires -
inherent to the architecture, not specific to this port.
"""
from typing import List, Optional, Sequence

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from xqc.networks.hetstat import (
    HetStatMeanHead_mf,
    StationaryFeatures,
    StationaryFeatures_mf,
    periodic_relu_activation,
    sin_cos_activation,
)

# periodic_relu_activation moved to xqc/networks/hetstat.py (this file's own
# local copy removed) so both files' StationaryFeatures*/ChiUNet share one
# definition - see hetstat.py's own copy for the full docstring.
_STATIONARY_ACTIVATIONS = {
    "sin_cos": sin_cos_activation,
    "periodic_relu": periodic_relu_activation,
}


def _pytorch_kernel_init(key: jax.random.PRNGKey, shape: Sequence[int], dtype=jnp.float32) -> jnp.ndarray:
    """Matches torch.nn.Linear/Conv1d/ConvTranspose1d's actual default weight
    init exactly: kaiming_uniform_(a=sqrt(5)), i.e.
    Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) (gain=sqrt(1/3) for
    leaky_relu(sqrt(5)), std=gain/sqrt(fan_in), bound=sqrt(3)*std -
    simplifies to bound=1/sqrt(fan_in) exactly). The reference's own
    ConditionalUnet1D never overrides its Conv1d/Linear/ConvTranspose1d
    layers' init, so this default is load-bearing, not incidental. Flax's
    own default (lecun_normal, std=1/sqrt(fan_in) - a factor of sqrt(3)
    LARGER std, and normal not uniform) is a real, uncorrected fidelity gap
    for every nn.Dense/nn.Conv in this file otherwise - see git history.
    `shape` is (*kernel_spatial_dims, in_channels, out_channels) for
    nn.Conv/nn.ConvTranspose, (in_features, out_features) for nn.Dense -
    fan_in is the product of every dim except the last either way."""
    fan_in = int(np.prod(shape[:-1]))
    bound = 1.0 / np.sqrt(fan_in)
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _pytorch_bias_init(fan_in: int):
    """torch.nn.Linear/Conv1d/ConvTranspose1d's default bias init:
    Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) - the SAME bound as the
    weight's own kaiming_uniform_ bound above, but keyed off the layer's
    fan_in (computed from its input, not its own (features,)-shaped bias
    array) - Flax's bias_init callback only ever sees the bias's own shape,
    so fan_in has to be threaded in explicitly by the caller at each
    nn.Dense/nn.Conv call site (see this module's own call sites for how)."""
    bound = 1.0 / np.sqrt(fan_in)

    def init(key: jax.random.PRNGKey, shape: Sequence[int], dtype=jnp.float32) -> jnp.ndarray:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def positional_embedding(x: jnp.ndarray, dim: int, max_positions: float = 10000.0) -> jnp.ndarray:
    """Sinusoidal timestep embedding (DDPM++/ADM style). x: (B,) -> (B, dim).

    Frequency spacing matches the reference's SinusoidalPosEmb exactly:
    `freqs_i = max_positions ** (-i / (n_freqs - 1))` for i in [0, n_freqs),
    n_freqs = dim // 2 - so freqs[0] == 1 and freqs[-1] == 1/max_positions
    exactly. Was dividing by n_freqs (not n_freqs - 1) - a fixed-point-only
    boundary matched (freqs[0] still 1 either way) but every other frequency,
    including the last, was off - see git history."""
    n_freqs = dim // 2
    freqs = jnp.arange(0, n_freqs, dtype=jnp.float32)
    freqs = freqs / max(n_freqs - 1, 1)
    freqs = (1.0 / max_positions) ** freqs
    args = x[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def group_norm(x: jnp.ndarray, num_groups: int) -> jnp.ndarray:
    """nn.GroupNorm(num_groups, ...) - a literal, fixed group count, matching
    the reference's own Conv1dBlock (diffusion_policy/model/diffusion/
    conv1d_components.py: nn.GroupNorm(n_groups, out_channels), n_groups=8
    by default, no adaptive logic at all) exactly. Was min(num_groups,
    dim // 4) - only ever coincided with the reference's fixed value
    because every channel width actually used in this repo (256+) is
    comfortably above the threshold where the two formulas would diverge -
    removed in favor of the reference's own simpler, literal formula."""
    return nn.GroupNorm(num_groups=num_groups, epsilon=1e-5)(x)


class FreezableBatchNorm(nn.Module):
    """BatchNorm over every axis but the channel one (batch and sequence).
    train=True normalises with the current batch's statistics and updates
    the running averages; train=False uses the running averages - a fixed
    per-channel affine map, so unlike GroupNorm it keeps an off-data input's
    scale at eval.

    freeze_stats=True (train only) stop-gradients the batch statistics, so
    each sample's derivatives - including forward-mode ones - are a
    function of that sample alone. MeanFlow's du/dt target needs exactly
    that per-sample jax.jvp, which live batch statistics would couple
    across the batch. It must NOT be used for the parameter-gradient pass
    as well: with frozen statistics, gradient steps along directions the
    next batch's normalisation removes never change the loss, so those
    parameters drift without bound and the lagging running averages stop
    matching the network at eval (Ioffe & Szegedy 2015, sec. 2). Ported
    from stationarity/network.py's toy version (verified there against a
    two-pass jax.jvp training loop, stationarity/training.py's
    make_mean_flow_loss jvp_apply_fn), with one real fix on top: the toy's
    two passes were two independent top-level .apply() calls, each reading
    the SAME pre-step running average and only one (apply_train)'s own
    mutation kept, the other's discarded - harmless there. xqc/agents/actors/mf/
    mean_flow.py's MeanFlow instead calls this same bound submodule (self.net)
    TWICE within one method (live pass, then frozen pass inside jax.jvp) -
    since both calls share the same mutable variable dict in that single
    trace, the second call would otherwise read the FIRST call's
    already-updated running average and blend again on top of it (a silent
    double EMA-decay step every training step). Fixed here by having
    freeze_stats also skip the running-average write outright (below), not
    just stop-gradient the normalisation - the frozen pass has no business
    touching the running average at all, only the live pass's own update
    should ever land."""
    momentum: float = 0.99
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool, freeze_stats: bool = False) -> jnp.ndarray:
        c = x.shape[-1]
        ra_mean = self.variable("batch_stats", "mean", lambda: jnp.zeros((c,), x.dtype))
        ra_var = self.variable("batch_stats", "var", lambda: jnp.ones((c,), x.dtype))
        if train:
            axes = tuple(range(x.ndim - 1))
            mean, var = x.mean(axes), x.var(axes)
            if freeze_stats:
                mean, var = jax.lax.stop_gradient(mean), jax.lax.stop_gradient(var)
            if not self.is_initializing() and not freeze_stats:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * jax.lax.stop_gradient(mean)
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * jax.lax.stop_gradient(var)
        else:
            mean, var = ra_mean.value, ra_var.value
        y = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        scale = self.param("scale", nn.initializers.ones, (c,))
        bias = self.param("bias", nn.initializers.zeros, (c,))
        return y * scale + bias


def normalize(
    x: jnp.ndarray, norm: str, num_groups: int, train: bool, freeze_stats: bool = False,
    epsilon: float = 1e-5,
) -> jnp.ndarray:
    """Dispatcher between this file's norm layers - norm: "group"
    (group_norm above, the default, unchanged behavior), "batch"
    (FreezableBatchNorm, see its own docstring for train/freeze_stats), or
    "batch_ref" (explicit request - see conversation history: a reference
    control, Flax's own standard nn.BatchNorm, used to check whether a
    training-dynamics issue seen with "batch" - e.g. a loss plateau - is
    specific to FreezableBatchNorm's own custom code, or reproduces
    identically with a battle-tested, independently-implemented BatchNorm
    too).

    "batch_ref" has no freeze_stats mechanism as such (Flax's own
    nn.BatchNorm exposes no way to compute live batch statistics but
    stop-gradient them the way FreezableBatchNorm does) - freeze_stats=True
    instead forces it to use its stored running average (use_running_average
    =True) even during an otherwise-live (train=True) call, i.e. it's
    treated as a plain constant w.r.t. differentiation (not computed from -
    and so not coupled to other samples in - the current batch at all),
    achieving the same goal as FreezableBatchNorm's stop-gradient trick via
    a different mechanism: MeanFlow.loss()'s own frozen jax.jvp pass never
    differentiates through (or mutates) live per-batch statistics. This is
    NOT optional - without it, calling this with train=True inside
    jax.jvp (as MeanFlow.loss() used to do for any norm other than "batch")
    crashes with jax.errors.UnexpectedTracerError: nn.BatchNorm's own
    internal running-average mutation is a side effect that isn't allowed
    to happen inside a jax.jvp-traced function (confirmed - see conversation
    history for the real traceback this produced). "none" is not offered
    here (unlike stationarity/network.py's toy version) since every
    existing call site in this file already always normalizes - add it
    if/when a caller actually needs to skip norm entirely."""
    if norm == "group":
        return group_norm(x, num_groups)
    if norm == "batch":
        return FreezableBatchNorm(epsilon=epsilon)(x, train, freeze_stats)
    if norm == "batch_ref":
        use_running_average = freeze_stats or not train
        return nn.BatchNorm(momentum=0.99, epsilon=epsilon, use_running_average=use_running_average)(x)
    raise ValueError(f"Unknown norm {norm!r}, expected 'group', 'batch' or 'batch_ref'")


class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(
            features=self.dim, kernel_size=(3,), strides=(2,), padding=[(1, 1)],
            kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(3 * x.shape[-1]),
        )(x)


class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # padding=[(1, 1)] (the PyTorch ConvTranspose1d(k=4,s=2,p=1) equivalent by
        # naive translation) does NOT reproduce exact 2x upsampling in Flax - the
        # padding conventions for transposed conv differ between frameworks.
        # padding='SAME' gives the right output length, but transpose_kernel=True
        # is also required to reproduce PyTorch's actual values (Flax's default
        # ConvTranspose does not spatially flip the kernel the way PyTorch's
        # ConvTranspose1d does) - verified via a direct weight-transplant
        # numerical parity test against the PyTorch reference ChiUNet.
        return nn.ConvTranspose(
            features=self.dim, kernel_size=(4,), strides=(2,), padding="SAME", transpose_kernel=True,
            kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(4 * x.shape[-1]),
        )(x)


class ChiResidualBlock(nn.Module): # residual block with film conditioning.
    """Port of ConditionalResidualBlock1D (conditional_unet1d.py): two
    Conv1dBlocks (Conv1d -> GroupNorm -> Mish) plus a FiLM cond_encoder
    (Mish -> Linear -> reshape). Was nn.gelu at all three activation sites
    here - reverted to jax.nn.mish (x*tanh(softplus(x)), the exact formula
    torch.nn.Mish computes) to match the reference exactly - see git
    history."""
    out_dim: int
    kernel_size: int = 5
    cond_predict_scale: bool = True
    n_groups: int = 8
    # See normalize()'s own docstring - "group" (default, unchanged),
    # "batch", or "batch_ref". train/freeze_stats below only matter when
    # norm="batch" (freeze_stats is a no-op for "batch_ref").
    norm: str = "group"
    # batch_norm_epsilon (explicit request - see conversation history: the
    # critic's own BatchNorm usage (xqc/networks/mlp.py) uses epsilon=1e-3,
    # 100x larger than this file's own default below - only meaningful for
    # norm in ("batch", "batch_ref"), threaded through to normalize()'s own
    # epsilon kwarg.
    batch_norm_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray, train: bool = True, freeze_stats: bool = False) -> jnp.ndarray:
        residual = x

        out = nn.Conv(
            features=self.out_dim, kernel_size=(self.kernel_size,),
            kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(self.kernel_size * x.shape[-1]),
        )(x)
        out = normalize(out, self.norm, self.n_groups, train, freeze_stats, epsilon=self.batch_norm_epsilon)
        out = jnn.mish(out)

        embed = jnn.mish(emb) # the conditioning.
        cond_dim = 2 * self.out_dim if self.cond_predict_scale else self.out_dim
        embed = nn.Dense(
            cond_dim, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(embed.shape[-1]),
        )(embed)
        if self.cond_predict_scale: # default case (see config).
            embed = embed.reshape(embed.shape[0], 2, -1)
            scale = embed[:, 0, :][:, None, :]
            bias = embed[:, 1, :][:, None, :]
            out = scale * out + bias # film formula.
        else:
            out = out + embed[:, None, :]

        out2 = nn.Conv(
            features=self.out_dim, kernel_size=(self.kernel_size,),
            kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(self.kernel_size * out.shape[-1]),
        )(out)
        out2 = normalize(out2, self.norm, self.n_groups, train, freeze_stats, epsilon=self.batch_norm_epsilon)
        out2 = jnn.mish(out2)

        if residual.shape[-1] != self.out_dim:
            residual = nn.Conv(
                features=self.out_dim, kernel_size=(1,),
                kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(residual.shape[-1]),
            )(residual)
        return out2 + residual  # skip connection.


class ChiUNet(nn.Module):
    act_dim: int
    Ta: int
    obs_dim: int
    To: int
    model_dim: int = 256
    emb_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    # GroupNorm's own group count, used literally (no adaptive logic) by
    # every ChiResidualBlock and the final output conv's own GroupNorm - see
    # group_norm's own docstring. Matches the reference's ConditionalUnet1D's
    # own n_groups field exactly (train_diffusion_unet_hybrid_workspace.yaml/
    # train_diffusion_unet_lowdim_workspace.yaml both set n_groups: 8).
    n_groups: int = 8
    # See normalize()'s own docstring - "group" (default, unchanged),
    # "batch" (FreezableBatchNorm), or "batch_ref" (Flax's own nn.BatchNorm,
    # a reference control for "batch" - see normalize()'s own docstring),
    # each threaded through every ChiResidualBlock here plus the final
    # pre-head norm below. train/freeze_stats on __call__ only matter when
    # this is "batch" (freeze_stats is a no-op for "batch_ref") - the
    # two-pass jax.jvp wiring that actually uses freeze_stats is plumbed
    # through from xqc/agents/actors/mf/mean_flow.py's MeanFlow, the only
    # agent that sets norm="batch" today.
    norm: str = "group"
    # batch_norm_epsilon (explicit request - see conversation history: the
    # critic's own BatchNorm usage (xqc/networks/mlp.py) uses epsilon=1e-3,
    # 100x larger than this file's own default below) - only meaningful when
    # norm is "batch"/"batch_ref", threaded through every ChiResidualBlock
    # here plus the final pre-head norm below, same as norm itself.
    batch_norm_epsilon: float = 1e-5
    obs_as_global_cond: bool = True
    dim_mult: Sequence[int] = (1, 2, 2)
    max_positions: float = 10000.0
    # Multiplies both time inputs (s, t) before the sinusoidal embedding
    # (positional_embedding above). That embedding's frequencies are sized
    # (matching the reference's own SinusoidalPosEmb, diffusion_policy/
    # model/diffusion/positional_embedding.py) for integer DDPM timesteps -
    # DP's own t is a genuine integer in [0, 99] (confirmed against the
    # reference repo and xqc/agents/actors/dp/diffusion_policy.py's own
    # predict_noise/loss), so max_positions=10000 already gives it real
    # resolution, unchanged, time_scale=1.0 default. MIP/MF/MIP-Flow/NF/AF
    # instead feed CONTINUOUS t/r in [0, 1] (MeanFlow especially has no
    # relationship to the DP reference at all - see mean_flow.py's own
    # module docstring) - on that scale most of the embedding's frequency
    # channels barely move across the whole [0,1] range (verified
    # numerically - see conversation history), so the network can barely
    # tell time apart. A caller with continuous time can set this to
    # rescale its own t/s onto a range the embedding was actually designed
    # for - see xqc/agents/actors/mf/mean_flow.py's own time_scale field,
    # forwarded straight through (default 1.0 there too - a caller like
    # conf/actor/mf.yaml can set 99.0, DP's own exact [0, 99] timestep
    # range, deliberately not stationarity/network.py's toy default of
    # 1000 - see that field's own comment for why).
    time_scale: float = 1.0
    # Override the input/output channel count independently of act_dim, so
    # this backbone can be reused for non-action-shaped I/O (e.g. the
    # normalizing-flow agent's masked coupling layers, xqc/agents/actors/nf/flow.py,
    # which feed in a masked-action + mask channel and read out (scale,
    # shift)). None (default) preserves MIP's own behavior exactly.
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    # DP pins `s` to 0 always (see xqc/agents/actors/dp/diffusion_policy.py's
    # predict_noise) - the shared s/t split below (each getting emb_dim//2,
    # i.e. half the sinusoidal frequency pairs) exists for MIP/MF's own
    # genuine two-time-scale conditioning, not needed here. With s pinned,
    # emb_s is a constant (cos(0)=1, sin(0)=0 regardless of frequency), so
    # half of ChiUNet's embedding capacity carries no information for DP -
    # the reference's own diffusion_step_encoder uses the FULL emb_dim width
    # (all emb_dim//2 frequency pairs) for its single timestep embedding.
    # True here skips the s embedding entirely and gives t the full width,
    # matching the reference exactly, without changing MIP/NF/MF's default
    # (False) two-time-scale behavior at all.
    use_full_width_t_embedding: bool = False
    # The reference's ConditionalUnet1D has NO learned projection between the
    # raw (flattened, uncompressed) obs-encoder output and the FiLM
    # conditioning - `global_feature = cat([diffusion_step_encoder(t),
    # global_cond])` is fed AS-IS into every residual block's own
    # Mish->Linear cond_encoder (conditional_unet1d.py's forward/__init__).
    # True (default, unchanged for MIP/NF/MF) projects the flattened
    # `condition` through an extra nn.Dense(emb_dim) first - required for
    # MIP's own architecture (its own reference has an equivalent
    # compression stage - see xqc/agents/common/obs_encoder.py's module
    # docstring), but not something the DP reference does.
    # False (DP-only) skips that Dense and concatenates cond_flat raw,
    # matching the reference exactly.
    project_condition: bool = True
    # The final output conv (below) is zero-initialized by default - True
    # here, unchanged for MIP/NF/MF. The reference's own ConditionalUnet1D
    # (conditional_unet1d.py) does NOT zero-init its final_conv - it's built
    # with a bare nn.Sequential(Conv1dBlock(...), nn.Conv1d(...)), no
    # _init_weights/zero_module override anywhere for this architecture
    # (only diffusion_policy's separate transformer-based policy variant,
    # not used here, has one). Since there's no comment here explaining
    # zero-init as a deliberate, load-bearing choice for MIP/NF/MF either,
    # False (matching the reference exactly, PyTorch/Flax default init)
    # is DP-only for now rather than assumed safe to flip for everyone.
    zero_init_final_conv: bool = True
    # use_stationary_head/bottleneck_dim/feature_dimension/
    # stationary_activation/w_mean_stddev/stationary_mode (explicit request
    # - see conversation history, and xqc/agents/common/chiunet_stationary.py's
    # own now-superseded StationaryChiUNet, whose tail was merged in here):
    # False (default) is today's plain final 1x1 conv, unchanged. True
    # replaces just the final projection with a bounded random-Fourier-
    # feature head (xqc/networks/hetstat.py's StationaryFeatures/
    # HetStatMeanHead_mf) - see those fields' own comments in
    # chiunet_stationary.py (now removed) for what each one does; only
    # meaningful when use_stationary_head=True.
    use_stationary_head: bool = False
    # None: no bottleneck Dense projection - StationaryFeatures is fed
    # z_input directly (it reads its own input dim off inputs.shape[-1], see
    # xqc/networks/hetstat.py, so this is a no-op architecturally beyond
    # skipping that one layer). A concrete int: project down to that dim
    # first, as before - this stays the yaml default (24) for backward
    # compatibility with existing checkpoints/configs.
    bottleneck_dim: Optional[int] = 24
    feature_dimension: int = 128
    stationary_activation: str = "sin_cos"
    w_mean_stddev: float = 1e-3
    stationary_mode: str = "concat"
    # log_ls_init: only meaningful when use_stationary_head=True, same as
    # the fields just above - forwarded straight to StationaryFeatures'
    # own log_ls_init field (xqc/networks/hetstat.py), the trainable
    # per-bottleneck-dim log-lengthscale's init value (bigger lengthscale
    # -> the stationary readout stays "on" further from training data
    # before reverting toward zero). -5.0 matches StationaryFeatures' own
    # default exactly, i.e. today's behavior, unchanged - the toy
    # comparison (stationarity/models.py's StationaryMLP.log_ls_init, see
    # stationarity/compare_stationarity_mf.py's --unet_log_ls_init sweep)
    # tried -2 as an alternative worth comparing in production too.
    log_ls_init: float = -5.0
    # stationary_features_variant: "default" (StationaryFeatures) or "mf"
    # (StationaryFeatures_mf) - which class builds the stationary readout's
    # random-Fourier-feature basis (xqc/networks/hetstat.py). Only
    # meaningful when use_stationary_head=True, same as the fields above.
    # "default" (unchanged) uses a pi/4 phase-shift constant tuned for
    # sin_cos_activation. "mf" uses pi/2, tuned for periodic_relu_activation
    # instead - matching stationarity/models.py's own toy-validated
    # MeanFlow stationary-readout construction exactly (see
    # StationaryFeatures_mf's own docstring for the full reasoning: MeanFlow
    # differentiates through this activation via jax.jvp, and only
    # periodic_relu_activation was ever validated for that). Independent of
    # stationary_activation just above (which activation function either
    # class actually runs) - to reproduce what the toy validated, set BOTH
    # stationary_features_variant="mf" AND stationary_activation=
    # "periodic_relu" together (see StationaryFeatures_mf's own docstring
    # for why "mf" alone, still paired with the default "sin_cos" activation,
    # is an untested combination, not a recommended one).
    stationary_features_variant: str = "default"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        s: jnp.ndarray,
        t: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        train: bool = True,
        freeze_stats: bool = False,
    ):
        """Args:
            x: (B, Ta, act_dim)
            s: (B,) source time
            t: (B,) target time
            condition: (B, To, obs_dim) if obs_as_global_cond, else (B, Ta,
                obs_dim) - local conditioning is a per-action-timestep signal,
                not a pooled observation history. None means "unconditional"
                in either mode.
            train/freeze_stats: only meaningful when norm="batch" - see
                FreezableBatchNorm's own docstring. Defaulted so every
                existing caller (DP/NF/MIP/MF/AF/MIP-Flow, all norm="group"
                today) is unaffected.

        Returns:
            y: (B, Ta, act_dim)
            scalar: (B, 1)
        """
        assert x.shape[1] & (x.shape[1] - 1) == 0, f"Ta must be a power of 2, got {x.shape[1]}"
        original_emb_dim = self.emb_dim
        s, t = s * self.time_scale, t * self.time_scale

        # all the embedding of the conditioning.
        if self.use_full_width_t_embedding:
            # DP-only path: s unused (see field docstring above) - t alone
            # gets the full emb_dim width, matching the reference's
            # diffusion_step_encoder exactly.
            emb = positional_embedding(t, original_emb_dim, self.max_positions)
        else:
            emb_s = positional_embedding(s, original_emb_dim // 2, self.max_positions) # positional embedding.
            emb_t = positional_embedding(t, original_emb_dim // 2, self.max_positions)
            emb = jnp.concatenate([emb_s, emb_t], axis=-1)
        emb = nn.Dense(
            original_emb_dim * 4, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(emb.shape[-1]),
        )(emb)
        emb = jnn.mish(emb)  # matches diffusion_step_encoder's nn.Mish() - was nn.gelu, see git history
        emb = nn.Dense(
            original_emb_dim, kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(emb.shape[-1]),
        )(emb)

        h_local = None
        if self.obs_as_global_cond:  # default is this one.
            if condition is not None:
                cond_flat = condition.reshape(condition.shape[0], -1)
                condition_emb = nn.Dense(
                    original_emb_dim, kernel_init=_pytorch_kernel_init,
                    bias_init=_pytorch_bias_init(cond_flat.shape[-1]),
                )(cond_flat) if self.project_condition else cond_flat
            else:
                condition_emb = jnp.zeros(
                    (x.shape[0], original_emb_dim if self.project_condition else self.obs_dim * self.To),
                    dtype=x.dtype,
                )
            emb = jnp.concatenate([emb, condition_emb], axis=-1)
        else: # not our case. 
            if condition is None:
                condition = jnp.zeros((x.shape[0], self.Ta, self.obs_dim), dtype=x.dtype)
            assert condition.shape[1] == x.shape[1], (
                f"Local conditioning requires condition length == Ta ({x.shape[1]}), "
                f"got {condition.shape[1]}"
            )
            local_block1 = ChiResidualBlock(self.model_dim, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)
            local_block2 = ChiResidualBlock(self.model_dim, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)
            local_downsample = Downsample1d(self.model_dim)
            h_local = [
                local_block1(condition, emb, train, freeze_stats),
                local_downsample(local_block2(condition, emb, train, freeze_stats)),
            ]

        in_channels = self.in_channels if self.in_channels is not None else self.act_dim # input set to action dim (takes noise as input of the size of the action). 
        out_channels = self.out_channels if self.out_channels is not None else self.act_dim # obviously the size the final action chunk. 

        dims = [self.model_dim * m for m in np.cumprod(self.dim_mult)]
        # dims = [256*1, 256*2, 256*4] = [256, 512, 1024]
        in_out = list(zip([in_channels] + dims[:-1], dims))
        # in_out = [(in_channels, 256), (256, 512), (512, 1024)]
        num_resolutions = len(in_out)

        # downsampling path. 
        h = []
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)
            x = ChiResidualBlock(dim_out, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)(x, emb, train, freeze_stats)
            if idx == 0 and h_local is not None:
                x = x + h_local[0]
            x = ChiResidualBlock(dim_out, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)(x, emb, train, freeze_stats)
            h.append(x)
            if not is_last:
                x = Downsample1d(dim_out)(x) # corresponds to the Max pooling.

        mid_dim = dims[-1]
        x = ChiResidualBlock(mid_dim, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)(x, emb, train, freeze_stats)
        x = ChiResidualBlock(mid_dim, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)(x, emb, train, freeze_stats)

        scalar = jnp.mean(x, axis=1)
        scalar = nn.Dense(128)(scalar)
        scalar = nn.silu(scalar)
        scalar = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(scalar)

        # upsampling path. 
        num_ups = len(in_out[1:])
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (num_resolutions - 1)
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = ChiResidualBlock(dim_in, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)(x, emb, train, freeze_stats)
            if idx == num_ups - 1 and h_local is not None:
                x = x + h_local[1]
            x = ChiResidualBlock(dim_in, self.kernel_size, self.cond_predict_scale, self.n_groups, self.norm, self.batch_norm_epsilon)(x, emb, train, freeze_stats)
            if not is_last:
                x = Upsample1d(dim_in)(x)

        # projects x to the model dimension => by default is 256. 
        x = nn.Conv(
            features=self.model_dim, kernel_size=(self.kernel_size,),
            kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(self.kernel_size * x.shape[-1]),
        )(x)
        x = normalize(x, self.norm, self.n_groups, train, freeze_stats, epsilon=self.batch_norm_epsilon)
        x = jnn.mish(x)  # final_conv's Conv1dBlock also uses Mish - was nn.gelu, see git history


        if self.use_stationary_head:
            # See this class's own use_stationary_head field comment, and
            # the (now-removed) xqc/agents/common/chiunet_stationary.py for
            # the original standalone version this was merged in from.
            if self.stationary_mode == "concat":
                z_input = x.reshape(x.shape[0], -1)
                mean_head_dims = out_channels * self.Ta
            elif self.stationary_mode == "independent":
                z_input = x
                mean_head_dims = out_channels
            else:
                raise ValueError(
                    f"Unknown stationary_mode: {self.stationary_mode!r}. Expected 'concat' or 'independent'."
                )
            if self.bottleneck_dim is not None:
                # Plain nn.Dense, no kernel_init/bias_init override - unlike
                # every other Dense/Conv in this file (which all use the
                # PyTorch-matching _pytorch_kernel_init/_pytorch_bias_init,
                # see that function's own docstring), this one deliberately
                # matches stationarity/models.py's own StationaryMLP
                # bottleneck Dense exactly (`nn.Dense(features=
                # self.bottleneck_dim)(x)`, no init args there either) - the
                # toy's own validated construction for this layer, Flax's
                # defaults (lecun_normal kernel, zeros bias), not a PyTorch
                # fidelity target the way the rest of ChiUNet is (this layer
                # has no reference-repo equivalent to match in the first
                # place - it's part of the stationary-head addition, not the
                # ported ConditionalUnet1D architecture) - explicit request,
                # see conversation history.
                z = nn.Dense(features=self.bottleneck_dim)(z_input)
            else:
                z = z_input
            if self.stationary_features_variant == "default":
                stationary_features_cls = StationaryFeatures
            elif self.stationary_features_variant == "mf":
                stationary_features_cls = StationaryFeatures_mf
            else:
                raise ValueError(
                    f"Unknown stationary_features_variant: {self.stationary_features_variant!r}. "
                    "Expected 'default' or 'mf'."
                )
            f = stationary_features_cls(
                feature_dimension=self.feature_dimension,
                stationary_activation=_STATIONARY_ACTIVATIONS[self.stationary_activation],
                log_ls_init=self.log_ls_init,
            )(z)
            loc_weights = HetStatMeanHead_mf(
                self.feature_dimension, mean_head_dims, w_mean_stddev=self.w_mean_stddev, name="HetStatMean",
            )()
            x = (f @ loc_weights).reshape(x.shape[0], self.Ta, out_channels)


        elif self.zero_init_final_conv: # we never enter here as zero_init_final_conv is set to False in the respective agents.
            x = nn.Conv(
                features=out_channels,
                kernel_size=(1,),
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(x)
        else:
            x = nn.Conv(
                features=out_channels, kernel_size=(1,),
                kernel_init=_pytorch_kernel_init, bias_init=_pytorch_bias_init(x.shape[-1]),
            )(x)

        return x, scalar
