"""Shared TD-target arithmetic and training-health diagnostics for every
chunked-SARSA critic variant (xqc/agents/critics/*). Factored out on
purpose (see conversation history): if each critic variant computed these
independently, a subtle bug or inconsistency in one of them would silently
make the cross-variant comparison meaningless - one critic could look
better purely because its target/metric formula differs, not because its
architecture does.

sarsa_td_target implements exactly:

    target = reward_{t:t+h} + gamma^h * mask * next_value

where `reward`/`mask`/`discount` (=gamma^h) come straight from a
ChunkedBatch (xqc/replay_buffer/chunked_qc_buffer.py) and
`next_value` is whatever the caller evaluated the target critic at
(next_observations, next_actions) - either a scalar Q per sample (the MSE/
HL-Gauss critics) or a per-bin value array broadcast against reward/mask/
discount (the categorical/C51 critic's projected-support target). Kept
shape-agnostic (plain elementwise arithmetic) for exactly that reason -
callers reshape reward/mask/discount however they need to broadcast
against next_value.

entropy_from_log_probs/fraction_at_boundary are health checks specific to
the two DISTRIBUTIONAL heads (xqc_style's C51, the transformer's HL-Gauss)
- a decoded scalar Q mean can look fine while the underlying distribution
has quietly collapsed or saturated at the value-support edges, which these
catch directly.

split_dense_and_batchnorm_norms exists because lumping every parameter
into one global weight_norm made xqc_style vs mlp's weight_norm_ratio
curves look contradictory (see conversation history): xqc_style's
norm_network only ever renormalizes the Dense-layer kernels/biases (see
xqc/networks/common.py) to unit column-norm, EVERY step - it never touches
a BatchNorm layer's own scale/bias parameters (use_batch_norm=True for
xqc_style), which drift freely under plain Adam with nothing pulling them
back. Splitting the total into these two pieces makes that visible
directly instead of requiring a manual explanation: the Dense-kernel norm
should stay essentially flat (it's being forced to every step), while any
real growth shows up in the BatchNorm norm instead. mlp has no BatchNorm
at all, so its "batchnorm_norm" should read ~0 throughout - a good sanity
check in its own right. transformer isn't covered by this (LayerNorm-based,
no norm_network/BatchNorm at all).

per_layer_norms goes further, at your supervisor's request: even the
dense/batchnorm split turned out not to be clean (the categorical
predictor's own BIAS is excluded from norm_network's renormalization too -
see conversation history), so rather than keep guessing which piece is
driving growth, this reports one norm per actual layer/module, for any of
the three architectures. Used by xqc/agents/chunking/chunked_learner.py's
own _update and chunked_learner_sarsa.py's own _update (each critic
variant's own former jit_critic_step wrapper functions, which used to do
this same logging, have been deleted - see conversation history) to log a
full per-layer breakdown (and, learner-side, a per-layer ratio-to-init)
under train/layer_norm/<layer path> and train/layer_norm_ratio/<layer path>.
"""
from typing import Dict, List, Optional, Sequence, Union

import flax.traverse_util
import jax.numpy as jnp
import optax

from xqc.agents.common.obs_encoder import encode_observation_from_models
from xqc.agents.critics.vision_utils import rescale_images
from xqc.networks.common import Model
from xqc.replay_buffer.chunked_qc_buffer import ChunkedBatch


def sarsa_td_target(
    reward: jnp.ndarray,
    mask: jnp.ndarray,
    discount: jnp.ndarray,
    next_value: Union[jnp.ndarray, float],
) -> jnp.ndarray:
    return reward + discount * mask * next_value # Mask in case we cannot use the Q function for the next chunk (Q(A4, B1))


def bootstrap_next_actions(
    batch: ChunkedBatch,
    actor_network_model: Optional[Model],
    actor_vision_models: Optional[Dict[str, Model]],
    actor_obs_encoder_model: Optional[Model],
    actor_obs_offset: Optional[jnp.ndarray],
    actor_obs_scale: Optional[jnp.ndarray],
    actor_camera_keys: Sequence[str],
    Ta: int,
    execution_horizon: int,
    act_dim: int,
    actor_rng: Optional[jnp.ndarray],
    flatten: bool = False,
) -> jnp.ndarray:
    """Shared by every critic variant's own {mse,categorical,hlgauss}_
    critic_loss_fn - the TD target's own bootstrap action, either
    chunked-SARSA (default - batch.next_actions, the real recorded next
    action) or standard off-policy bootstrapping on the actor's OWN
    predicted next action, when actor_network_model is given. Matches
    xqc/agents/xqc/critic.py's own `dist = actor(batch.next_observations);
    next_actions = dist.sample(seed=key)` as closely as this architecture
    allows: LIVE actor params (no separate target-actor concept - xqc has
    none either, only target_critic is Polyak-averaged - see conversation
    history), but this network needs its raw next_observations
    vision-encoded first (unlike xqc's state-only actor) and returns a
    whole Ta-length action chunk needing the same [:, :execution_horizon]
    slice xqc/agents/chunking/actor.py's own update_actor already does
    elsewhere (see conversation history: the reference's own
    (To-1)-steps-before-"now" alignment was dropped when the IL/RL
    datasets were unified) - not a single next action directly. Whichever
    branch runs, the caller's own
    jax.lax.stop_gradient on target_q/next_q cuts off any gradient flow
    through this, so it can never train the actor from here.

    flatten=True reproduces mlp/xqc_style's own _flatten(x) ==
    x.reshape(x.shape[0], -1) on the result (their critic consumes flat
    (obs_window, action_chunk) inputs); flatten=False (transformer) keeps
    the (B, execution_horizon, act_dim) sequence shape VMapTransformerCritic
    consumes directly.

    Lives here, alongside sarsa_td_target, rather than in vision_utils.py -
    despite calling encode_observation_from_models - because this is TD-
    target arithmetic (the bootstrap half of the target), not a vision
    concern; see this module's own docstring. Was previously duplicated
    near-verbatim in each of mlp/xqc_style/transformer's own critic.py,
    then briefly centralized in vision_utils.py before moving here - see
    conversation history."""
    if actor_network_model is None:
        next_actions = batch.next_actions
    else:
        actor_next_obs, _ = encode_observation_from_models(
            actor_vision_models, {cam: m.params for cam, m in actor_vision_models.items()},
            actor_obs_encoder_model, actor_obs_encoder_model.params,
            batch.next_observations, rescale_images(batch.next_images), actor_obs_offset, actor_obs_scale, actor_camera_keys,
            train=False, random_crop=False, crop_shape=(0, 0), crop_key=None,
        )
        batch_size = batch.next_observations.shape[0]
        # actor_network_model's own batch_stats (explicit request - see
        # conversation history, and xqc/agents/chunking/actor.py's own
        # infer() docstring for the identical pattern): threaded through
        # (read-only - no `mutable` requested, matching infer()'s own
        # eval-mode call) only when actor_network_model.batch_stats is not
        # None - a None-vs-pytree structural check resolved at jax.jit trace
        # time. Every actor without BatchNorm (batch_stats always None -
        # matches Model.create's own `variables.get("batch_stats", None)`)
        # traces through the exact same code path as before this was added.
        # This is the ACTOR-BOOTSTRAP branch (actor_network_model given -
        # standard off-policy TD-target, as opposed to chunked-SARSA's own
        # batch.next_actions above) - xqc/agents/actors/xqc/xqcfd.py's XQCFD
        # is the first actor to need this, via conf/agent/xqcfd_chunked.yaml's
        # own pretrain_use_sarsa=false (matching xqc/agents/xqc/xqc_learner.py's
        # own critic, which ALWAYS bootstraps off the live actor, never SARSA).
        actor_variables = {"params": actor_network_model.params}
        if actor_network_model.batch_stats is not None:
            actor_variables["batch_stats"] = actor_network_model.batch_stats
        sampled_next_action = actor_network_model.apply_fn.apply(
            actor_variables, actor_rng, (batch_size, Ta, act_dim), actor_next_obs, method="sample"
        )
        next_actions = sampled_next_action[:, :execution_horizon]
    return next_actions.reshape(next_actions.shape[0], -1) if flatten else next_actions


def entropy_from_log_probs(log_probs: jnp.ndarray) -> jnp.ndarray:
    """Per-sample entropy of a categorical distribution given as log-probs
    (..., num_bins) -> (...,). A collapsing (near-zero) or maxed-out (near
    log(num_bins), uniform) entropy both indicate the predicted
    distribution stopped carrying useful information - something a decoded
    scalar Q mean alone would hide."""
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def fraction_at_boundary(
    values: jnp.ndarray, min_v: float, max_v: float, atol: float = 1e-3
) -> jnp.ndarray:
    """Fraction of `values` sitting within `atol` of either support edge -
    a large fraction here means the critic's fixed [min_v, max_v] value
    support is miscalibrated for the actual return scale (predictions
    saturating at the boundary instead of representing the true value)."""
    at_min = jnp.abs(values - min_v) < atol
    at_max = jnp.abs(values - max_v) < atol
    return jnp.mean(at_min | at_max)


def split_dense_and_batchnorm_norms(params) -> Dict[str, jnp.ndarray]:
    """Splits a critic's params into {"dense_norm": ..., "batchnorm_norm":
    ...} - see this module's own docstring for why. Path-matching (just
    "does 'BatchNorm' appear in this leaf's path") mirrors how
    xqc/networks/common.py's norm_network itself identifies BatchNorm
    layers to SKIP, so this stays consistent with that file if its naming
    ever changes. Every leaf in a VMapCritic/Critic's params is either a
    Dense kernel/bias or a BatchNorm scale/bias - no other param types
    exist in this architecture - so a plain binary split covers the whole
    tree exactly, no leaves silently dropped."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    bn_leaves = [v for k, v in flat.items() if "BatchNorm" in k]
    dense_leaves = [v for k, v in flat.items() if "BatchNorm" not in k]
    return {
        "dense_norm": optax.global_norm(dense_leaves) if dense_leaves else jnp.array(0.0),
        "batchnorm_norm": optax.global_norm(bn_leaves) if bn_leaves else jnp.array(0.0),
    }


_LEAF_PARAM_NAMES = ("kernel", "bias", "scale")


def per_layer_norms(params) -> Dict[str, jnp.ndarray]:
    """One norm per named layer/module, for any of the three critic
    architectures - e.g. each hidden block's Dense kernel, each BatchNorm
    layer's scale+bias combined, the predictor/output layer, or (for the
    transformer) each attention projection/FFN Dense/LayerNorm/standalone
    param (action_pos_bias, cls_token). See this module's own docstring
    for why this exists (per-layer, not just per-bucket).

    Grouping rule: a leaf whose own name is one of the generic
    "kernel"/"bias"/"scale" gets grouped with its sibling(s) under the
    same parent module (a Dense layer's kernel+bias count as ONE layer,
    not two; a BatchNorm/LayerNorm's scale+bias likewise); anything else
    (a standalone raw param like the transformer's action_pos_bias/
    cls_token, which has no such sibling) keeps its own full path as its
    own group - so every leaf ends up in exactly one group, none dropped.
    The outermost VMap wrapper's own name (e.g. "VmapCritic_0",
    "VmapTransformerCritic_0") is stripped since it's identical for every
    leaf and would just be visual noise in every key - but ONLY when it
    really is a single uniform wrapper shared by every leaf (checked below
    via top_level_keys, not assumed): every critic architecture has exactly
    one (the vmap-ensemble child), and so does e.g. the MF/DP/FM/MIP/AF/
    MIP-Flow actors' own single `self.net = ChiUNet(...)` (top level is
    just "net"). This is NOT true of every actor, though - xqcfd's own
    hetstat actor builds several distinct named top-level children directly
    (MLP_0/StationaryFeatures_0/HetStatMean/HetStatCovSqrt/
    BatchNormEmbedder_0), and NF's `self.couplings = [ChiUNet(...), ...]`
    produces one top-level key per coupling layer (couplings_0/
    couplings_1/...) - blindly stripping the first segment there would
    merge or mislabel genuinely distinct layers instead of just removing
    noise, so it's skipped whenever more than one top-level key exists."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    top_level_keys = {path.split("/", 1)[0] for path in flat}
    strip_prefix = len(top_level_keys) == 1
    groups: Dict[str, List[jnp.ndarray]] = {}
    for path, leaf in flat.items():
        parts = path.split("/")[1:] if strip_prefix else path.split("/")
        if parts and parts[-1] in _LEAF_PARAM_NAMES:
            layer_path = "/".join(parts[:-1])
        else:
            layer_path = "/".join(parts)
        groups.setdefault(layer_path, []).append(leaf)
    return {layer_path: optax.global_norm(leaves) for layer_path, leaves in groups.items()}


def layer_norm_ratios(
    info: Dict[str, jnp.ndarray], init_layer_norms: Dict[str, float], prefix: str = "layer_norm"
) -> Dict[str, float]:
    """Turns the per-step `info["layer_norm/<path>"]` values `jit_critic_step`
    already logged into `"layer_norm_ratio/<path>"` values, relative to each
    layer's OWN init norm (`init_layer_norms`, captured once at construction
    - see each Learner's own __init__). Same reasoning as weight_norm_ratio,
    just one ratio per layer instead of one total - factored out here so the
    three Learner classes don't each reimplement this same loop.

    prefix (default "layer_norm"): the info key family to read/write, e.g.
    "actor_layer_norm" reads `info["actor_layer_norm/<path>"]` and writes
    `"actor_layer_norm_ratio/<path>"` - lets ChunkedPolicyLearner reuse this
    for the actor's own per-layer norms without colliding with the critic's
    identically-shaped keys once critic_info/actor_info get merged."""
    return {
        f"{prefix}_ratio/{k}": float(info[f"{prefix}/{k}"]) / max(v, 1e-8)
        for k, v in init_layer_norms.items()
    }
