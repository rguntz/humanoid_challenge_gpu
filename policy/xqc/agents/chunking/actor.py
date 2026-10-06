"""The actor's own train/eval/infer functions, factored out of
xqc/agents/chunking/chunked_learner.py (see conversation history) to match
xqc/agents/xqc/xqc_learner.py's own convention: that file imports
update_actor/update_actor_bc from xqc/agents/xqc/actor.py rather than
defining them inline, keeping xqc_learner.py itself as a thin orchestrator.
This file is the chunked-action-family equivalent - update_actor_bc is
literally the same rename xqc_learner.py already uses for its own BC-mode
actor update.

val_loss/infer live here too rather than in a third location (unlike
xqc_learner.py, which keeps its own inference function - policies.
sample_actions_with_log_probs - alongside its one hardcoded network
architecture in xqc/networks/policies.py): chunked_learner.py has no single
network definition of its own (each agent brings its own - DP/FM/MF/... -
via network_cls), so val_loss/infer are generic across whichever one is
plugged in, exactly like update_actor_bc - all three are "the actor's full
set of operations regardless of concrete architecture," which is what this
file now holds together.

encode_condition lives in xqc/agents/common/obs_encoder.py instead of here
(see conversation history): it's condition-construction, not action
prediction, and a future critic needs the exact same function, not a copy -
so it lives in the neutral module both depend on. apply_model_gradient
still lives here for now (a separate, not-yet-resolved question - see
conversation history).

update_actor_bc/val_loss take one `batch: ChunkedBatch` (xqc.replay_buffer.
chunked_qc_buffer.ChunkedBatch - see conversation history: the former
xqc.agents.common.dataset.ChunkBatch is retired as part of unifying the IL
and RL datasets onto one shared Dataset/ChunkedBatch) instead of separate
act_window/obs_window/images positional args - mirrors xqc/agents/xqc/actor.py's
own update_actor/update_actor_bc, which take one `batch: Batch` rather than
`batch.observations`/`batch.actions` spelled out as separate parameters.
"""
import functools
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import optax

from xqc.agents.common.obs_encoder import encode_observation_from_models
from xqc.networks.common import Model, norm_network
from xqc.networks.distribution_utils import kl_divergence
from xqc.replay_buffer.chunked_qc_buffer import ChunkedBatch


def apply_model_gradient(model: Model, grads) -> Model:
    """Applies an externally-computed gradient to `model`'s own optimizer -
    mirrors what Model.apply_gradient does internally, but takes an
    already-computed gradient instead of computing one itself, since here
    the gradient comes from one joint computation shared across three
    separate Models (see update_actor_bc's own docstring)."""
    updates, new_opt_state = model.tx.update(grads, model.opt_state, model.params)
    new_params = optax.apply_updates(model.params, updates)
    return model.replace(step=model.step + 1, params=new_params, opt_state=new_opt_state)


def _apply_joint_gradients(
    network_model: Model,
    vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    network_grads,
    vision_grads,
    obs_encoder_grads,
    new_vision_batch_stats,
    grad_clip_norm: Optional[float],
    new_network_batch_stats=None,
):
    """Applies a jointly-computed network+vision+obs_encoder gradient to
    each of the three Models - shared by update_actor_bc/update_actor (see
    their own docstrings for why the gradient must be computed jointly in
    the first place).

    grad_clip_norm, when set, clips the GLOBAL norm across all three
    gradient trees TOGETHER (network + every vision camera + obs_encoder,
    as one combined pytree) before any of them is applied - matching the
    reference DiffusionPolicy implementation exactly (one combined
    torch.optim.AdamW over vision+UNet's parameters together, and one
    single torch.nn.utils.clip_grad_norm_ call over that same combined
    parameter set), NOT three independent per-component clips. The AdamW
    step itself doesn't care how params are grouped into optimizers (Adam
    tracks per-parameter moments independently of grouping - see
    ChunkedPolicyLearner's own _make_tx), but clipping is inherently a
    cross-parameter operation, so this is the one place the three
    separately-instantiated optimizers genuinely need to cooperate rather
    than act independently - see conversation history. Every existing
    agent config currently sets grad_clip_norm=null (see conf/agent/
    training/{vision,state}.yaml), so this is presently a no-op everywhere;
    the mechanism still has to be correct for whenever it isn't.

    new_network_batch_stats (explicit request - see conversation history):
    the network's OWN updated batch_stats collection, when its network_cls
    uses BatchNorm (currently only xqc/agents/actors/xqc/xqcfd.py's XQCFD -
    every other network in this family has none, batch_stats is always None
    for them, matching Model.create's own `variables.get("batch_stats",
    None)`). None (default) leaves network_model's batch_stats untouched
    (reproduces today's behavior exactly for every other agent) - mirrors
    new_vision_batch_stats' own per-camera replace just below, just for the
    one network_model instead of N vision cameras."""
    if grad_clip_norm is not None:
        combined_grads = (network_grads, vision_grads, obs_encoder_grads)
        clip_tx = optax.clip_by_global_norm(grad_clip_norm)
        combined_grads, _ = clip_tx.update(combined_grads, clip_tx.init(combined_grads))
        network_grads, vision_grads, obs_encoder_grads = combined_grads

    new_network_model = apply_model_gradient(network_model, network_grads)
    if new_network_batch_stats is not None:
        new_network_model = new_network_model.replace(batch_stats=new_network_batch_stats)
    new_obs_encoder_model = apply_model_gradient(obs_encoder_model, obs_encoder_grads)
    new_vision_models = {
        cam: apply_model_gradient(vision_models[cam], vision_grads[cam]).replace(batch_stats=new_vision_batch_stats[cam])
        for cam in vision_models
    }
    return new_network_model, new_vision_models, new_obs_encoder_model



@functools.partial(jax.jit, static_argnums=(3, 12, 13, 14, 16, 17, 18, 19, 20, 21, 26, 28, 29, 30, 31))
def update_actor(
    network_model: Model,
    vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    compute_q_fn,
    critic_model: Model,
    critic_vision_models: Dict[str, Model],
    critic_obs_encoder_model: Model,
    obs_window, images,
    obs_offset, obs_scale, rng,
    camera_keys, random_crop, crop_shape, crop_key,
    Ta, execution_horizon, act_dim,
    grad_clip_norm: Optional[float] = None,
    wasserstein_regularization: Optional[float] = None,
    actor_name: str = "",
    prior_network_model: Optional[Model] = None,
    prior_vision_models: Optional[Dict[str, Model]] = None,
    prior_obs_encoder_model: Optional[Model] = None,
    in_reference_region=None,
    mask_reference_region: bool = False,
    offline_mask=None,
    wasserstein_offline_only: bool = False,
    actor_weight: float = 1.0,
    kl_regularization: Optional[float] = None,
    use_weight_norm: bool = False,
):
    """DDPG-style actor update: sample an action from the network
    (differentiably - MeanFlow's own .sample() is one forward pass, not an
    iterative chain, which is what makes backprop-through-sampling cheap
    here - see conversation history), query the critic's induced Q-value at
    that action, and maximize it. No entropy/KL regularization (unlike
    xqc/agents/xqc/actor.py's own update_actor) - these generative
    networks have no tractable log_prob to regularize with at all (see
    chunked_learner.py's own _rl_not_implemented_msg).

    wasserstein_regularization (None/0 = off, else the fixed loss
    coefficient - no learned auto-tuned temperature here, unlike
    xqc/agents/xqc/actor.py's own kl_regularization/temperature() pair, by
    explicit request - see conversation history): a Wasserstein-2 UPPER
    BOUND regularizer, in place of the KL/entropy term that reference uses
    (this class's own generative actors have no tractable log_prob to
    build a real KL from - the same reason update_actor never had entropy
    regularization at all until now). Implements the shared-noise
    distillation trick from stationarity/wasserstein.py's own module
    docstring: for a FIXED shared z, ||mu_w(s,z) - mu_theta(s,z)||^2
    (mu_w = this live actor being trained, "student"; mu_theta =
    prior_network/prior_vision/prior_obs_encoder, the frozen BC-pretrained
    snapshot taken at the offline->online transition, "teacher") upper-
    bounds W_2(pi_w(.|s), pi_theta(.|s))^2 for ANY shared coupling,
    regardless of how (non-)optimal that particular coupling is - so
    minimizing it still pulls the online-RL-updated actor's conditional
    action distribution back toward its BC prior, exactly like KL
    regularization does in the reference, just without needing a tractable
    density. The "shared z" part costs nothing extra to arrange here:
    MeanFlow's own .sample(rng, shape, cond) draws its OWN z =
    jax.random.normal(rng, shape) internally as a pure function of
    (rng, shape) - calling both the student's and teacher's .sample() with
    the SAME rng/shape (only cond differs, since each encodes obs through
    its own vision/obs_encoder) already gives them the identical z, with
    no extra plumbing needed.

    mask_reference_region (static, config-controlled ablation knob - same
    flag and same underlying batch.in_reference_region field as
    update_actor_bc's own, see that function's own docstring and
    chunked_qc_buffer.py's ChunkedBatch docstring): when True, weights the
    Q-maximization loss below by in_reference_region instead of taking a
    plain mean - i.e. samples whose EXECUTED (execution_horizon-length)
    action chunk is guaranteed-real given this call's own Ta get full
    weight, near-boundary samples get zeroed out, matching
    update_actor_bc's own "don't train on the padded region" intent for
    the RL actor update too (explicit request - see conversation history).
    in_reference_region must be passed fresh each call (the caller's own
    critic_batch.in_reference_region - see chunked_learner.py's own _update)
    since it's a traced per-sample array, not something bindable once at
    construction. Default False reproduces today's behavior exactly (plain
    mean, every sample weighted equally).

    offline_mask/wasserstein_offline_only (explicit request - see
    conversation history): a SEPARATE masking axis from mask_reference_region
    above, applied ONLY to wasserstein_penalty below (never to the
    Q-maximization term - regularizing the actor toward its frozen BC prior
    is the concern here, not the RL objective itself). offline_mask is a
    per-sample (B,) array, 1.0 for a row drawn from the OFFLINE buffer and
    0.0 for a row drawn from the ONLINE replay buffer - built by the caller
    (train_parallel_chunking.py, at the exact point it concatenates
    online+offline batches into one critic_batch, where online_batch_size/
    offline_batch_size are already known plain ints) and passed fresh each
    call, same convention as in_reference_region. wasserstein_offline_only
    (static): False (default, reproduces today's behavior - every sample in
    the mixed batch counts) regularizes the actor everywhere the Q-term is
    also trained; True restricts the distillation pull to offline-buffer
    rows only. Rationale: pulling the actor toward prior_network at an
    ONLINE (possibly off-the-BC-manifold) state asks the frozen teacher to
    extrapolate to a state it may never have seen during BC pretraining -
    prior_network can only "hallucinate" a plausible-looking action there,
    which is not a trustworthy target to regularize the online actor
    against. When both this and mask_reference_region are active, a sample
    must satisfy BOTH to count (AND, not OR) - the two are orthogonal
    concerns (which buffer a row came from vs. how much of its own action
    window is real vs. near-boundary padding) that compose rather than
    override each other.

    loss_info["offline_prior_mse"] (explicit request, debugging-only - see
    this function's own body for the full rationale): computed whenever
    actor_name=="MFLearner" and prior_network_model is usable and
    offline_mask has any offline rows this step - INDEPENDENT of
    wasserstein_regularization (on or off) and of wasserstein_offline_only's
    own value. A plain per-element MSE between the live actor and its
    frozen prior_network, restricted to offline-buffer rows only
    (guaranteed in-distribution for prior_network). Exists so ANY online
    run - not just ones with wasserstein_regularization on - can check how
    far the actor has actually drifted from its BC prior on states that
    prior is trustworthy on; also lets an online+offline run's own
    wasserstein_penalty (which mixes that in-distribution drift with the
    teacher's own out-of-distribution extrapolation error on online states)
    be sanity-checked against this narrower, always-well-posed question -
    never added to actor_loss, logging only.

    actor_weight (explicit request - debugging ablation, see conversation
    history): scales ONLY the Q-maximization term, applied right where it's
    first computed below (BEFORE the wasserstein branch adds its own
    separately-coefficient term), i.e. `actor_loss = actor_weight *
    (-q_values...) + wasserstein_regularization * wasserstein_penalty`.
    Default 1.0 reproduces today's behavior exactly. actor_weight=0.0 with
    wasserstein_regularization on isolates PURE distillation-toward-prior
    with the RL objective completely removed - a sanity check on the
    distillation mechanism itself, decoupled from whatever the Q-term is
    doing: does the actor's success rate stay near its BC-pretrained
    starting point, and does offline_prior_mse actually converge toward 0,
    when nothing is pulling the actor away from prior_network at all? If
    not, the collapse seen in wasserstein-regularized runs isn't purely a
    "Q-term overpowers a working distillation pull" story.

    MF-only for now (explicit request - the shared-z trick generalizes to
    any actor whose .sample() is a deterministic, rng-seeded map, but
    that's unverified for DP/FM/MIP/NF/BC/AF today). Deliberately NOT
    silently skipped for other actor types when the flag is set - crashes
    loudly instead (see actor_name check below and
    ChunkedPolicyLearner.__init__'s own matching fail-fast check), so a
    config mistake (accidentally turning this on for e.g. DP) is caught
    immediately rather than silently training without the intended
    regularization.

    compute_q_fn/critic_model/critic_vision_models/critic_obs_encoder_model
    are the CRITIC's own pieces, passed in as explicit pytree arguments
    (matching xqc/agents/xqc/actor.py's own update_actor, which takes
    `critic` as a direct Model argument, not through a stateful Python
    wrapper) - the caller's job to pass the critic's LIVE params (not
    target), fresh each call, so nothing goes stale under jit. compute_q_fn
    is one of xqc/agents/critics/{mlp,xqc_style,transformer}/critic.py's
    own jit-decorated compute_q free functions, already bound (via
    functools.partial, once at construction time - see chunked_learner.py's
    own __init__) with that critic's own obs_offset/obs_scale/camera_keys -
    so this function only needs to pass it the three pieces that genuinely
    vary per call (critic_model/critic_vision_models/critic_obs_encoder_model,
    obs_window/images, and the freshly-sampled executed_action below), not
    re-thread the critic's own static normalization config through every
    call (a plain Python callable is still a valid static/hashable jit
    argument regardless of what's bound into it) - this is how a single
    update_actor stays swappable across critic architectures without
    needing to import any of them itself (see conversation history for why
    xqc's own reference never needed this: one fixed critic architecture,
    no swapping).

    The network's raw .sample() output is (B, Ta, act_dim), windowed the
    same way as BC training now - starting AT "now" (t), matching the
    critic's own actions[t:t+h] convention (chunked_qc_buffer.py) exactly -
    see conversation history: the reference's own (To-1)-steps-before-"now"
    alignment was dropped when the IL/RL datasets were unified. The critic
    expects (B, execution_horizon, act_dim) - slicing [:, :execution_horizon]
    takes just the executed prefix of the Ta-length window (the same slice
    ChunkExecutor.sample_actions already does for real deployment, see its
    own docstring). The sliced action is fed to the critic WITHOUT denormalizing
    (unlike infer()'s own wrapper,
    which does `* action_scale + action_offset` for real env use) - the
    raw .sample() output is already in the same normalized units the
    critic was trained on, and actor/critic share the same
    action_offset/action_scale (see conversation history), so no further
    conversion is needed here.

    kl_regularization (None/0 = off, else the fixed loss coefficient -
    explicit request, see conversation history): the analytic-KL analogue
    of wasserstein_regularization above, for xqc/agents/actors/xqc/xqcfd.py's
    XQCFD actor specifically (gated on actor_name=="XQCFDLearner" - same
    fail-fast-if-mismatched philosophy as wasserstein_regularization's own
    actor_name=="MFLearner" gate just below, see that check's own comment).
    Unlike wasserstein_regularization, this needs no shared-noise trick at
    all: hetstat's own Gaussian-tanh distribution has a closed-form KL
    (xqc.networks.distribution_utils.kl_divergence) between the live actor's
    distribution and prior_network's own (the same frozen BC-pretrained
    snapshot wasserstein_regularization uses, reused unchanged here - see
    train_parallel_chunking.py's own first_online_transition block). Matches
    xqc/agents/xqc/actor.py's own update_actor with kl_regularization="prior"
    exactly: `actor_loss = (temperature()*kl - q_values).mean()`, just with
    a fixed coefficient in place of that reference's own learned
    temperature() Model (xqcfd's own conf/agent/xqcfd.yaml sets temp_lr=0.0,
    i.e. a FIXED temperature in the reference too - "Crucial -> Fixed
    temperature for xqcfd" - so this is not a simplification relative to
    the config actually being reproduced). kl_regularization="uniform"
    (entropy-only, no prior) is NOT implemented - xqcfd.yaml never uses it,
    and MF-family's own wasserstein_regularization has no such alternate
    mode either.

    When kl_regularization is set, the live actor is queried via
    XQCFD's own `.distribution()` method (NOT `.sample()`, unlike every
    other branch of this function) so this function can read `dist.log_prob`/
    feed `dist`+prior_dist into kl_divergence - the SAME dist.sample() call
    also supplies executed_action for the Q-maximization term below (one
    forward pass covers both, mirroring xqc/agents/xqc/actor.py's own
    `dist = actor(...); actions = dist.sample(seed=key)`), rather than a
    second redundant call through `.sample()`."""
    if kl_regularization:
        if actor_name != "XQCFDLearner":
            raise ValueError(
                f"kl_regularization is only implemented for XQCFD (got actor_name={actor_name!r}) "
                "- see conversation history. Either turn kl_regularization off for this agent, "
                "or verify+extend the analytic-KL path in update_actor for this actor type first."
            )
        if prior_network_model is None:
            raise ValueError(
                "kl_regularization=True needs prior_network_model (the frozen BC-pretrained teacher "
                "snapshot) - got None. It's only populated once the offline->online transition has "
                "actually run (see train_parallel_chunking.py's own first_online_transition block)."
            )
    if wasserstein_regularization:
        if actor_name != "MFLearner":
            raise ValueError(
                f"wasserstein_regularization is only implemented for MF (got actor_name={actor_name!r}) "
                "- see conversation history. Either turn wasserstein_regularization off for this agent, "
                "or verify+extend the shared-z coupling trick in update_actor for this actor type first."
            )
        if prior_network_model is None or prior_vision_models is None or prior_obs_encoder_model is None:
            raise ValueError(
                "wasserstein_regularization=True needs prior_network_model/prior_vision_models/"
                "prior_obs_encoder_model (the frozen BC-pretrained teacher snapshot) - got None. "
                "These are only populated once the offline->online transition has actually run "
                "(see train_parallel_chunking.py's own first_online_transition block)."
            )

    vision_params = {cam: m.params for cam, m in vision_models.items()}
    has_batch_stats = network_model.batch_stats is not None

    def actor_loss_fn(network_params, vision_params, obs_encoder_params):
        obs, new_vision_batch_stats = encode_observation_from_models(
            vision_models, vision_params, obs_encoder_model, obs_encoder_params,
            obs_window, images, obs_offset, obs_scale, camera_keys, train=True,
            random_crop=random_crop, crop_shape=crop_shape, crop_key=crop_key,
        )

        batch_size = obs_window.shape[0]
        act_shape = (batch_size, Ta, act_dim)
        network_variables = {"params": network_params}
        if has_batch_stats:
            network_variables["batch_stats"] = network_model.batch_stats

        # kl_regularization needs the actual distribution object (dist.
        # log_prob/kl_divergence, not just a sample from it) - queried via
        # XQCFD's own .distribution() method instead of .sample(), and
        # dist.sample() below supplies executed_action for the
        # Q-maximization term too, so this ONE forward pass covers both -
        # see this function's own docstring. Every other actor_name takes
        # the .sample()-only path exactly as before this branch existed.
        dist = None
        if kl_regularization:
            apply_kwargs = {"mutable": ["batch_stats"]} if has_batch_stats else {}
            result = network_model.apply_fn.apply(
                network_variables, obs, training=True, method="distribution", **apply_kwargs
            )
            if has_batch_stats:
                dist, network_mutated = result
                new_network_batch_stats = network_mutated["batch_stats"]
            else:
                dist = result
                new_network_batch_stats = None
            sampled_action = dist.sample(seed=rng).reshape(act_shape)
        elif has_batch_stats:
            sampled_action, network_mutated = network_model.apply_fn.apply(
                network_variables, rng, act_shape, obs, method="sample", mutable=["batch_stats"]
            )
            new_network_batch_stats = network_mutated["batch_stats"]
        else:
            sampled_action = network_model.apply_fn.apply(network_variables, rng, act_shape, obs, method="sample")
            new_network_batch_stats = None
        executed_action = sampled_action[:, :execution_horizon]

        q_values = compute_q_fn(
            critic_model, critic_vision_models, critic_obs_encoder_model,
            obs_window=obs_window, images=images, action=executed_action,
        )
        # compute_q_fn returns the RAW per-ensemble-member Q values
        # (n_critics, B), not already aggregated - matches
        # xqc/agents/xqc/actor.py's own update_actor exactly, which does
        # this same jnp.min right after its own bare critic(...) call (see
        # conversation history: min-over-ensemble is an actor-loss-level
        # choice, not something baked into how a critic variant decodes
        # its own output).
        q_values = jnp.min(q_values, axis=0)
        if mask_reference_region:
            denom = jnp.maximum(in_reference_region.sum(), 1.0)
            actor_loss = -(q_values * in_reference_region).sum() / denom
        else:
            actor_loss = -q_values.mean()
        actor_loss = actor_weight * actor_loss
        loss_info = {"actor_loss": actor_loss, "q_values": q_values.mean()}

        # offline_action_{mean,max,min} / online_action_{mean,max,min}
        # (explicit request - see conversation history): plain output
        # statistics of the actor's own RAW sampled_action (full Ta window,
        # same convention as wasserstein_penalty/offline_prior_mse below) -
        # deliberately NORMALIZED/network-output units, NOT rescaled to
        # real action units (unlike ChunkedPolicyLearner.compute_action_mse/
        # infer()'s own convention) - explicit request: the point is to see
        # the network's own raw output scale directly, to catch a scaling
        # issue in the network itself, which real-unit rescaling would
        # mask. ChunkedPolicyLearner.compute_action_range's own
        # offline-phase counterpart inverts infer_fn's own rescale back to
        # normalized units for the exact same reason, so both halves of
        # the bare-key-merged curve (train_parallel_chunking.py, same
        # convention as "MSE"/offline_prior_mse) stay in the same units.
        # Split by offline_mask - NOT a distance-to-teacher metric like
        # offline_prior_mse, and NOT gated on actor_name=="MFLearner" or
        # prior_network_model (needs neither - this only reads the
        # student's own output). Purpose: directly check whether the
        # policy's own action magnitude stays bounded (imitation-constrained)
        # on offline/demo-buffer rows vs. grows unconstrained on
        # online-buffer rows once wasserstein_offline_only leaves the
        # latter with no distillation pull at all - see conversation
        # history for the "low offline_prior_mse yet collapsed eval
        # performance" investigation this is following up on. Only ever
        # populated when offline_mask is not None (same as
        # offline_prior_mse - i.e. never during pre_train=True/BC calls,
        # which don't build one - update_actor_bc never touches this
        # function at all). Degrades gracefully rather than being omitted
        # if a group is literally empty in a given batch (unlike
        # offline_prior_mse's own full omission) - offline_data_ratio > 0
        # guarantees both groups are populated in every run this is
        # actually used for, so this stays a plain jit-friendly computation
        # rather than a Python-level branch on a traced value.
        if offline_mask is not None:
            online_mask = 1.0 - offline_mask
            offline_weights = offline_mask[:, None, None]
            online_weights = online_mask[:, None, None]
            offline_denom = jnp.maximum(offline_mask.sum(), 1.0) * Ta * act_dim
            online_denom = jnp.maximum(online_mask.sum(), 1.0) * Ta * act_dim
            loss_info["offline_action_mean"] = (sampled_action * offline_weights).sum() / offline_denom
            loss_info["online_action_mean"] = (sampled_action * online_weights).sum() / online_denom
            offline_masked_for_max = jnp.where(offline_weights > 0, sampled_action, -jnp.inf)
            offline_masked_for_min = jnp.where(offline_weights > 0, sampled_action, jnp.inf)
            online_masked_for_max = jnp.where(online_weights > 0, sampled_action, -jnp.inf)
            online_masked_for_min = jnp.where(online_weights > 0, sampled_action, jnp.inf)
            loss_info["offline_action_max"] = offline_masked_for_max.max()
            loss_info["offline_action_min"] = offline_masked_for_min.min()
            loss_info["online_action_max"] = online_masked_for_max.max()
            loss_info["online_action_min"] = online_masked_for_min.min()

        # teacher_action / offline_prior_mse (explicit request - see
        # conversation history): computed whenever prior_network_model is
        # actually usable, INDEPENDENT of wasserstein_regularization - this
        # used to live only inside the `if wasserstein_regularization:`
        # block below (it originally just reused that block's own
        # teacher_action), but the diagnostic itself doesn't need the loss
        # term to be active. actor_name=="MFLearner" gate: teacher_action
        # relies on the exact same "shared z" trick (same rng/act_shape for
        # student and teacher's own .sample() calls) wasserstein_regularization's
        # fail-fast check already restricts to MF - unverified for other
        # actor types (see this function's own docstring) - so this stays
        # silently skipped (not a hard error) for them, since it's a
        # diagnostic, not a correctness-critical training path. By the time
        # update_actor ever runs at all (pre_train=False, i.e. only online,
        # only after the offline->online transition has already snapshotted
        # prior_network/etc unconditionally - see train_parallel_chunking.py's
        # own first_online_transition block), prior_network_model is only
        # ever None for an agent built before that attribute existed (old
        # checkpoints - see chunked_learner.py's own getattr default).
        teacher_action = None
        if actor_name == "MFLearner" and prior_network_model is not None:
            # Teacher encodes obs through its OWN frozen vision/obs_encoder
            # (never the live/training ones) - matches xqc/agents/xqc/
            # actor.py's own prior_dist computation (prior_actor.apply
            # with prior_actor's own params), and chunked_critic_target.py's
            # own bootstrap_next_actions pattern for the same reason: the
            # teacher's conditional distribution pi_theta(.|s) is only
            # well-defined w.r.t. ITS OWN encoding pipeline.
            prior_vision_params = {cam: m.params for cam, m in prior_vision_models.items()}
            prior_obs, _ = encode_observation_from_models(
                prior_vision_models, prior_vision_params, prior_obs_encoder_model, prior_obs_encoder_model.params,
                obs_window, images, obs_offset, obs_scale, camera_keys, train=False,
                random_crop=False, crop_shape=crop_shape, crop_key=None,
            )
            # SAME rng as the student's own sampled_action above - see this
            # function's own docstring: this is the entire "shared z"
            # coupling, no separate z variable needed since .sample()'s own
            # internal jax.random.normal(rng, shape) draw is a pure
            # function of (rng, shape).
            # prior_network_model's own batch_stats (explicit request - see
            # conversation history: this call site only ever threaded
            # through "params" before, an unnoticed gap while every agent
            # that reaches here used norm="group" (no batch_stats collection
            # at all) - norm="batch"/"batch_ref" needs it too, same
            # None-vs-pytree structural check convention as every other
            # Model.apply_fn.apply call in this file (e.g. infer()'s own
            # network_batch_stats threading), otherwise Flax raises
            # ScopeCollectionNotFound trying to read an empty "batch_stats"
            # collection inside the teacher's own train=False forward pass.
            prior_network_variables = {"params": prior_network_model.params}
            if prior_network_model.batch_stats is not None:
                prior_network_variables["batch_stats"] = prior_network_model.batch_stats
            teacher_action = jax.lax.stop_gradient(
                prior_network_model.apply_fn.apply(
                    prior_network_variables, rng, act_shape, prior_obs, method="sample"
                )
            )

            # offline_prior_mse (debugging-only, never added to actor_loss):
            # a wasserstein_penalty-shaped quantity is a mix of two very
            # different things when computed over a mixed online+offline
            # batch - in-distribution drift from the BC prior on offline/
            # demo-covered states, and the teacher's own out-of-distribution
            # EXTRAPOLATION error on online states it was never trained on
            # (prior_network is frozen at the offline->online transition).
            # A large value there is ambiguous between "the policy has
            # drifted" and "the teacher is just unreliable here" - it can't
            # tell those apart. This metric answers a narrower,
            # always-well-posed question instead: on ONLY the
            # offline-buffer rows in this batch (guaranteed in-distribution
            # for prior_network, since that's exactly what it was
            # BC-trained on), how far has the live actor's own action
            # distribution actually moved from its frozen BC prior? A
            # literal per-element MSE (mean over BOTH Ta and act_dim, NOT
            # wasserstein_penalty's own sum-over-act_dim/mean-over-Ta
            # convention - deliberately a plain, standard MSE here since
            # that's what was asked for). None of offline_mask's rows exist
            # (e.g. agent.offline_data_ratio == 0, no offline buffer ever
            # mixed into this batch) -> the key is simply omitted this step
            # (nothing meaningful to report) rather than logging a fake 0.
            if offline_mask is not None:
                offline_denom = jnp.maximum(offline_mask.sum(), 1.0)
                per_sample_mse = ((sampled_action - teacher_action) ** 2).mean(axis=(-2, -1))
                loss_info["offline_prior_mse"] = (per_sample_mse * offline_mask).sum() / offline_denom

        if wasserstein_regularization:
            # teacher_action already computed above - this block's own
            # fail-fast check (actor_name must be "MFLearner",
            # prior_network_model must not be None - see this function's
            # own opening lines) guarantees the block above already ran and
            # populated it, so it's never None here.
            #
            # Over the full Ta-length window (not just executed_action's
            # execution_horizon-length prefix) - matches sampled_action's
            # own full generative output; the point is keeping the actor's
            # whole predicted chunk close to its BC prior, not just the
            # slice that happens to get executed this step. Reduced over Ta
            # first (per_sample_penalty, shape (B,)) so the SAME
            # in_reference_region batch-mask used for the Q-maximization
            # term above also applies here when mask_reference_region=True -
            # this term is a per-sample actor-training loss on the same
            # critic_batch positions as q_values, so leaving it unmasked
            # would keep pulling the actor toward the teacher at
            # near-boundary/out-of-reference-region states even while the Q
            # term itself is excluding them. mask_reference_region=False
            # (default) is numerically identical to the old plain
            # `.mean()` over both axes, since every sample's Ta window is
            # the same length (mean-of-per-sample-means == overall mean).
            #
            # wasserstein_offline_only (see this function's own docstring):
            # a SECOND, independent mask - ANDed with in_reference_region
            # when both are active, since they gate on orthogonal
            # properties of a sample (buffer origin vs. padding) rather
            # than being alternatives to each other.
            per_sample_penalty = ((sampled_action - teacher_action) ** 2).sum(axis=-1).mean(axis=-1)
            wasserstein_mask = None
            if mask_reference_region:
                wasserstein_mask = in_reference_region
            if wasserstein_offline_only:
                wasserstein_mask = offline_mask if wasserstein_mask is None else wasserstein_mask * offline_mask
            if wasserstein_mask is not None:
                denom = jnp.maximum(wasserstein_mask.sum(), 1.0)
                wasserstein_penalty = (per_sample_penalty * wasserstein_mask).sum() / denom
            else:
                wasserstein_penalty = per_sample_penalty.mean()
            actor_loss = actor_loss + wasserstein_regularization * wasserstein_penalty
            loss_info["actor_loss"] = actor_loss
            loss_info["wasserstein_penalty"] = wasserstein_penalty

        if kl_regularization:
            # dist already computed above (the kl_regularization branch of
            # the sampling code just above this function's own docstring
            # guarantees actor_name=="XQCFDLearner", so dist is never None
            # here). prior_dist is queried the SAME way - via .distribution()
            # rather than .sample() - through prior_network_model's own
            # frozen params/batch_stats, never mutated (training=False, no
            # `mutable` requested - a pure read, matching xqc/agents/xqc/
            # actor.py's own prior_dist computation via prior_actor.apply).
            # No shared-rng trick needed (unlike wasserstein_regularization
            # above) - kl_divergence below is the exact closed-form
            # Gaussian KL, not a Monte Carlo estimate.
            prior_vision_params = {cam: m.params for cam, m in prior_vision_models.items()}
            prior_obs, _ = encode_observation_from_models(
                prior_vision_models, prior_vision_params, prior_obs_encoder_model, prior_obs_encoder_model.params,
                obs_window, images, obs_offset, obs_scale, camera_keys, train=False,
                random_crop=False, crop_shape=crop_shape, crop_key=None,
            )
            prior_variables = {"params": prior_network_model.params}
            if prior_network_model.batch_stats is not None:
                prior_variables["batch_stats"] = prior_network_model.batch_stats
            prior_dist = prior_network_model.apply_fn.apply(
                prior_variables, prior_obs, training=False, method="distribution"
            )
            kl = kl_divergence(dist, prior_dist)
            if mask_reference_region:
                denom = jnp.maximum(in_reference_region.sum(), 1.0)
                kl_penalty = (kl * in_reference_region).sum() / denom
            else:
                kl_penalty = kl.mean()
            actor_loss = actor_loss + kl_regularization * kl_penalty
            loss_info["actor_loss"] = actor_loss
            loss_info["kl_penalty"] = kl_penalty

        return actor_loss, (loss_info, new_vision_batch_stats, new_network_batch_stats)

    (_, (infos, new_vision_batch_stats, new_network_batch_stats)), (network_grads, vision_grads, obs_encoder_grads) = jax.value_and_grad(
        actor_loss_fn, argnums=(0, 1, 2), has_aux=True
    )(network_model.params, vision_params, obs_encoder_model.params)

    new_network_model, new_vision_models, new_obs_encoder_model = _apply_joint_gradients(
        network_model, vision_models, obs_encoder_model,
        network_grads, vision_grads, obs_encoder_grads, new_vision_batch_stats,
        grad_clip_norm, new_network_batch_stats=new_network_batch_stats,
    )
    if use_weight_norm:
        new_network_model = norm_network(new_network_model)
    return new_network_model, new_vision_models, new_obs_encoder_model, infos


@functools.partial(jax.jit, static_argnums=(9, 10, 11, 13, 14, 15))
def update_actor_bc(
    network_model: Model,
    vision_models: Dict[str, Model],
    obs_encoder_model: Model,
    batch: ChunkedBatch,
    obs_offset, obs_scale, action_offset, action_scale, rng,
    camera_keys, random_crop, crop_shape, crop_key,
    grad_clip_norm: Optional[float] = None,
    mask_reference_region: bool = False,
    use_weight_norm: bool = False,
):
    """One joint gradient step for network+vision+obs_encoder. Unlike
    xqc_learner.py's actor/critic/temperature (three genuinely separate
    objectives, each already correctly handled by its own single
    Model.apply_gradient call), network/vision/obs_encoder here are one
    connected computational graph feeding a single loss - vision's output
    is part of what the network sees. Their three gradients must all be
    computed from the SAME pre-step snapshot of every component, not
    staggered (e.g. computing vision's gradient against an
    already-updated network would be wrong - see conversation history for
    the full reasoning), so this uses one `jax.value_and_grad` call with
    multiple argnums to get all three gradients from one backward pass,
    then applies each to its own Model via apply_model_gradient.

    mask_reference_region (static, config-controlled ablation knob - see
    xqc/replay_buffer/chunked_qc_buffer.py's ChunkedBatch docstring and
    xqc/agents/common/loss_masking.py): when True, passes
    batch.in_reference_region as the network's own loss() mask, zeroing out
    the BC loss contribution of samples whose Ta-length window needed more
    padding than the reference Diffusion Policy's own SequenceSampler would
    have tolerated - the sampling itself is untouched (every position is
    still drawn, matching the actor/critic coverage-unification fix), only
    the padded-region samples' gradient contribution is ablated. Default
    False reproduces today's behavior exactly (every sample weighted
    equally).

    use_weight_norm (static, explicit request - see conversation history):
    renormalizes network_model's own Dense-layer kernels to unit column-norm
    right after this step's gradient is applied - matches xqc/agents/xqc/
    actor.py's own `if use_weight_norm: new_actor = norm_network(new_actor)`,
    which xqc_style/critic.py's own update_critic already mirrors for the
    critic side of this same reference config. norm_network already skips
    any "HetStat"-named param path (xqc/networks/common.py), so this is safe
    to enable for xqc/agents/actors/xqc/xqcfd.py's XQCFD specifically.
    Default False reproduces today's behavior exactly for every other
    actor (none of DP/FM/MF/MIP/MIP-Flow/NF/BC/AF's own configs set this).

    network_model's own batch_stats (explicit request - see conversation
    history): threaded through only when network_model.batch_stats is not
    None - a None-vs-pytree structural check resolved at jax.jit trace time
    (same convention as offline_mask/in_reference_region elsewhere in this
    file), so every actor without BatchNorm (batch_stats always None -
    matches Model.create's own `variables.get("batch_stats", None)`) traces
    through the exact same code path as before this was added."""
    vision_params = {cam: m.params for cam, m in vision_models.items()}
    mask = batch.in_reference_region if mask_reference_region else None
    has_batch_stats = network_model.batch_stats is not None

    def actor_loss_fn(network_params, vision_params, obs_encoder_params):
        obs, new_vision_batch_stats = encode_observation_from_models(
            vision_models, vision_params, obs_encoder_model, obs_encoder_params,
            batch.observations, batch.images, obs_offset, obs_scale, camera_keys, train=True,
            random_crop=random_crop, crop_shape=crop_shape, crop_key=crop_key,
        )
        act = (batch.actions - action_offset) / action_scale
        network_variables = {"params": network_params}
        if has_batch_stats:
            network_variables["batch_stats"] = network_model.batch_stats
            (loss, infos), network_mutated = network_model.apply_fn.apply(
                network_variables, act, obs, rng, mask=mask, method="loss", mutable=["batch_stats"],
            )
            new_network_batch_stats = network_mutated["batch_stats"]
        else:
            loss, infos = network_model.apply_fn.apply(network_variables, act, obs, rng, mask=mask, method="loss")
            new_network_batch_stats = None
        return loss, (infos, new_vision_batch_stats, new_network_batch_stats)

    (_, (infos, new_vision_batch_stats, new_network_batch_stats)), (network_grads, vision_grads, obs_encoder_grads) = jax.value_and_grad(
        actor_loss_fn, argnums=(0, 1, 2), has_aux=True
    )(network_model.params, vision_params, obs_encoder_model.params)

    new_network_model, new_vision_models, new_obs_encoder_model = _apply_joint_gradients(
        network_model, vision_models, obs_encoder_model,
        network_grads, vision_grads, obs_encoder_grads, new_vision_batch_stats,
        grad_clip_norm, new_network_batch_stats=new_network_batch_stats,
    )
    if use_weight_norm:
        new_network_model = norm_network(new_network_model)
    return new_network_model, new_vision_models, new_obs_encoder_model, infos


@functools.partial(jax.jit, static_argnums=(9, 13, 14))
def val_loss(
    network_model: Model, vision_models: Dict[str, Model], obs_encoder_model: Model,
    network_params, vision_params, obs_encoder_params,
    batch: ChunkedBatch,
    obs_offset, obs_scale, camera_keys, action_offset, action_scale, rng,
    mask_reference_region: bool = False,
    network_accepts_train: bool = True,
):
    """Held-out-episode loss, no gradient step - matches the reference's own
    `with torch.no_grad(): loss = self.model.compute_loss(batch)` over its
    val_dataloader (train_diffusion_unet_lowdim_workspace.py's run()) - same
    loss formula as update_actor_bc's actor_loss_fn, just no state update, and
    train=False for encode_condition (deterministic vision forward, no
    random-crop/batch_stats mutation - irrelevant for lowdim/no-vision
    agents like DP-state anyway, camera_keys=()). network_model/
    vision_models/obs_encoder_model are Model objects (architecture never
    differs between live and EMA); network_params/vision_params/
    obs_encoder_params are separate so the caller can pass either the live
    training params or the EMA params without this function caring which -
    see encode_observation_from_models's own docstring for why one
    calling convention covers both gradient computation and inference.

    mask_reference_region: same ablation knob as update_actor_bc's own (see
    its docstring) - kept in sync so the reported val loss reflects
    whatever the actor is actually being trained against, not a mismatched
    metric.

    network_model's own batch_stats (explicit request - see conversation
    history, and update_actor_bc's own matching docstring): threaded through
    only when network_model.batch_stats is not None (every actor without
    BatchNorm has batch_stats always None, so traces through the exact same
    code path as before this was added). This passes `mutable=["batch_stats"]`
    and then simply DISCARDS the resulting mutation rather than applying it
    to any Model - this function returns only a scalar loss, no Model, so
    "no gradient step, no side effects" (this function's whole point) is
    preserved regardless of which internal BatchNorm mode the loss forward
    pass happened to use.

    network_accepts_train (explicit request - see conversation history: a
    real bug, not just a stale comment - XQCFD's own .loss() (xqc/agents/
    actors/xqc/xqcfd.py) has no `train` parameter at all - the train=False
    fix below was added for MeanFlow's own two-pass batch-norm mechanism
    (MeanFlow.loss()'s own train field docstring) and only ever tested
    against MeanFlow, so it broke XQCFD the first time this function
    actually got reached with a BatchNorm-using XQCFD run - TypeError:
    loss() got an unexpected keyword argument 'train'. Set by the caller
    (ChunkedPolicyLearner's own val_loss_fn construction - type(self).__name__
    != "XQCFDLearner") - True (default) preserves this function's own
    train=False behavior for every other actor; False reproduces XQCFD's
    original pre-train=False behavior exactly (always training=True
    internally, mutation discarded as described above)."""
    obs, _ = encode_observation_from_models(
        vision_models, vision_params, obs_encoder_model, obs_encoder_params,
        batch.observations, batch.images, obs_offset, obs_scale, camera_keys, train=False,
        random_crop=False, crop_shape=(0, 0),
    )
    act = (batch.actions - action_offset) / action_scale
    mask = batch.in_reference_region if mask_reference_region else None
    network_variables = {"params": network_params}
    if network_model.batch_stats is not None:
        network_variables["batch_stats"] = network_model.batch_stats
        # train=False (explicit request - see conversation history and
        # MeanFlow.loss()'s own train field docstring): this is a read-only,
        # no-gradient-step call (this function's whole point), so batch-norm
        # variants should normalize using the RUNNING AVERAGE, not live
        # statistics freshly computed from this held-out val batch - same
        # principle as sample()'s own solver_step, already train=False.
        # mutable=["batch_stats"] stays harmless here even though train=False
        # never actually mutates anything - the result is discarded either way.
        train_kwarg = {"train": False} if network_accepts_train else {}
        (loss, _), _ = network_model.apply_fn.apply(
            network_variables, act, obs, rng, mask=mask, method="loss", mutable=["batch_stats"], **train_kwarg
        )
    else:
        loss, _ = network_model.apply_fn.apply(network_variables, act, obs, rng, mask=mask, method="loss")
    return loss


@functools.partial(jax.jit, static_argnums=(8, 9, 15, 16, 18))
def infer(
    network_model: Model, vision_models: Dict[str, Model], obs_encoder_model: Model,
    network_params, vision_params, obs_encoder_params,
    obs_window, images,
    camera_keys, act_shape, obs_offset, obs_scale, action_offset, action_scale, rng,
    random_crop, crop_shape,
    temperature: float = 1.0,
    network_accepts_temperature: bool = False,
    network_batch_stats=None,
):
    """network_params/vision_params/obs_encoder_params are plain dicts (not
    read off the Model objects), so the caller can pass either the live
    training params or the EMA params without this function caring which -
    see encode_observation_from_models's own docstring. random_crop/
    crop_shape: eval always uses the deterministic center-crop branch of
    encode_condition (train=False), no RNG needed for the crop itself -
    `rng` here is only the network's own `.sample` randomness (some
    networks, e.g. MIP, ignore it entirely).

    network_batch_stats (explicit request - see conversation history; this
    used to always read network_model.batch_stats directly instead of
    taking it as its own argument - a real bug, not just a stale comment:
    it meant EMA params (network_params=network_ema_params) always got
    paired with the LIVE self.network.batch_stats regardless, two
    different-speed summaries of the network's history mixed together -
    see ChunkedPolicyLearner.network_ema_batch_stats's own construction-
    site comment for the full mechanism and why it matters specifically
    for rollout/eval, the only caller that uses EMA params here). Callers
    now explicitly pass whichever batch_stats belongs with the
    network_params they're also passing (EMA batch_stats alongside EMA
    params, live alongside live - see chunked_learner.py's own
    _inference_params, the single place that decides the pairing).
    Threaded through read-only (no `mutable` requested, since XQCFD's own
    .sample() always runs its BatchNorm in eval/training=False mode
    internally, which only READS the existing running stats, never writes
    them) only when not None.

    temperature/network_accepts_temperature (explicit request - see
    conversation history): matches xqc/agents/xqc/xqc_learner.py's own
    sample_actions_with_log_probs(temperature) exactly - xqc/networks/
    hetstat.py's own StationaryHeteroskedasticNormalTanhDistribution scales
    its distribution's stddev by `temperature` (`scale * temperature`), so
    temperature=0.0 (the reference's own eval_env.evaluate() default - see
    xqc/envs/envs.py) makes XQCFD's own .sample() deterministic (zero-
    variance, mode-equivalent), matching the reference's own eval
    methodology exactly. Every OTHER actor in this family (DP/FM/MF/MIP/NF/
    BC/AF) has a deterministic .sample() already (no stochastic component
    at all), so their own .sample(rng, shape, cond) signatures accept no
    such kwarg - passing `temperature=` to them would raise a TypeError,
    which is why this is gated on network_accepts_temperature (static,
    bound once in chunked_learner.py's own _build_actor_bound_fns via
    `type(self).__name__ == "XQCFDLearner"`) rather than passed
    unconditionally. False (default) reproduces today's behavior exactly
    for every other agent - this kwarg is simply never added to the
    .sample() call at all."""
    obs, _ = encode_observation_from_models(
        vision_models, vision_params, obs_encoder_model, obs_encoder_params,
        obs_window, images, obs_offset, obs_scale, camera_keys, train=False,
        random_crop=random_crop, crop_shape=crop_shape,
    )
    network_variables = {"params": network_params}
    if network_batch_stats is not None:
        network_variables["batch_stats"] = network_batch_stats
    temperature_kwarg = {"temperature": temperature} if network_accepts_temperature else {}
    action = network_model.apply_fn.apply(
        network_variables, rng, act_shape, obs, method="sample", **temperature_kwarg
    )
    return action * action_scale + action_offset
