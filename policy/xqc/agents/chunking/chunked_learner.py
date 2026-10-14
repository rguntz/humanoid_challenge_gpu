"""Shared learner for every chunked-action policy that trains a network
exposing the uniform `.loss(act, cond, rng) -> (loss, infos)` /
`.sample(rng, act_shape, cond) -> action` Flax-method interface:
DiffusionPolicy (xqc/agents/dp), FlowMatching (xqc/agents/fm), MeanFlow
(xqc/agents/mf), MIP (xqc/agents/mip), MIPFlow (xqc/agents/mip_flow),
RealNVP (xqc/agents/nf), MLPPolicy (xqc/agents/bc), and AF (xqc/agents/af).

Each of those eight concrete Learner classes is now a thin subclass: its own
__init__ just splits its yaml config's fields into the common ones handled
here and the network-specific ones forwarded to `network_cls(...)` (see
e.g. xqc/agents/actors/dp/dp_learner.py). Everything else - vision/obs-encoder
setup, dataset + normalization, optimizer/LR-schedule/EMA, update()/save()/
load() - lives here exactly once, ported from xqc/agents/actors/dp/dp_learner.py
(DP is the fidelity reference: its dataset windowing/normalization/EMA/etc.
were independently verified against the actual reference Diffusion Policy
implementation - see xqc/agents/common/dataset.py's
ChunkedOfflineImageDataset/ChunkedOfflineLowdimDataset docstrings).

The actor's own train/eval/infer functions (update_actor_bc/val_loss/infer)
live in xqc/agents/chunking/actor.py, not inline here - matching
xqc_learner.py's own convention of importing update_actor/update_actor_bc
from xqc/agents/xqc/actor.py rather than defining them inline (see
actor.py's own module docstring for the full rationale).

The receding-horizon action-chunk execution bookkeeping (sample_actions/
reset_policy/reset_policy_idx and their buffered state) lives separately in
xqc/agents/common/chunk_execution.py's ChunkExecutor, not inline here - see
conversation history: it's pure imitation-learning/actor plumbing with no
critic equivalent, so keeping it inline would only get more cramped once
this class also hosts a critic. This class's own sample_actions/
reset_policy/reset_policy_idx are now thin delegators to self._executor.

Weights container: three separate `Model` instances (xqc/networks/common.py)
- self.network, self.vision (one Model per camera), self.obs_encoder -
mirroring xqc/agents/xqc/xqc_learner.py's own pattern of several
independently-tracked named Models (self.actor/self.critic/
self.target_critic/self.temperature) instead of one combined TrainState
dict, per conversation history: chunked_learner.py is meant to eventually
host both actor and critic together, the same way xqc_learner.py hosts
actor+critic, so it adopts that same structural pattern rather than just
Model's building block in isolation. Each Model gets its own optimizer,
independently instantiated but defaulted to identical settings - Adam-style
optimizers track per-parameter moment estimates independently of how the
pytree is grouped, so three separately-instantiated optax.adamw objects
with identical lr/betas/weight_decay/schedule produce numerically the same
per-weight updates as one shared optimizer over one combined dict (see
conversation history) - reproducing today's exact training dynamics by
default, with per-component settings a config change away, not a rewrite,
if ever wanted later. Since network/vision/obs_encoder train from one
connected loss (not separate objectives, unlike xqc_learner.py's actor vs
critic), actor.update_actor_bc computes one joint gradient across all three
Models at once (see its own docstring) rather than three separate
Model.apply_gradient calls.
"""
import functools
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from xqc.agents.chunking.actor import distribution_policy_loglik, infer, policy_loglik, update_actor, update_actor_bc, val_loss
from xqc.agents.common.chunk_execution import ChunkExecutor
from xqc.agents.common.chunked_critic_target import layer_norm_ratios, per_layer_norms, split_dense_and_batchnorm_norms
from xqc.agents.common.ema import ema_decay as _ema_decay, ema_update as _ema_update
from xqc.agents.common.obs_encoder import IdentityObsEncoder, build_vision_and_obs_encoder
from xqc.agents.critics.mlp import critic as mlp_critic_fns
from xqc.agents.critics.transformer import critic as transformer_critic_fns
from xqc.agents.critics.transformer.network import VMapTransformerCritic
from xqc.agents.critics.vision_utils import rescale_images
from xqc.agents.critics.xqc_style import critic as xqc_style_critic_fns
from xqc.agents.xqc.utils import target_update
from xqc.logging.logging import print_total_param_count
from xqc.networks import critic_net
from xqc.networks.common import InfoDict, Model, norm_network
from xqc.replay_buffer.chunked_qc_buffer import ChunkedBatch, Dataset, compute_chunked_stats


# Components saved/restored via one Model-shaped orbax checkpoint each - see
# save()/load() below. Mirrors xqc/agents/xqc/xqc_learner.py's own
# _NETWORK_NAMES ("actor", "prior_actor", ...): everything there is a single
# named Model, while here `vision`/`prior_vision` are dicts (one Model per
# camera, size varies per agent config) so they get their own loop right
# after this one rather than being folded into the tuple.
# prior_network/prior_obs_encoder: the frozen BC-pretrained snapshot (see
# conversation history) - always constructed in __init__ regardless of
# critic_name, so unconditionally checkpointed here same as network/
# obs_encoder themselves.
_MODEL_NAMES = ("network", "obs_encoder", "prior_network", "prior_obs_encoder")

# The critic's own equivalent, saved/restored only when self.critic is not
# None - xqc_learner.py's own _NETWORK_NAMES includes "critic"/
# "target_critic" right alongside "actor" in ONE flat tuple (checkpointed
# every time, since xqc always has a critic); ours is conditional instead,
# since most existing agent configs have no critic at all - see
# conversation history.
_CRITIC_MODEL_NAMES = ("critic", "target_critic", "critic_obs_encoder", "critic_target_obs_encoder")

# critic_name (a plain config string) -> the module holding that variant's
# own compute_q/update_critic/eval_critic - mirrors xqc_learner.py's own
# convention of importing update_critic straight from xqc/agents/xqc/
# critic.py and calling it directly (no wrapper object at all); the only
# reason this is a dict of THREE modules instead of one direct import is
# that (unlike xqc, which has one fixed critic architecture) this class
# supports three swappable ones - see conversation history.
_CRITIC_FNS_REGISTRY = {
    "mlp": mlp_critic_fns,
    "xqc_style": xqc_style_critic_fns,
    "transformer": transformer_critic_fns,
}


def _model_checkpoint_dict(model: Model) -> Dict[str, Any]:
    return {"params": model.params, "opt_state": model.opt_state, "step": model.step, "batch_stats": model.batch_stats}


def _adam_moment_norms(opt_state) -> Dict[str, float]:
    """Global norm of Adam's own first/second-moment accumulators (mu/nu)
    found anywhere inside opt_state (explicit request - see conversation
    history: comparing an MF run against xqcfd_chunked_square_floriant at
    the offline->online transition, specifically whether
    reset_optimizers's own opt_state=tx.init(...) - MFLearner only, a
    no-op for XQCFDLearner, see reset_optimizers's own docstring - visibly
    zeroes momentum there).

    Found by LEAF PATH NAME, not by assuming a fixed nesting depth -
    optax.scale_by_adam's own ScaleByAdamState names these fields 'mu'/
    'nu' no matter how deeply optax.adamw's own chain is wrapped
    (optax.conditionally_mask for policy_delay>1 - see _maybe_delay_tx -
    adds one more level of nesting on top, confirmed directly: a quick
    standalone check of optax.conditionally_mask(optax.adamw(...), ...)'s
    own init() shows mu/nu landing at .inner_state[0].mu/.nu, one level
    deeper than plain adamw's own .mu/.nu - this walks past that
    automatically rather than hardcoding either depth). Returns 0.0 for
    either if no such leaf exists (lr_schedule="constant" still uses
    adamw, so this should always find them for every agent in this repo
    today, but kept safe rather than assuming)."""
    mu_leaves, nu_leaves = [], []
    for path, leaf in jax.tree_util.tree_leaves_with_path(opt_state):
        # 'mu'/'nu' name an inner GetAttrKey, not path[-1] - path[-1] is
        # whatever key structure params/mu/nu themselves have (e.g. a
        # DictKey('w') for a {'w': array} param pytree) - confirmed
        # directly against optax.adamw's own ScaleByAdamState nesting
        # (see this function's own docstring), so every entry in path
        # needs checking, not just the last one.
        path_names = {getattr(p, "name", None) for p in path}
        if "mu" in path_names:
            mu_leaves.append(leaf)
        elif "nu" in path_names:
            nu_leaves.append(leaf)
    return {
        "actor_adam_mu_norm": optax.global_norm(mu_leaves) if mu_leaves else jnp.asarray(0.0),
        "actor_adam_nu_norm": optax.global_norm(nu_leaves) if nu_leaves else jnp.asarray(0.0),
    }


@functools.partial(
    jax.jit,
    static_argnames=(
        "update_actor_bc_fn", "update_actor_fn", "update_critic_fn", "critic_name", "pre_train",
        "camera_keys", "Ta", "execution_horizon", "act_dim", "update_actor", "pretrain_use_sarsa",
        "critic_bootstrap_best_of_n",
    ),
)
def _update(
    rng,
    network_model: Model, vision_models: Dict[str, Model], obs_encoder_model: Model,
    critic_model: Optional[Model], target_critic: Optional[Model],
    critic_vision_models: Optional[Dict[str, Model]], critic_target_vision_models: Optional[Dict[str, Model]],
    critic_obs_encoder_model: Optional[Model], critic_target_obs_encoder_model: Optional[Model],
    actor_batch, critic_batch,
    obs_offset, obs_scale,
    critic_tau: float, critic_target_net_update_freq: int,
    update_actor_bc_fn, update_actor_fn, update_critic_fn,
    critic_name: Optional[str], pre_train: bool,
    camera_keys=(), Ta: int = 0, execution_horizon: int = 0, act_dim: int = 0,
    update_actor: bool = True,
    pretrain_use_sarsa: bool = True,
    prior_network_model: Optional[Model] = None,
    prior_vision_models: Optional[Dict[str, Model]] = None,
    prior_obs_encoder_model: Optional[Model] = None,
    offline_mask=None,
    critic_bootstrap_network_model: Optional[Model] = None,
    critic_bootstrap_vision_models: Optional[Dict[str, Model]] = None,
    critic_bootstrap_obs_encoder_model: Optional[Model] = None,
    critic_bootstrap_best_of_n: int = 1,
):
    """The chunking-family equivalent of xqc_learner.py's own _update -
    critic updates FIRST, then the actor updates using that SAME-STEP,
    FRESHLY-UPDATED critic (matching xqc_learner.py's own
    `new_actor, actor_info = actor_update_fn(critic=new_critic, ...)`
    exactly - see conversation history: our own code used to update the
    actor first against the PREVIOUS step's critic, one step stale
    relative to the reference).

    Calls update_critic directly (via update_critic_fn, bound once in
    _build_critic_bound_fns) and does the target-network update +
    diagnostics INLINE, right here - NOT through a separate
    jit_critic_step-style wrapper function, matching xqc_learner.py's own
    _update exactly (it inlines `tau = jnp.where(...);
    new_target_critic = target_update(...)` itself rather than delegating
    to a wrapper). train_critic_sarsa.py's own standalone critic-only
    training - a use case with no equivalent in xqc's own reference at all
    (xqc never trains a critic without an actor) - now runs through
    xqc/agents/chunking/chunked_learner_sarsa.py's own ChunkedCriticSARSALearner
    instead, which does this identical target-update+diagnostics sequence
    inline itself (its own module-level _update, copied from this one) -
    see conversation history. The old per-variant jit_critic_step wrapper
    functions that used to exist for that purpose have been deleted.

    update_actor_bc_fn/update_actor_fn/update_critic_fn are already fully
    bound (via functools.partial, in _build_actor_bound_fns/
    _build_critic_bound_fns) with every piece of the CRITIC's/ACTOR's own
    static config they need (critic_obs_offset/critic_obs_scale/
    critic_camera_keys/... for update_critic_fn; obs_offset/obs_scale/
    camera_keys/... for update_actor_bc_fn/update_actor_fn) - so this
    function itself only ever touches what genuinely varies per call: the
    Models, the batches, rng, critic_tau/critic_target_net_update_freq
    (kept traced, not bound, since this function needs the PRE-step
    critic_model.step to compute them), and the actor's OWN obs_offset/
    obs_scale/camera_keys/Ta/execution_horizon/act_dim - these last five
    are passed straight through from the caller rather than bound anywhere
    (there's no actor-side bound-fn to bind them into here), needed ONLY
    to feed update_critic_fn's own actor-bootstrap kwargs below when
    pre_train=False (the critic's own update_critic_fn is otherwise bound
    with the CRITIC's, not the actor's, obs_offset/obs_scale/camera_keys -
    a different normalizer/camera set, see conversation history).

    critic_model/target_critic/... and critic_batch are None for a
    critic-less learner (every existing DP/FM/MF/... agent config) or for
    a pre_train=True call that has no batch this round - the critic step
    is skipped entirely then, all six critic-side Models pass through
    unchanged, matching xqc_learner.py's own "no critic step, batch=None"
    ergonomics for the equivalent case. When critic_batch IS given and
    pre_train=False (or pre_train=True with pretrain_use_sarsa=False -
    conf/agent/base_agent.yaml's own pretrain_use_sarsa knob), update_critic_fn
    additionally gets the actor's own LIVE (pre-this-step) network_model/
    vision_models/obs_encoder_model passed as actor_network_model/
    actor_vision_models/actor_obs_encoder_model, switching the critic's own
    TD target from chunked-SARSA to the actor's own predicted next action -
    see chunked_critic_target.py's own bootstrap_next_actions docstring for
    the full rationale.

    NOTE: this function currently carries its OWN @jax.jit (unlike
    xqc_learner.py's own _update, which is a plain function - jit lives
    only on the outer _do_multiple_updates there). This class doesn't have
    a _do_multiple_updates-equivalent YET (see conversation history -
    deliberately deferred, a separate, later change); ChunkedPolicyLearner.
    update() still loops over num_updates in plain Python, calling this
    jitted function once per iteration. Once _do_multiple_updates exists,
    the @jax.jit above moves there and this becomes a plain function too,
    matching the reference exactly."""
    rng, critic_crop_key, actor_crop_key, dropout_key, actor_bootstrap_rng = jax.random.split(rng, 5)

    if critic_batch is not None:
        dropout_kwarg = {"dropout_key": dropout_key} if critic_name == "transformer" else {}
        # offline_mask (explicit request - see conversation history): only
        # xqc_style's own update_critic accepts this so far (q_max_offline/
        # q_max_online diagnostic - see its own docstring) - mlp/transformer
        # don't have the parameter at all, so this must stay conditional,
        # same pattern as dropout_kwarg just above (a critic-specific
        # optional kwarg, not a universal one every variant needs to
        # accept-and-ignore).
        offline_mask_kwarg = {"offline_mask": offline_mask} if critic_name == "xqc_style" else {}
        # pre_train=False (RL/online) always takes the actor-bootstrap
        # branch, switching the critic's own TD target from chunked-SARSA
        # (batch.next_actions) to the actor's OWN predicted next action -
        # mandatory there, unaffected by pretrain_use_sarsa. pre_train=True
        # (offline) now ALSO takes this branch when pretrain_use_sarsa=
        # False (conf/agent/base_agent.yaml's own pretrain_use_sarsa,
        # default True = the old hard-coded SARSA-only pre-training
        # behavior) - see each critic variant's own critic.py,
        # chunked_critic_target.py's bootstrap_next_actions docstring.
        # Uses the PRE-this-step actor Models (network_model/vision_models/
        # obs_encoder_model, not yet updated below - the actor update
        # happens AFTER this), matching xqc's own critic bootstrap using
        # the actor's LIVE (not target) params - UNLESS critic_bootstrap_
        # network_model/vision_models/obs_encoder_model were given (explicit
        # request - see conversation history and ChunkedPolicyLearner's own
        # critic_bootstrap_use_ema docstring): update()'s own call site
        # builds those as .replace()'d EMA-params copies of the same Models
        # when self.critic_bootstrap_use_ema is True, None (default)
        # otherwise - so this `or` falls back to the live network_model/
        # vision_models/obs_encoder_model exactly as before whenever the
        # toggle is off, a true no-op.
        bootstrap_network_model = critic_bootstrap_network_model if critic_bootstrap_network_model is not None else network_model
        bootstrap_vision_models = critic_bootstrap_vision_models if critic_bootstrap_vision_models is not None else vision_models
        bootstrap_obs_encoder_model = critic_bootstrap_obs_encoder_model if critic_bootstrap_obs_encoder_model is not None else obs_encoder_model
        actor_bootstrap_kwargs = (
            {}
            if (pre_train and pretrain_use_sarsa)
            else dict(
                actor_network_model=bootstrap_network_model,
                actor_vision_models=bootstrap_vision_models,
                actor_obs_encoder_model=bootstrap_obs_encoder_model,
                actor_obs_offset=obs_offset,
                actor_obs_scale=obs_scale,
                actor_camera_keys=camera_keys,
                Ta=Ta, execution_horizon=execution_horizon, act_dim=act_dim,
                actor_rng=actor_bootstrap_rng,
                best_of_n=critic_bootstrap_best_of_n,
            )
        )
        new_critic, new_critic_vision, new_critic_obs_encoder, critic_info = update_critic_fn(
            critic_model, target_critic, critic_vision_models, critic_target_vision_models,
            critic_obs_encoder_model, critic_target_obs_encoder_model, critic_batch,
            crop_key=critic_crop_key,
            **dropout_kwarg,
            **offline_mask_kwarg,
            **actor_bootstrap_kwargs,
        )
        # Optimization-health diagnostics - see each critic variant's own
        # (former) jit_critic_step for the identical rationale.
        critic_info["grad_norm"] = optax.global_norm(critic_info.pop("critic_grads"))
        critic_info.pop("critic_activations", None)
        critic_info["weight_norm"] = optax.global_norm(new_critic.params)
        if critic_name != "transformer":
            critic_info.update(split_dense_and_batchnorm_norms(new_critic.params))
        critic_info.update({f"layer_norm/{k}": v for k, v in per_layer_norms(new_critic.params).items()})

        tau_ = jnp.where(critic_model.step % critic_target_net_update_freq == 0, critic_tau, 0.0)
        new_target_critic = target_update(new_critic, target_critic, tau_)
        new_critic_target_vision = {
            cam: target_update(new_critic_vision[cam], critic_target_vision_models[cam], tau_) for cam in new_critic_vision
        }
        new_critic_target_obs_encoder = target_update(new_critic_obs_encoder, critic_target_obs_encoder_model, tau_)
    else:
        new_critic, new_target_critic = critic_model, target_critic
        new_critic_vision, new_critic_target_vision = critic_vision_models, critic_target_vision_models
        new_critic_obs_encoder, new_critic_target_obs_encoder = critic_obs_encoder_model, critic_target_obs_encoder_model
        critic_info: InfoDict = {}

    # update_actor=False (see ChunkedPolicyLearner.update()'s own docstring):
    # the actor has already hit its own actor_pre_training_steps budget but
    # the critic hasn't hit ITS OWN (independent) critic_pre_training_steps
    # yet - skip the actor's gradient step entirely this call (identity
    # pass-through, matching the critic's own "no batch this round" no-op
    # convention just above) while the critic keeps training - see
    # conversation history.
    if not update_actor:
        new_network, new_vision, new_obs_encoder = network_model, vision_models, obs_encoder_model
        actor_info: InfoDict = {}
    elif pre_train:
        new_network, new_vision, new_obs_encoder, actor_info = update_actor_bc_fn(
            network_model, vision_models, obs_encoder_model, actor_batch,
            rng=rng, crop_key=actor_crop_key,
        )
    else:
        new_network, new_vision, new_obs_encoder, actor_info = update_actor_fn(
            network_model, vision_models, obs_encoder_model,
            critic_model=new_critic, critic_vision_models=new_critic_vision, critic_obs_encoder_model=new_critic_obs_encoder,
            obs_window=critic_batch.observations, images=critic_batch.images,
            rng=rng, crop_key=actor_crop_key,
            # Read fresh each call (never bound via functools.partial in
            # _build_actor_bound_fns, unlike compute_q_fn/etc) - these get
            # REPLACED once, at the offline->online transition
            # (train_parallel_chunking.py's own first_online_transition
            # block), and a partial-bound copy taken at __init__ time would
            # go stale (still pointing at the freshly-initialized-random
            # snapshot from construction) - see conversation history.
            prior_network_model=prior_network_model,
            prior_vision_models=prior_vision_models,
            prior_obs_encoder_model=prior_obs_encoder_model,
            # See update_actor's own docstring - critic_batch's own
            # in_reference_region (needs the caller's online sample_sequence
            # call to have passed Ta= for this to be non-trivial, matching
            # the offline BC call site - see conversation history) is a
            # traced per-sample array, threaded fresh each call rather than
            # bound in _build_actor_bound_fns.
            in_reference_region=critic_batch.in_reference_region,
            # offline_mask: NOT a ChunkedBatch field (unlike
            # in_reference_region above) - built by train_parallel_chunking.py
            # at its own online/offline concat site and threaded through
            # ChunkedPolicyLearner.update()'s own offline_mask parameter down
            # to here, rather than through critic_batch - see actor.py's own
            # update_actor docstring and this module's own update() method.
            offline_mask=offline_mask,
            # bc_regularization's own target (explicit request - see
            # actor.py's own update_actor docstring): critic_batch.actions
            # is already normalized by this point (update()'s own
            # per-iteration loop does `batch_i = batch_i._replace(actions=
            # (batch_i.actions - self.action_offset)/self.action_scale,
            # ...)` BEFORE batch_i/critic_batch ever reaches _update) -
            # matches sampled_action/executed_action's own units exactly,
            # no further conversion needed here. Harmless to always pass
            # (bc_regularization=None/0 means update_actor never reads it).
            actions=critic_batch.actions,
        )

    if update_actor:
        # Actor-side counterpart of the critic's own weight_norm/dense_norm/
        # batchnorm_norm/layer_norm block above - "actor_"-prefixed so these
        # keys don't collide with the critic's identically-named ones once
        # {**critic_info, **actor_info} merges below (train_parallel_chunking.py's
        # own log_train_infos routes the "actor_"-prefixed family to its own
        # wandb section instead of weight_information/{phase}).
        actor_info["actor_weight_norm"] = optax.global_norm(new_network.params)
        actor_split = split_dense_and_batchnorm_norms(new_network.params)
        actor_info["actor_dense_norm"] = actor_split["dense_norm"]
        actor_info["actor_batchnorm_norm"] = actor_split["batchnorm_norm"]
        actor_info.update({f"actor_layer_norm/{k}": v for k, v in per_layer_norms(new_network.params).items()})
        # actor_adam_mu_norm/actor_adam_nu_norm (explicit request - see
        # _adam_moment_norms' own docstring): read AFTER this step's update
        # (new_network.opt_state, not the pre-update network_model.opt_state)
        # so a reset_optimizers() call earlier this same transition already
        # shows up as exactly 0 here, not one step stale.
        actor_info.update(_adam_moment_norms(new_network.opt_state))

    return (
        rng, new_network, new_vision, new_obs_encoder,
        new_critic, new_target_critic, new_critic_vision, new_critic_target_vision,
        new_critic_obs_encoder, new_critic_target_obs_encoder,
        {**critic_info, **actor_info},
    )


class ChunkedPolicyLearner:
    """Base class - see module docstring. Concrete subclasses (DPLearner,
    FMLearner, MFLearner, MIPLearner, MIPFlowLearner, NFLearner, BCLearner,
    AFLearner) call `super().__init__(network_cls=..., network_extra_kwargs=
    {...}, **kwargs)` from their own thin __init__, which exists only to
    keep each yaml config's field names/defaults unchanged."""

    def __init__(
        self,
        seed: int,
        observations: np.ndarray,
        actions: np.ndarray,
        train_dataset: Dataset,
        val_dataset: Dataset,
        raw_obs_all: np.ndarray,
        raw_actions_all: np.ndarray,
        num_robots: int,
        gripper_dim: int,
        network_cls: Type[nn.Module],
        network_extra_kwargs: Dict[str, Any],
        critic_name: Optional[str] = None,
        critic_kwargs: Optional[Dict[str, Any]] = None,
        To: int = 2,
        Ta: int = 16,
        execution_horizon: int = 8,
        batch_size: int = 64,
        camera_keys: Sequence[str] = (),
        vision_encoder: str = "resnet",
        vision_norm_type: str = "group",
        random_crop: bool = True,
        crop_shape: Sequence[int] = (76, 76),
        # Absolute-action training (state/lowdim only - camera_keys=() -
        # matches the reference's own square_lowdim_abs.yaml). When True,
        # the dataset converts each recorded 7-dim [pos(3), axis-angle
        # rot(3), gripper(1)] action to 10-dim [pos(3), rotation_6d(6),
        # gripper(1)] once at load time (xqc/agents/common/dataset.py's own
        # _load_windows), and sample_actions below converts the network's
        # own 10-dim rotation_6d output back to axis-angle before returning
        # it - the env itself must ALSO be constructed with its controller
        # in absolute-pose mode (input_type="absolute") for this to make
        # physical sense, which is xqc/envs/robomimic.py's own
        # make_robomimic_env's responsibility, not this class's - see
        # scripts/convert_dataset_to_abs_action.py's own docstring for the
        # full delta-vs-absolute background.
        abs_action: bool = False,
        # Vision/normalizer architecture constants (conf/agent/training/
        # vision.yaml, state.yaml) - fixed facts about the encoders/
        # checkpoints/reference formulas in use, kept in yaml rather than as
        # Python module constants (see xqc/agents/common/vision.py,
        # xqc/agents/common/dataset.py). vision_feature_dims[vision_encoder]
        # and spatial_softmax_feature_dim must agree for "resnet" (the
        # SpatialSoftmax feature width IS the resnet path's pooled output
        # width) - keep them in sync if you ever change one.
        vision_feature_dims: Dict[str, int] = None,
        spatial_softmax_num_kp: int = 32,
        spatial_softmax_feature_dim: int = 64,
        vit_image_size: int = 224,
        eef_pos_dim: int = 3,
        eef_quat_dim: int = 4,
        range_eps: float = 1e-7,
        # ChunkedOfflineLowdimDataset's own normalizer scheme (state/lowdim
        # obs only - camera_keys=() - unused for vision agents, which always
        # use ChunkedOfflineImageDataset's own per-key range normalizer
        # instead). "global" (default) matches the reference's own lowdim
        # get_normalizer exactly - see ChunkedOfflineLowdimDataset's own
        # docstring for what "per_dim_range" is and why it exists (a
        # debugging lever, not a reference-matching alternative).
        lowdim_normalizer: str = "global",
        num_encoder_layers: int = 2,  # unused - IdentityObsEncoder has no compression stage - kept accepted so conf/agent/*.yaml doesn't need updating.
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        beta1: float = 0.95,
        beta2: float = 0.999,
        grad_clip_norm: Optional[float] = 10.0,
        lr_schedule: str = "cosine_warmup",
        lr_warmup_steps: int = 500,
        # num_interactions/updates_per_step/lr_end (explicit request - see
        # conversation history - this used to be read ONLY inside a
        # hardcoded `type(self).__name__ == "XQCFDLearner"` branch, a
        # special case reproducing xqc/agents/xqc/xqc_learner.py's own
        # lr_schedule_fn by hand; unified below into the SAME formula every
        # agent uses, no agent-type branch): num_interactions is
        # train_parallel_chunking.py's own int(cfg.max_steps/
        # cfg.env.action_repeat) - already passed into agent_kwargs for
        # EVERY agent regardless of type, previously only consumed by
        # XQCFDLearner. None (default) for any construction path that
        # doesn't pass it at all (not every caller runs through
        # train_parallel_chunking.py) - treated as 0 below, same as a
        # pure-offline agent's own script explicitly setting max_steps=0.
        num_interactions: Optional[int] = None,
        updates_per_step: int = 1,
        # lr_end (explicit request - see conversation history): 0.0
        # (default) reproduces today's cosine_warmup/linear behavior for
        # every agent that doesn't override it - was 3e-5 here purely
        # because that used to be the ONLY value XQCFDLearner's own
        # special-cased branch ever read; now that cosine_warmup/linear
        # both honor it uniformly, keeping that nonzero default would have
        # silently changed every OTHER agent's end-of-schedule LR from 0 to
        # 3e-5. XQCFD_CHUNKED's own actor yaml (conf/actor/xqcfd_state.yaml)
        # now sets lr_end=3e-5 explicitly instead, to keep ITS behavior
        # unchanged.
        lr_end: float = 0.0,
        ema_rate: float = 0.995,
        ema_schedule: str = "warmup",
        ema_power: float = 0.75,
        ema_max_value: float = 0.9999,
        ema_rate_online: Optional[float] = None,
        actor_pre_training_steps: int = 500_000,
        mask_reference_region: bool = False,
        wasserstein_regularization: Optional[float] = None,
        # pretrain_use_sarsa (conf/agent/base_agent.yaml): True (default,
        # matches the old hard-coded behavior) -> the critic's TD target
        # during offline pre_train=True steps uses chunked-SARSA
        # (batch.next_actions); False -> it uses actor-bootstrap instead,
        # the same bootstrap_next_actions path online (pre_train=False)
        # always uses regardless of this flag - see this module's own
        # _update docstring/comments.
        pretrain_use_sarsa: bool = True,
        # critic_bootstrap_use_ema (explicit request - see conversation
        # history, supervisor's own suggestion): only meaningful together
        # with pretrain_use_sarsa=False (the actor-bootstrap TD-target path -
        # see this module's own _update docstring), since that's the only
        # path that ever queries the actor for a next-action at all. False
        # (default): today's behavior exactly - the critic's TD target
        # bootstraps off the LIVE actor (self.network/self.vision/
        # self.obs_encoder), same object the actor update itself trains that
        # same step. True: the SAME bootstrap call instead queries a
        # .replace()'d copy of those Models with EMA params/batch_stats
        # swapped in (self.network_ema_params/self.network_ema_batch_stats/
        # self.vision_ema_params/self.obs_encoder_ema_params - already
        # maintained every step whenever self.ema_rate<1, see _ema_update's
        # own call site below) - apply_fn/tx/opt_state/step stay the live
        # Model's own (bootstrap_next_actions never reads those). Applies
        # uniformly to BOTH the pretrain_use_sarsa=False offline-BC path and
        # the always-actor-bootstrap online-RL path (same shared
        # actor_bootstrap_kwargs dict in _update - no pre_train-specific
        # split here, by explicit request). Only a real behavioral change
        # when self.ema_rate<1 - XQCFD_CHUNKED's own default ema_rate=1.0
        # makes this a harmless no-op (EMA and live are already the same
        # tensor) unless ema_rate is ALSO lowered.
        critic_bootstrap_use_ema: bool = False,
        # critic_bootstrap_best_of_n (explicit request - see conversation
        # history: the "optimal policy" the supervisor described - sample N
        # candidate chunks from the actor, rank by the TARGET critic's own
        # Q, min-over-ensemble, keep the best. NO actor gradient update is
        # involved at all - this is purely a sampling-time ranking wrapper
        # around bootstrap_next_actions's existing single-.sample() call,
        # same shared-across-{mlp,xqc_style,transformer} mechanism
        # critic_bootstrap_use_ema above plugs into. 1 (default): the exact
        # single-sample path bootstrap_next_actions already took before
        # this was added - score_fn is never built/called, a true no-op.
        # >1: candidates are sampled from whichever actor the
        # critic_bootstrap_network_model/etc. fallback above resolves to
        # (live, or EMA when critic_bootstrap_use_ema=True - the two
        # toggles compose freely), then ranked by the critic. Same
        # pretrain_use_sarsa-gated scope as critic_bootstrap_use_ema: only
        # ever reaches bootstrap_next_actions at all when pretrain_use_sarsa
        # =False (offline) or always (online, unconditionally) - see this
        # module's own _update docstring.
        critic_bootstrap_best_of_n: int = 1,
        # rollout_best_of_n (explicit request - see conversation history):
        # the ACTING-side counterpart of critic_bootstrap_best_of_n above -
        # same "optimal policy" (sample N, rank by Q, no actor gradient
        # update), but for sample_actions's own online-rollout call
        # (train_parallel_chunking.py's own agent.sample_actions(...) call
        # site, the one that actually picks the action executed in the
        # env) rather than the critic's TD target. A SEPARATE knob, not
        # reused from critic_bootstrap_best_of_n, because the two rank
        # against different critics by design: the TD target ranks against
        # the FROZEN target critic (matching the TD target's own
        # bootstrap-off-target-network convention), while rollout should
        # always act on the most up-to-date LIVE critic - see
        # sample_actions's own docstring. 1 (default): today's exact
        # single-sample behavior, a true no-op. Set equal to
        # critic_bootstrap_best_of_n if symmetric N is what's wanted -
        # nothing stops that, they're just independently defaulted.
        rollout_best_of_n: int = 1,
        # wasserstein_offline_only (conf/actor/base_actor.yaml, explicit
        # request - see xqc/agents/chunking/actor.py's own update_actor
        # docstring for the full rationale): False (default) regularizes the
        # actor toward prior_network on every sample in the mixed online+
        # offline critic_batch, same as before this flag existed. True
        # restricts it to offline-buffer rows only - pulling the actor
        # toward its frozen BC prior at an ONLINE (possibly off-the-BC-
        # manifold) state risks asking that frozen prior to extrapolate/
        # hallucinate an action for a state it never saw during pretraining,
        # which is not a trustworthy distillation target.
        wasserstein_offline_only: bool = False,
        # actor_weight (conf/actor/base_actor.yaml, explicit request -
        # debugging ablation, see xqc/agents/chunking/actor.py's own
        # update_actor docstring for the full rationale): scales ONLY the
        # Q-maximization term of the actor loss. Default 1.0 reproduces
        # today's behavior exactly. 0.0 (paired with wasserstein_regularization
        # on) isolates pure distillation-toward-prior with the RL objective
        # removed entirely - a sanity check on whether the distillation
        # mechanism itself works (actor stays near its BC starting
        # performance, offline_prior_mse -> 0) independent of whatever the
        # Q-term does.
        actor_weight: float = 1.0,
        # kl_regularization (conf/actor/base_actor.yaml, explicit request -
        # see xqc/agents/chunking/actor.py's own update_actor docstring for
        # the full rationale): the analytic-KL analogue of
        # wasserstein_regularization above, for xqc/agents/actors/xqc/
        # xqc_learner.py's own XQCFDLearner specifically - a FIXED
        # coefficient (matching conf/agent/xqcfd.yaml's own temp_lr=0.0,
        # i.e. a fixed, not learned, temperature in the reference this is
        # reproducing), gated the same way wasserstein_regularization is
        # gated to MFLearner just below.
        kl_regularization: Optional[float] = None,
        # bc_regularization (conf/actor/base_actor.yaml, explicit request -
        # see xqc/agents/chunking/actor.py's own update_actor docstring for
        # the full rationale): an ADDITIONAL plain BC loss term, against the
        # CURRENT batch's own real recorded action (critic_batch.actions,
        # the mixed offline+online buffer during online RL) - the reference
        # repo this session is reproducing keeps training the actor this way
        # throughout online RL, not just during offline pretraining. Unlike
        # wasserstein_regularization/kl_regularization, this needs NO
        # actor_name gate - every actor in this family already implements
        # .loss() for offline BC, so this generalizes to all of them, not
        # just MF/XQCFD. None/0 (default): off, a true no-op (actor.py's own
        # update_actor never even reads critic_batch.actions in that case).
        # Useful standalone too (e.g. actor_weight=0.0 + bc_regularization>0
        # = pure BC fine-tuning on the growing buffer, no RL objective at
        # all for that step) - not only paired with critic_bootstrap_best_of_n/
        # rollout_best_of_n above, though that pairing was the motivating
        # case (see conversation history).
        bc_regularization: Optional[float] = None,
        # use_weight_norm (conf/actor/base_actor.yaml, explicit request):
        # renormalizes the ACTOR network's own Dense-layer kernels after
        # every actor gradient step (xqc/agents/chunking/actor.py's own
        # update_actor_bc/update_actor, via xqc.networks.common.norm_network)
        # - matches xqc/agents/xqc/actor.py's own `if use_weight_norm:
        # new_actor = norm_network(new_actor)`, and xqc_style/critic.py's
        # own identically-purposed critic-side use_weight_norm (a SEPARATE
        # flag, inside critic_kwargs - this one is the actor's own). Default
        # False reproduces today's behavior exactly for every actor that
        # doesn't set it (DP/FM/MF/MIP/MIP-Flow/NF/BC/AF).
        use_weight_norm: bool = False,
        # policy_delay (conf/agent/xqc_base_agent.yaml, explicit request):
        # the actor's own optimizer only actually applies an update once
        # every `policy_delay` steps (optax.conditionally_mask, wrapped
        # around actor_tx/encoder_tx below and in reset_optimizers) -
        # matches xqc/agents/xqc/xqc_learner.py's own
        # `optax.conditionally_mask(actor_tx, lambda step: jnp.mod(step,
        # policy_delay)==0)` exactly (TD3-style delayed policy updates -
        # the critic updates every step regardless, only the actor/encoder
        # tx are gated). Default 1 (every step) is numerically a no-op
        # (`step % 1 == 0` always True) but is kept as a PLAIN, unwrapped
        # optax.adamw when policy_delay<=1 (rather than always wrapping)
        # specifically to leave every existing agent's opt_state pytree
        # shape - and hence checkpoint compatibility - completely
        # unchanged; only an agent that actually sets policy_delay>1 gets
        # the wrapped tx (and its differently-shaped opt_state).
        policy_delay: int = 1,
        # offline_data_ratio (conf/agent/base_agent.yaml): pulled out of
        # **kwargs (it used to be absorbed there, read only via
        # cfg.agent.offline_data_ratio directly in train_parallel_chunking.py)
        # specifically so wasserstein_offline_only's own fail-fast check just
        # below can see it at construction time - offline_data_ratio<=0.0
        # means the online loop's own concat block (train_parallel_chunking.py)
        # never runs, so no row is ever tagged offline, silently making
        # wasserstein_offline_only=True regularize against nothing every
        # step - see conversation history (explicit request: fail loudly
        # here rather than silently training with an inert term, same
        # philosophy as the wasserstein_regularization/actor_name check
        # just below).
        offline_data_ratio: float = 1.0,
        **kwargs: Any,  # absorbs fields consumed elsewhere in train_parallel.py
        # (name, reward_normalization, num_seeds, updates_per_step,
        # num_interactions, critic_pre_training_steps (lives inside
        # critic_kwargs, read directly off cfg by the driver loop, not by
        # this constructor - see conversation history), ...)
    ):
        assert Ta & (Ta - 1) == 0, f"Ta must be a power of 2, got {Ta}"
        # The predicted window starts AT "now" (t), matching the critic's
        # own actions[t:t+h] convention - see conversation history: the
        # reference's own (To-1)-steps-before-"now" alignment was
        # deliberately dropped when the IL/RL datasets were unified onto
        # xqc/replay_buffer/chunked_qc_buffer.py's Dataset. Ta only needs to
        # reach execution_horizon steps now, no To-1 term.
        assert Ta >= execution_horizon, (
            f"Ta ({Ta}) must be >= execution_horizon ({execution_horizon})."
        )
        self.To = To
        self.Ta = Ta
        self.execution_horizon = execution_horizon
        self.batch_size = batch_size
        self.camera_keys = tuple(camera_keys)
        self.random_crop = random_crop
        self.crop_shape = tuple(crop_shape)
        self.abs_action = abs_action
        self.ema_rate = ema_rate
        self.ema_schedule = ema_schedule
        self.ema_power = ema_power
        self.ema_max_value = ema_max_value
        # ema_rate_online (explicit request - see conversation history):
        # None (default) changes nothing for any existing script - every
        # sample_actions call keeps gating off self.ema_rate exactly as
        # before. Set e.g. agent.ema_rate_online=1.0 to force the ONLINE
        # rollout's own sample_actions call (train_parallel_chunking.py's
        # online loop, the one that actually fills the replay buffer) onto
        # LIVE weights specifically, while offline-phase eval rollouts, the
        # EMA buffer's own maintenance (_ema_update below), and any other
        # sample_actions caller (e.g. eval, both phases) stay governed by
        # self.ema_rate, untouched - see _inference_params's own ema_rate
        # param. Kept as a separate field (not a mutation of ema_rate
        # itself) specifically so it shows up as its own entry in the
        # wandb config dump, next to ema_rate, instead of overloading one
        # field with two meanings.
        self.ema_rate_online = ema_rate_online
        self.grad_clip_norm = grad_clip_norm
        # Ablation knob (default off - see xqc/agents/common/loss_masking.py
        # and ChunkedBatch's own in_reference_region docstring): sampling
        # stays uniform across every position either way (the actor/critic
        # coverage-unification fix this flag ablates is never re-disabled
        # here), only the BC loss's per-sample weighting changes.
        self.mask_reference_region = mask_reference_region
        self.pretrain_use_sarsa = pretrain_use_sarsa
        self.critic_bootstrap_use_ema = critic_bootstrap_use_ema
        self.critic_bootstrap_best_of_n = critic_bootstrap_best_of_n
        self.rollout_best_of_n = rollout_best_of_n
        # Fail fast at construction (before any pretraining compute is
        # spent) rather than only at the first online-RL actor update -
        # see actor.py's own update_actor docstring/matching runtime check
        # for the full rationale (MF-only for now, by explicit request;
        # crashes loudly for any other actor type rather than silently
        # skipping the regularization).
        self.wasserstein_regularization = wasserstein_regularization
        if wasserstein_regularization and type(self).__name__ not in ("MFLearner", "MIPLearner"):
            raise ValueError(
                f"wasserstein_regularization is only implemented for MF/MIP (got {type(self).__name__}) - "
                "see conversation history. Either turn wasserstein_regularization off for this agent, "
                "or verify+extend the shared-z coupling trick in xqc/agents/chunking/actor.py's own "
                "update_actor for this actor type first."
            )
        self.wasserstein_offline_only = wasserstein_offline_only
        if wasserstein_offline_only and offline_data_ratio <= 0.0:
            raise ValueError(
                f"wasserstein_offline_only=True needs agent.offline_data_ratio > 0.0 (got {offline_data_ratio}) - "
                "otherwise train_parallel_chunking.py's own online loop never mixes any offline-buffer row into "
                "critic_batch, so every sample would be tagged online and wasserstein_penalty would be masked to "
                "zero on every single step, silently training with the regularizer effectively off. Either raise "
                "offline_data_ratio above 0, or turn wasserstein_offline_only off."
            )
        self.actor_weight = actor_weight
        # Fail fast at construction - same philosophy/placement as
        # wasserstein_regularization's own matching check above (MF-only
        # there, XQCFD-only here - see xqc/agents/chunking/actor.py's own
        # update_actor docstring).
        self.kl_regularization = kl_regularization
        self.bc_regularization = bc_regularization
        if kl_regularization and type(self).__name__ != "XQCFDLearner":
            raise ValueError(
                f"kl_regularization is only implemented for XQCFD (got {type(self).__name__}) - "
                "see conversation history. Either turn kl_regularization off for this agent, "
                "or verify+extend the analytic-KL path in xqc/agents/chunking/actor.py's own "
                "update_actor for this actor type first."
            )
        self.use_weight_norm = use_weight_norm
        self.policy_delay = policy_delay

        if vision_feature_dims is None:
            vision_feature_dims = {"resnet": 64, "vit": 768}

        low_dim_obs_dim = observations.shape[-1]
        act_dim = actions.shape[-1]
        self.low_dim_obs_dim = low_dim_obs_dim
        self.act_dim = act_dim
        self.vision_encoder = vision_encoder
        self.vision_dim = len(self.camera_keys) * vision_feature_dims[vision_encoder]
        raw_condition_dim = low_dim_obs_dim + self.vision_dim

        rng = jax.random.PRNGKey(seed)
        rng, init_key, vision_key, obs_encoder_key, network_rng = jax.random.split(rng, 5)
        self.rng = rng

        # One shared Dataset (xqc/replay_buffer/chunked_qc_buffer.py) for
        # both state and vision, and for the actor AND the critic - see
        # conversation history for the IL/RL dataset unification this
        # replaces (previously two separate reference-fidelity classes
        # here, ChunkedOfflineImageDataset/ChunkedOfflineLowdimDataset in
        # xqc/agents/common/dataset.py, with their own stricter
        # near-boundary-exclusion sampling rule - a real bug in an
        # actor-critic setting: the actor never trained on states near an
        # episode boundary, exactly what the critic's bootstrap target most
        # needs it to be competent at). train_dataset/val_dataset are
        # loaded ONCE by the caller (train_parallel.py/
        # train_parallel_chunking.py, via load_robomimic_flat) and handed
        # in directly - NOT loaded again here - so a critic that also wants
        # offline data (offline_data_ratio>0) can share the same
        # already-loaded arrays instead of re-reading the same hdf5 file a
        # second time.
        self.dataset = train_dataset
        self.val_dataset = val_dataset
        # Own seeded shuffle RNG (matches train_critic_sarsa.py's own
        # shuffle_rng pattern) - Dataset.sample's own rng param needs
        # explicit seeding here or it silently falls back to the unseeded
        # global RNG.
        self._dataset_rng = np.random.default_rng(seed=seed)

        # actor_optimizer_resets_online (explicit request - see
        # conversation history): named condition for reset_optimizers' own
        # early-return check below, kept as a real attribute (not just
        # inlined there) purely so reset_optimizers itself stays readable/
        # callable on its own terms. NOT read by train_parallel_chunking.py
        # any more - its offline->online transition no longer calls
        # reset_optimizers for ANY agent (every agent's own lr_schedule_fn
        # now already spans its correct combined offline+online budget
        # from construction - see this __init__'s own _schedule_span
        # comment below - so there's nothing left to reset), and its
        # online lr-logging now always queries agent.network.step/
        # agent.critic.step (the real, never-reset counters) unconditionally
        # instead of branching on this. Kept (both this attribute and
        # reset_optimizers itself) purely so that decision can be revisited
        # later without rewriting this from scratch - see
        # reset_optimizers's own docstring for why it's otherwise currently
        # dead code, deliberately.
        self.actor_optimizer_resets_online = type(self).__name__ != "XQCFDLearner"

        # Optimizer definition - shared lr/weight_decay/grad_clip config,
        # but a SEPARATE optax instance per component below (network,
        # each vision camera, obs_encoder) rather than one instance shared
        # across all of them. See this file's own module docstring for why
        # this reproduces today's exact joint-training dynamics by default.
        #
        # _schedule_span (explicit request - see conversation history: this
        # used to be a hardcoded XQCFDLearner-only special case with its
        # own separate formula - unified into ONE span every lr_schedule
        # mode below uses, and went through a real wrong turn on the way
        # here too - see below for the fix and why the first version of
        # this was still wrong):
        #   - actor_pre_training_steps + (num_interactions or 0) *
        #     updates_per_step: a SUM, not a max (explicit request - see
        #     conversation history, a real bug caught directly from a
        #     live run's own logged LR curve: with the max() this used to
        #     be, a real online-RL run's schedule hit end_value/flatlined
        #     tens of thousands of steps BEFORE the run's actual end,
        #     because the counter this schedule is queried with
        #     (agent.network.step/agent.critic.step, see those fields' own
        #     construction-site comments) is NEVER reset - it keeps
        #     counting through BOTH the offline pretraining budget AND the
        #     online one, back to back, so the schedule's own span has to
        #     cover BOTH added together, not just whichever one happens to
        #     be larger. A pure-offline agent/script (no online RL phase,
        #     so num_interactions=0 for its own run) still collapses this
        #     to exactly actor_pre_training_steps, today's unchanged value
        #     - (num_interactions or 0) is 0 there, same as before.
        #     num_interactions = train_parallel_chunking.py's own
        #     int(cfg.max_steps/cfg.env.action_repeat), already passed into
        #     agent_kwargs for every agent - None here only for a
        #     construction path that never passes it at all, treated as 0.
        #     Letting the SAME schedule span the whole offline+online run
        #     (now actually correctly sized to do so) means no reset is
        #     needed at the transition (see reset_optimizers's own
        #     docstring: no longer called from train_parallel_chunking.py's
        #     offline->online transition for ANY agent, kept defined/
        #     callable only in case this decision gets revisited).
        #   - max(..., lr_warmup_steps + 1): optax's own
        #     warmup_cosine_decay_schedule requires decay_steps >
        #     warmup_steps internally - this floor only matters for
        #     actor_pre_training_steps=0 AND num_interactions=0 together (a
        #     critic-only run - see conversation history), where the actor
        #     never takes a gradient step anyway (see update_actor gating
        #     below) so the schedule's actual shape is irrelevant, just
        #     needs to construct without crashing.
        _schedule_span = max(actor_pre_training_steps + (num_interactions or 0) * updates_per_step, lr_warmup_steps + 1)
        if lr_schedule == "cosine_warmup":
            lr_or_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=lr, warmup_steps=lr_warmup_steps,
                decay_steps=_schedule_span, end_value=lr_end,
            )
        elif lr_schedule == "linear":
            # No warmup - matches xqc_learner.py's own lr_schedule_fn
            # exactly (XQCFD_CHUNKED's own actor yaml, conf/actor/
            # xqcfd_state.yaml, sets lr_schedule="linear" specifically for
            # this - see conversation history for why that reference has
            # no warmup phase at all, unlike every ChiUNet-backed actor).
            lr_or_schedule = optax.linear_schedule(init_value=lr, end_value=lr_end, transition_steps=_schedule_span)
        elif lr_schedule == "constant":
            lr_or_schedule = lr
        else:
            raise ValueError(f"Unknown lr_schedule {lr_schedule!r}, expected 'cosine_warmup', 'linear', or 'constant'")
        # Exposed so callers (train_parallel.py) can log the actual current
        # LR value, matching the reference's own per-step
        # `lr_scheduler.get_last_lr()[0]` logging - optax schedules are
        # plain step->lr callables, `constant` isn't, so normalize both to
        # one callable interface here.
        self.lr_schedule_fn = lr_or_schedule if callable(lr_or_schedule) else (lambda step: lr_or_schedule)
        # Stashed for reset_optimizers(online_steps=...) below, which needs
        # to REBUILD (not just re-init state on) a fresh schedule sized for
        # the online phase's own duration at the offline->online transition
        # - see conversation history (actor_pre_training_steps is the
        # OFFLINE budget; reusing it as online's own decay_steps too made
        # online fine-tuning barely decay, or decay too early, depending on
        # how the two budgets happened to compare).
        self._lr_schedule_mode = lr_schedule
        self._lr_warmup_steps = lr_warmup_steps
        self._actor_peak_lr = lr
        self._actor_beta1, self._actor_beta2, self._actor_weight_decay = beta1, beta2, weight_decay

        # actor_tx/encoder_tx: named + built once here, same level/placement
        # as xqc_learner.py's own actor_tx/critic_tx/temperature_tx (built
        # right after the LR schedule, before any Model construction uses
        # them) - see conversation history. actor_tx is the prediction
        # network's own optimizer ("actor" = the policy/prediction network,
        # same naming xqc_learner.py itself uses); encoder_tx is shared by
        # every vision camera AND obs_encoder (currently identical
        # hyperparameters to actor_tx, but a separate named object/config
        # slot - giving vision its own LR later is then a one-line change
        # here, not a rewrite). No gradient clipping baked in here anymore
        # (see conversation history): the reference DiffusionPolicy
        # implementation clips ONE global norm over vision+UNet's gradients
        # COMBINED (one torch.nn.utils.clip_grad_norm_ call over the whole
        # model's parameters), not per-component - clipping happens jointly
        # instead, across all three gradient trees at once, in actor.py's
        # own _apply_joint_gradients (see its docstring), using this
        # class's own self.grad_clip_norm.
        # policy_delay / undelayed_until (explicit request - see
        # conversation history - this went through a real wrong turn
        # first, see _maybe_delay_tx's own docstring for the full story):
        # XQCFDLearner wants the delay active from the very first step,
        # matching its own reference (TD3-style delay active during
        # pre-training too, not just online - see conf/actor/
        # xqcfd_state.yaml's own policy_delay field comment) -
        # undelayed_until=0 (this method's own default) reproduces that.
        # Every OTHER agent (MF included) wants pre-training completely
        # undelayed on purpose - every offline BC step is a real actor
        # update regardless of policy_delay - with the configured delay
        # only taking effect once online RL starts - undelayed_until=
        # actor_pre_training_steps reproduces THAT. One optax transform,
        # built once here, that already knows how to do both (see
        # _maybe_delay_tx's own docstring for why no separate rebuild is
        # needed at the offline->online transition any more - that used to
        # be reset_optimizers's job before this session's own LR-schedule
        # unification removed its only remaining caller).
        _undelayed_until = 0 if type(self).__name__ == "XQCFDLearner" else actor_pre_training_steps
        actor_tx = self._maybe_delay_tx(optax.adamw(lr_or_schedule, b1=beta1, b2=beta2, weight_decay=weight_decay), _undelayed_until)
        encoder_tx = self._maybe_delay_tx(optax.adamw(lr_or_schedule, b1=beta1, b2=beta2, weight_decay=weight_decay), _undelayed_until)

        # Raw (pre-crop) per-camera resolution read straight from the loaded
        # dataset rather than assumed - sample_actions' own _image_history
        # buffer reuses this same dict so the two can never disagree.
        self.raw_image_shape = (
            {cam_key: tuple(int(d) for d in self.dataset[cam_key].shape[1:3]) for cam_key in self.camera_keys}
            if self.camera_keys else {}
        )
        self.vision, self.obs_encoder = build_vision_and_obs_encoder(
            self.camera_keys, self.raw_image_shape, vision_encoder, vision_norm_type,
            spatial_softmax_num_kp, spatial_softmax_feature_dim, vit_image_size,
            random_crop, self.crop_shape, raw_condition_dim, To, vision_key, obs_encoder_key, encoder_tx,
        )

        low_dim_stats = compute_chunked_stats(
            self.dataset, raw_obs_all, raw_actions_all, self.camera_keys, abs_action,
            normalizer=lowdim_normalizer, eef_pos_dim=eef_pos_dim, eef_quat_dim=eef_quat_dim,
            range_eps=range_eps, gripper_dim=gripper_dim, num_robots=num_robots,
        )
        self.obs_offset = jnp.asarray(np.concatenate([low_dim_stats.obs_offset, np.zeros(self.vision_dim, np.float32)]))
        self.obs_scale = jnp.asarray(np.concatenate([low_dim_stats.obs_scale, np.ones(self.vision_dim, np.float32)]))
        self.action_offset = jnp.asarray(low_dim_stats.action_offset)
        self.action_scale = jnp.asarray(low_dim_stats.action_scale)

        # Re-derive act_dim from the dataset's OWN (possibly abs-action-
        # converted, 10-dim pos+rotation_6d+gripper) actions, not the
        # outer `actions` argument above - that one is sourced from
        # OfflineReplayBuffer (train_parallel.py's shared, un-converted
        # data path, always the raw 7-dim recorded action), which would
        # silently build a wrong-sized network for abs_action - see this
        # class's own abs_action field docstring.
        act_dim = self.dataset["actions"].shape[-1]
        self.act_dim = act_dim
        self.num_robots = num_robots
        # Robot-executable action dim (what env.step() actually accepts) -
        # equal to act_dim except in abs_action mode, where each robot's own
        # 10-dim rotation_6d output gets converted back down to 7-dim
        # axis-angle before ever reaching the env (3 fewer dims PER ROBOT) -
        # see sample_actions below.
        self.robot_action_dim = act_dim - 3 * self.num_robots if abs_action else act_dim

        # Only the fields every network module needs regardless of
        # architecture (act_dim/Ta/cond_dim/To) are passed positionally by
        # name here - everything else (model_dim/emb_dim/kernel_size/
        # cond_predict_scale/dim_mult for the ChiUNet-backed networks,
        # hidden_dims for BC's plain MLPPolicy, t_two_step/norm_type/... for
        # MIP, etc.) lives in network_extra_kwargs, built by each concrete
        # subclass's own __init__ - keeps this base class agnostic to which
        # architecture-specific fields a given network_cls actually accepts.
        network_extra_kwargs = self._data_dependent_network_kwargs(
            network_extra_kwargs,
            (np.asarray(raw_actions_all) - np.asarray(self.action_offset)) / np.asarray(self.action_scale),
        )
        network_def = network_cls(
            act_dim=act_dim,
            Ta=Ta,
            cond_dim=raw_condition_dim,
            To=To,
            **network_extra_kwargs,
        )
        dummy_act = jnp.zeros((1, Ta, act_dim))
        dummy_obs = jnp.zeros((1, To, raw_condition_dim))
        self.network = Model.create(
            network_def, [{"params": init_key}, dummy_act, dummy_obs, network_rng], tx=actor_tx, method="loss"
        )

        # Baseline for actor_weight_norm_ratio/actor_layer_norm_ratio (see
        # _update's own "actor_weight_norm" block and update()'s own
        # "actor_weight_norm" ratio block below) - captured here, at the
        # actor's raw random init, same moment self.critic's own
        # _critic_init_weight_norm/_critic_init_layer_norms are captured
        # relative to the critic's init further down.
        self._actor_init_weight_norm = float(optax.global_norm(self.network.params))
        init_actor_split = split_dense_and_batchnorm_norms(self.network.params)
        self._actor_init_dense_norm = float(init_actor_split["dense_norm"])
        self._actor_init_batchnorm_norm = float(init_actor_split["batchnorm_norm"])
        self._actor_init_layer_norms = {k: float(v) for k, v in per_layer_norms(self.network.params).items()}

        # prior_network/prior_vision/prior_obs_encoder: a frozen snapshot of
        # the BC-pretrained actor, for online-RL KL regularization against
        # drifting too far from it - matches xqc_learner.py's own
        # prior_actor exactly (tx=None: never gradient-updated directly,
        # only ever wholesale .replace()'d - see train_parallel_chunking.py's
        # own `elif is_chunked_agent:` block, right where reset_optimizers()
        # is called, for the actual snapshot-taking moment). Reuses the same
        # keys as the real actor's own construction just above/below (same
        # convention xqc_learner.py's own prior_actor uses) - the initial
        # random params here are thrown away the moment the snapshot is
        # taken, before this is ever read for anything, so which key is used
        # doesn't matter. Not yet wired into any loss function - see
        # conversation history: scaffolding for a later change, this class
        # doesn't use KL regularization yet.
        self.prior_network = Model.create(
            network_def, [{"params": init_key}, dummy_act, dummy_obs, network_rng], tx=None, method="loss"
        )
        self.prior_vision, self.prior_obs_encoder = build_vision_and_obs_encoder(
            self.camera_keys, self.raw_image_shape, vision_encoder, vision_norm_type,
            spatial_softmax_num_kp, spatial_softmax_feature_dim, vit_image_size,
            random_crop, self.crop_shape, raw_condition_dim, To, vision_key, obs_encoder_key, None,
        )

        # EMA copies - plain params pytrees, not Models (no gradients ever
        # flow into them; they're nudged toward the live params every step
        # instead - see module docstring and _ema_decay/_ema_update below).
        # No equivalent of this exists in xqc_learner.py (its target_critic
        # is a differently-purposed, differently-computed, tau-based
        # mechanism that's never used for actions - see conversation
        # history) - this reproduces DP/FM's own schedule-based EMA, which
        # IS what sample_actions/compute_action_mse actually use to act.
        self.network_ema_params = self.network.params
        # network_ema_batch_stats (explicit request - see conversation
        # history): a SEPARATE EMA of self.network's own batch_stats
        # (FreezableBatchNorm's running mean/var, norm="batch"/"batch_ref"
        # only - None otherwise, matching self.network.batch_stats itself),
        # updated with the exact same ema_update/decay call as
        # network_ema_params every step (see that update site below) - NOT
        # FreezableBatchNorm's own internal running-average momentum (0.99,
        # a much shorter memory), a second, independent smoothing on top of
        # it. Without this, sample_actions(use_ema=True) (rollout/eval)
        # paired network_ema_params (a slow, long-memory average of the
        # weights) with self.network.batch_stats directly (a much
        # faster-moving average of the LIVE network's own recent
        # activations) - two different-speed summaries of the network's
        # history, mismatched at exactly the place that matters for real
        # task success. This makes both move at the same speed instead.
        self.network_ema_batch_stats = self.network.batch_stats
        self.vision_ema_params = {cam: model.params for cam, model in self.vision.items()}
        self.obs_encoder_ema_params = self.obs_encoder.params

        # Matches xqc_learner.py's own print_total_param_count([actor, critic])
        # call right after model construction (xqc/logging/logging.py) -
        # note that function currently only computes total_params and never
        # prints or returns it (a no-op in the original repo too, not
        # something introduced here - see conversation history).
        print_total_param_count([self.network, self.obs_encoder, *self.vision.values()])

        # Receding-horizon action-chunk execution bookkeeping - see this
        # file's own module docstring and chunk_execution.py's ChunkExecutor
        # for why this is a separate object rather than inline state here.
        self._executor = ChunkExecutor(
            To=To, Ta=Ta, execution_horizon=execution_horizon,
            low_dim_obs_dim=low_dim_obs_dim, camera_keys=self.camera_keys,
            raw_image_shape=self.raw_image_shape, robot_action_dim=self.robot_action_dim,
            act_dim=act_dim, abs_action=abs_action, num_robots=self.num_robots,
        )

        # Optional critic, built directly here as flat Model attributes -
        # matching xqc_learner.py's own pattern exactly (self.critic/
        # self.target_critic built inline in __init__, no wrapper object
        # at all - see conversation history: this replaces an earlier
        # design that instantiated one of xqc/agents/critics/{mlp,
        # xqc_style,transformer}'s own *CriticLearner classes via
        # composition; those classes have since been deleted entirely -
        # train_critic_sarsa.py's own standalone critic-only training now
        # uses xqc/agents/chunking/chunked_learner_sarsa.py's own
        # ChunkedCriticSARSALearner instead, a copy of THIS class's own
        # critic block with every actor-only piece removed - see
        # conversation history).
        #
        # critic_name (a plain config string, one of "mlp"/"xqc_style"/
        # "transformer") selects both the critic network architecture and
        # which module's own compute_q/update_critic/eval_critic this
        # class calls directly from update() below - the three variants'
        # own update/loss orchestration genuinely differs (dropout+HL-Gauss
        # vs BatchNorm+categorical/MSE, different extra config), so this is
        # the one place that branches on it. None (default) means no
        # critic - every existing agent config is unaffected.
        #
        # The critic gets its own SEPARATE vision/obs_encoder (built via
        # the same build_vision_and_obs_encoder call as the actor's own
        # self.vision/self.obs_encoder above, just with different keys/
        # optimizer) rather than sharing the actor's - "separate vision for
        # actor and critic" is the deliberately simple starting point (see
        # conversation history); shared or frozen-actor-vision modes are a
        # later generalization, not implemented here.
        #
        # Deliberately reuses low_dim_stats/self.raw_image_shape (already
        # computed above for the actor's own self.dataset) rather than
        # building a separate offline Dataset (chunked_qc_buffer.py) here -
        # the two would produce numerically identical obs_offset/obs_scale/
        # action_offset/action_scale (same underlying data, same
        # xqc/agents/common/dataset.py formula the replay buffer's own
        # compute_stats calls directly - see conversation history), so a
        # second buffer here would only be redundant construction cost, not
        # a genuinely different source of truth. The actor's own
        # self.action_offset/self.action_scale are reused for normalizing
        # the critic's batch actions too (update() below) rather than a
        # separate critic-specific copy, for the same reason. The actual
        # replay buffer (for SAMPLING training batches, not shape/stat
        # derivation) lives in train_parallel_chunking.py instead, matching
        # xqc_learner.py's own "batch passed in from outside" convention.
        self.critic_name = critic_name
        self.critic = None
        self.target_critic = None
        # Harmless placeholders for a critic-less learner (every existing
        # DP/FM/MF/... agent config) - update()'s own call to _update below
        # reads these attributes unconditionally every iteration (not just
        # when a critic exists), same as self.critic/self.target_critic
        # above, so they must exist either way.
        self.critic_vision: Dict[str, Model] = {}
        self.critic_target_vision: Dict[str, Model] = {}
        self.critic_obs_encoder = None
        self.critic_target_obs_encoder = None
        if critic_name is not None:
            if critic_name not in _CRITIC_FNS_REGISTRY:
                raise ValueError(f"Unknown critic_name={critic_name!r}, expected one of {list(_CRITIC_FNS_REGISTRY)}")
            self.critic_fns = _CRITIC_FNS_REGISTRY[critic_name]
            critic_kwargs = dict(critic_kwargs or {})
            critic_seed = critic_kwargs.pop("seed", seed)
            self.critic_camera_keys = tuple(critic_kwargs.pop("camera_keys", self.camera_keys))
            self.critic_random_crop = critic_kwargs.pop("random_crop", random_crop)
            self.critic_crop_shape = tuple(critic_kwargs.pop("crop_shape", self.crop_shape))
            critic_vision_encoder = critic_kwargs.pop("vision_encoder", vision_encoder)
            critic_vision_norm_type = critic_kwargs.pop("vision_norm_type", vision_norm_type)
            critic_vision_lr = critic_kwargs.pop("vision_lr", None)
            critic_lr = critic_kwargs.pop("critic_lr", 3e-4)
            # critic_pre_training_steps: popped here (not used to build
            # anything in this __init__ any more - see this file's own
            # critic_lr_schedule_fn construction below, now scoped to
            # num_interactions*updates_per_step for every agent instead -
            # explicit request, see conversation history) purely so it
            # doesn't leak into critic_kwargs wherever that dict gets used
            # elsewhere. Its own gating role (WHETHER the critic still gets
            # gradient updates during offline pretraining, independent of
            # what LR schedule shape those updates use) lives entirely in
            # train_parallel_chunking.py's own offline-loop bound, read
            # directly off cfg.agent.critic_kwargs, not through this local.
            critic_pre_training_steps = critic_kwargs.pop("critic_pre_training_steps", actor_pre_training_steps)
            critic_weight_decay = critic_kwargs.pop("weight_decay", 0.0)
            self.critic_tau = critic_kwargs.pop("tau", 0.005)
            self.critic_target_net_update_freq = critic_kwargs.pop("target_net_update_freq", 1)

            critic_rng = jax.random.PRNGKey(critic_seed)
            critic_rng, critic_vision_key, critic_obs_encoder_key, critic_key = jax.random.split(critic_rng, 4)
            critic_vision_dim = (
                len(self.critic_camera_keys) * vision_feature_dims[critic_vision_encoder]
                if self.critic_camera_keys else 0
            )
            critic_condition_dim = low_dim_obs_dim + critic_vision_dim

            # critic_tx/critic_encoder_tx: named + built together here, right
            # below each other - same convention as actor_tx/encoder_tx
            # above and xqc_learner.py's own actor_tx/critic_tx (built right
            # after LR-schedule-equivalent setup, before any Model
            # construction uses them). Can't sit literally next to actor_tx
            # at the very top of __init__ like xqc_learner.py's own
            # critic_tx does - critic_lr/critic_weight_decay live inside
            # critic_kwargs (a nested dict, only parsed once critic_name is
            # known to be set), not as flat top-level constructor args the
            # way xqc_learner.py's own critic_lr is - so this is the
            # earliest point they're actually available.
            if num_interactions is None:
                raise ValueError(
                    "Any critic-enabled agent needs num_interactions (train_parallel_chunking.py's own "
                    "int(cfg.max_steps/cfg.env.action_repeat), already passed into agent_kwargs for every "
                    "agent) to build its critic's own reference-matching linear LR schedule."
                )
            #
            # Linear decay (optax.linear_schedule), not the actor's own
            # warmup_cosine_decay_schedule - matches the critic's own
            # schedule shape on the xqcfd (main) branch's XQCLearner
            # (xqc/agents/xqc/xqc_learner.py's own lr_schedule_fn) - explicit
            # request, see conversation history: now the critic's own
            # schedule for EVERY agent (MF/DP/FM/MIP/NF/XQCFD/...), not just
            # XQCFDLearner - only the ACTOR's own schedule stays agent-
            # specific (XQCFDLearner's own linear-no-warmup branch above vs
            # every other agent's own cosine_warmup, kept deliberately
            # unchanged - explicit request). No warmup phase (unlike the
            # actor never had one for the critic either).
            #
            # _critic_schedule_span (explicit request - see conversation
            # history, and the actor's own _schedule_span comment above for
            # the full story of the bug this fixes): critic_pre_training_steps
            # + (num_interactions or 0) * updates_per_step - a SUM, not just
            # num_interactions*updates_per_step alone. agent.critic.step
            # (the real counter this schedule is queried with) is never
            # reset - it counts through critic_pre_training_steps worth of
            # offline updates FIRST, then keeps going into the online
            # phase's own updates - so the schedule's span has to cover
            # both, or it hits end_value early (confirmed directly from a
            # real run's own logged critic_lr curve flatlining tens of
            # thousands of steps before the run's actual end - the online-
            # only span was consistently smaller than what the counter
            # actually reaches by the time online finishes). end_value=
            # lr_end (was hardcoded 0.0 - lr_end is xqc_base_agent.yaml's
            # own default value, 3e-5, reused here as this class's own
            # default too so nothing changes for an agent that never sets
            # it). Stashed for reset_optimizers(online_steps=...) below -
            # same reasoning as the actor's own _actor_peak_lr/etc above.
            self._critic_peak_lr = critic_lr
            self._critic_vision_lr = critic_vision_lr
            self._critic_weight_decay = critic_weight_decay
            _critic_schedule_span = max(critic_pre_training_steps + (num_interactions or 0) * updates_per_step, 1)
            self.critic_lr_schedule_fn = optax.linear_schedule(
                init_value=critic_lr, end_value=lr_end, transition_steps=_critic_schedule_span,
            )
            critic_tx = optax.adamw(self.critic_lr_schedule_fn, weight_decay=critic_weight_decay)
            critic_encoder_tx = optax.adamw(
                self.critic_lr_schedule_fn if critic_vision_lr is None else optax.linear_schedule(
                    init_value=critic_vision_lr, end_value=lr_end, transition_steps=_critic_schedule_span,
                ),
                weight_decay=critic_weight_decay,
            )

            self.critic_vision, self.critic_obs_encoder = build_vision_and_obs_encoder(
                self.critic_camera_keys, self.raw_image_shape, critic_vision_encoder, critic_vision_norm_type,
                spatial_softmax_num_kp, spatial_softmax_feature_dim, vit_image_size,
                self.critic_random_crop, self.critic_crop_shape, critic_condition_dim, To,
                critic_vision_key, critic_obs_encoder_key, critic_encoder_tx,
            )
            self.critic_target_vision, self.critic_target_obs_encoder = build_vision_and_obs_encoder(
                self.critic_camera_keys, self.raw_image_shape, critic_vision_encoder, critic_vision_norm_type,
                spatial_softmax_num_kp, spatial_softmax_feature_dim, vit_image_size,
                self.critic_random_crop, self.critic_crop_shape, critic_condition_dim, To,
                critic_vision_key, critic_obs_encoder_key, None,
            )
            self.critic_target_vision = {
                cam: self.critic_target_vision[cam].replace(params=self.critic_vision[cam].params)
                for cam in self.critic_vision
            }
            self.critic_target_obs_encoder = self.critic_target_obs_encoder.replace(params=self.critic_obs_encoder.params)

            self.critic_obs_offset = jnp.asarray(np.concatenate([low_dim_stats.obs_offset, np.zeros(critic_vision_dim, np.float32)]))
            self.critic_obs_scale = jnp.asarray(np.concatenate([low_dim_stats.obs_scale, np.ones(critic_vision_dim, np.float32)]))

            use_weight_norm = False
            self._critic_step_extra_kwargs: Dict[str, Any] = {}
            # compute_q's own extra kwargs, distinct from update_critic's above:
            # xqc_style's compute_q takes NO critic_loss at all (its
            # docstring: critic_model's own __call__ always returns an
            # already-decoded scalar Q regardless of critic_loss mode, so no
            # critic_loss-specific branching is needed there) - only
            # transformer's compute_q needs extras (support/sigma), matching
            # what its own compute_q signature actually accepts.
            self._critic_compute_q_extra_kwargs: Dict[str, Any] = {}
            # eval_critic's own extra kwargs - same per-variant shape as
            # _critic_step_extra_kwargs (xqc_style needs critic_loss,
            # transformer needs support/sigma, mlp needs neither), just
            # without use_weight_norm (eval_critic never applies weight
            # norm - a no-gradient pass has nothing to renormalize). Mirrors
            # chunked_learner_sarsa.py's own construction exactly - see
            # conversation history (compute_critic_val_loss below).
            self._critic_eval_extra_kwargs: Dict[str, Any] = {}
            self.critic_support = None
            self.critic_sigma = None
            if critic_name == "mlp":
                critic_def = critic_net.VMapCritic(
                    hidden_dims=tuple(critic_kwargs.pop("hidden_dims_critic", (512, 512, 512, 512))),
                    n_outputs=1,
                    max_v=critic_kwargs.pop("max_v", 5.0),
                    min_v=critic_kwargs.pop("min_v", -5.0),
                    n_critics=critic_kwargs.pop("n_critics", 2),
                    pre_activation_bn=False,
                    use_layer_norm=False,
                    use_batch_norm=False,
                    skip_connections=False,
                )
                # update_critic itself requires use_weight_norm (no
                # default) - always False for this baseline, previously
                # hardcoded as a literal inside the now-removed
                # jit_critic_step rather than threaded through this dict;
                # update_critic_fn's own binding (_build_critic_bound_fns)
                # needs it here now instead - see conversation history.
                self._critic_step_extra_kwargs = {"use_weight_norm": False}
            elif critic_name == "xqc_style":
                critic_loss = critic_kwargs.pop("critic_loss", "categorical")
                use_weight_norm = critic_kwargs.pop("use_weight_norm", True)
                n_outputs = {"categorical": 101, "mse": 1}[critic_loss]
                critic_def = critic_net.VMapCritic(
                    hidden_dims=tuple(critic_kwargs.pop("hidden_dims_critic", (512, 512, 512, 512))),
                    n_outputs=n_outputs,
                    max_v=critic_kwargs.pop("max_v", 5.0),
                    min_v=critic_kwargs.pop("min_v", -5.0),
                    n_critics=critic_kwargs.pop("n_critics", 2),
                    pre_activation_bn=critic_kwargs.pop("pre_activation_bn", True),
                    use_layer_norm=critic_kwargs.pop("use_layer_norm", False),
                    use_batch_norm=critic_kwargs.pop("use_batch_norm", True),
                    skip_connections=critic_kwargs.pop("skip_connections", False),
                )
                self._critic_step_extra_kwargs = {"use_weight_norm": use_weight_norm, "critic_loss": critic_loss}
                self._critic_eval_extra_kwargs = {"critic_loss": critic_loss}
            elif critic_name == "transformer":
                min_v = critic_kwargs.pop("min_v", -5.0)
                max_v = critic_kwargs.pop("max_v", 5.0)
                num_bins = critic_kwargs.pop("num_bins", 101)
                sigma = critic_kwargs.pop("sigma", None)
                self.critic_support = jnp.linspace(min_v, max_v, num_bins + 1)
                self.critic_sigma = sigma if sigma is not None else (max_v - min_v) / (4 * num_bins)
                critic_def = VMapTransformerCritic(
                    To=To, h=execution_horizon,
                    d_model=critic_kwargs.pop("d_model", 256),
                    nhead=critic_kwargs.pop("nhead", 8),
                    num_encoder_layers=critic_kwargs.pop("num_encoder_layers", 3),
                    dim_feedforward=critic_kwargs.pop("dim_feedforward", 256),
                    dropout=critic_kwargs.pop("dropout", 0.1),
                    num_bins=num_bins,
                    n_critics=critic_kwargs.pop("n_critics", 2),
                )
                self._critic_step_extra_kwargs = {"support": self.critic_support, "sigma": self.critic_sigma}
                self._critic_compute_q_extra_kwargs = {"support": self.critic_support, "sigma": self.critic_sigma}
                self._critic_eval_extra_kwargs = {"support": self.critic_support, "sigma": self.critic_sigma}

            if critic_name == "transformer":
                dummy_critic_obs = jnp.zeros((1, To, critic_condition_dim))
                dummy_critic_actions = jnp.zeros((1, execution_horizon, act_dim))
            else:
                dummy_critic_obs = jnp.zeros((1, To * critic_condition_dim))
                dummy_critic_actions = jnp.zeros((1, execution_horizon * act_dim))

            self.critic = Model.create(
                critic_def, inputs=[critic_key, dummy_critic_obs, dummy_critic_actions, False],
                tx=critic_tx,
            )
            self.target_critic = Model.create(critic_def, inputs=[critic_key, dummy_critic_obs, dummy_critic_actions, False])
            if use_weight_norm:
                self.critic = norm_network(self.critic)
            self.target_critic = self.target_critic.replace(params=self.critic.params)

            self._critic_init_weight_norm = float(optax.global_norm(self.critic.params))
            if critic_name != "transformer":
                init_split = split_dense_and_batchnorm_norms(self.critic.params)
                self._critic_init_dense_norm = float(init_split["dense_norm"])
                self._critic_init_batchnorm_norm = float(init_split["batchnorm_norm"])
            self._critic_init_layer_norms = {k: float(v) for k, v in per_layer_norms(self.critic.params).items()}

            self._build_critic_bound_fns()

        # Called once, here, AFTER the critic block above (not right after
        # self.network is built) - update_actor_fn (built inside, when a
        # critic exists) needs self.compute_q_fn, which only exists once
        # _build_critic_bound_fns has already run - see conversation
        # history.
        self._build_actor_bound_fns()

    def _build_actor_bound_fns(self) -> None:
        """(Re)builds self.update_actor_bc_fn/val_loss_fn/infer_fn (plus
        update_actor_fn, when a critic exists), binding obs_offset/
        obs_scale/action_offset/action_scale/camera_keys/random_crop/
        crop_shape once via functools.partial - these never change for
        this learner's lifetime EXCEPT across a load() call (which
        restores obs_offset/obs_scale/action_offset/action_scale from a
        checkpoint's own meta.npz), so load() calls this again afterward
        rather than leaving these bound functions stale - see conversation
        history. camera_keys/random_crop/crop_shape must stay static for
        jax.jit while obs_offset/obs_scale are traced arrays - binding both
        via one closure (rather than bundling them into one config object
        passed INTO a jitted function) sidesteps that mismatch entirely.

        update_actor_fn is only built here `if self.critic is not None`
        (rather than unconditionally, like the other three) since it needs
        self.compute_q_fn - a critic-derived object that doesn't exist
        otherwise. It's an ACTOR function (calls update_actor, defined in
        actor.py) despite depending on a critic-built object, so it
        belongs here, not in _build_critic_bound_fns - see conversation
        history. This is exactly why __init__ calls
        _build_critic_bound_fns BEFORE this method (see its own call
        site): self.compute_q_fn must already exist by the time this runs.
        update_actor itself no longer takes critic_obs_offset/
        critic_obs_scale/critic_camera_keys at all (self.compute_q_fn
        already carries them) - see actor.py's own update_actor docstring.
        Any positional argument added at a call site AFTER a partial-bound
        keyword must itself be passed by keyword too - mixing the two
        causes a "multiple values for argument" collision (see
        conversation history)."""
        self.update_actor_bc_fn = functools.partial(
            update_actor_bc,
            obs_offset=self.obs_offset, obs_scale=self.obs_scale,
            action_offset=self.action_offset, action_scale=self.action_scale,
            camera_keys=self.camera_keys, random_crop=self.random_crop, crop_shape=self.crop_shape,
            grad_clip_norm=self.grad_clip_norm, mask_reference_region=self.mask_reference_region,
            use_weight_norm=self.use_weight_norm,
        )
        self.val_loss_fn = functools.partial(
            val_loss,
            obs_offset=self.obs_offset, obs_scale=self.obs_scale, camera_keys=self.camera_keys,
            action_offset=self.action_offset, action_scale=self.action_scale,
            mask_reference_region=self.mask_reference_region,
            # network_accepts_train (explicit request - see conversation
            # history and val_loss's own matching docstring): XQCFD's own
            # .loss() has no `train` parameter at all - same
            # type(self).__name__ gate already used for
            # network_accepts_temperature elsewhere in this file.
            network_accepts_train=type(self).__name__ != "XQCFDLearner",
        )
        self.infer_fn = functools.partial(
            infer,
            camera_keys=self.camera_keys, obs_offset=self.obs_offset, obs_scale=self.obs_scale,
            action_offset=self.action_offset, action_scale=self.action_scale,
            random_crop=self.random_crop, crop_shape=self.crop_shape,
            # network_accepts_temperature (explicit request - see actor.py's
            # own infer() docstring): only XQCFD's own .sample() accepts a
            # temperature kwarg at all - every other actor's .sample() would
            # raise TypeError if passed one.
            network_accepts_temperature=type(self).__name__ == "XQCFDLearner",
        )
        if self.critic is not None:
            self.update_actor_fn = functools.partial(
                update_actor,
                compute_q_fn=self.compute_q_fn,
                obs_offset=self.obs_offset, obs_scale=self.obs_scale,
                camera_keys=self.camera_keys, random_crop=self.random_crop, crop_shape=self.crop_shape,
                Ta=self.Ta, execution_horizon=self.execution_horizon, act_dim=self.act_dim,
                grad_clip_norm=self.grad_clip_norm,
                # Safe to bind once here (unlike prior_network_model/etc,
                # passed fresh at each _update call instead - see its own
                # docstring): these two never change after construction.
                wasserstein_regularization=self.wasserstein_regularization,
                actor_name=type(self).__name__,
                mask_reference_region=self.mask_reference_region,
                wasserstein_offline_only=self.wasserstein_offline_only,
                actor_weight=self.actor_weight,
                kl_regularization=self.kl_regularization,
                use_weight_norm=self.use_weight_norm,
                bc_regularization=self.bc_regularization,
            )

    def _build_critic_bound_fns(self) -> None:
        """(Re)builds self.compute_q_fn/update_critic_fn, binding
        critic_obs_offset/critic_obs_scale/critic_camera_keys/
        critic_random_crop/critic_crop_shape (plus each variant's own extra
        kwargs) once via functools.partial - same reasoning as
        _build_actor_bound_fns above, just for the critic's own functions.
        Only ever called once, from __init__ - unlike the actor's own
        bound fns, nothing here is currently reloadable from a checkpoint
        (no critic save/load exists yet), so there's no load()-triggered
        staleness to guard against.

        Also builds self.critic_eval_fn (eval_critic) - used by
        compute_critic_val_loss below for a held-out, no-gradient critic
        diagnostic on the fused train_parallel_chunking.py path, mirroring
        xqc/agents/chunking/chunked_learner_sarsa.py's own identical
        construction (train_critic_sarsa.py's standalone critic-only
        training) - see conversation history.

        update_critic_fn binds update_critic directly (there is no
        jit_critic_step-style wrapper anywhere any more - see conversation
        history: the old per-variant jit_critic_step functions, which used
        to exist only for train_critic_sarsa.py's own standalone
        critic-only training, have been deleted; that script now runs
        through xqc/agents/chunking/chunked_learner_sarsa.py's own
        ChunkedCriticSARSALearner instead, whose own critic step does this
        same target-update+diagnostics sequence inline, exactly the way
        THIS file's own fused _update below does). chunked_learner.py's own
        fused _update below calls update_critic directly and does the
        target-update + diagnostics inline itself, matching xqc_learner.py's
        own _update exactly, which likewise calls update_critic directly and
        inlines `tau = jnp.where(...); target_update(...)` rather than
        delegating to a separate wrapper function). tau/
        target_net_update_freq are NOT bound here (unlike the old
        critic_step_fn) - _update needs the PRE-critic-step critic.step to
        compute them, so it reads self.critic_tau/critic_target_net_update_freq
        directly instead.

        Does NOT build update_actor_fn (an ACTOR function, despite needing
        self.compute_q_fn built here) - that lives in
        _build_actor_bound_fns instead, called right after this one - see
        conversation history and that method's own docstring."""
        self.compute_q_fn = functools.partial(
            self.critic_fns.compute_q,
            obs_offset=self.critic_obs_offset, obs_scale=self.critic_obs_scale, camera_keys=self.critic_camera_keys,
            **self._critic_compute_q_extra_kwargs,
        )
        self.update_critic_fn = functools.partial(
            self.critic_fns.update_critic,
            obs_offset=self.critic_obs_offset, obs_scale=self.critic_obs_scale, camera_keys=self.critic_camera_keys,
            random_crop=self.critic_random_crop, crop_shape=self.critic_crop_shape,
            **self._critic_step_extra_kwargs,
        )
        self.critic_eval_fn = functools.partial(
            self.critic_fns.eval_critic,
            obs_offset=self.critic_obs_offset, obs_scale=self.critic_obs_scale, camera_keys=self.critic_camera_keys,
            random_crop=self.critic_random_crop, crop_shape=self.critic_crop_shape,
            **self._critic_eval_extra_kwargs,
        )

    def _rl_not_implemented_msg(self) -> str:
        return (
            f"{type(self).__name__} only implements BC pretraining (network.loss on offline "
            "demonstrations). Online RL fine-tuning is not implemented - this generative "
            "denoising/flow-map network provides no log_prob at all (it's not a distribution "
            "with a tractable density), so this would need a genuinely different technique "
            "before RL fine-tuning could use anything log-prob-based here."
        )


    def update(
        self, batch=None, num_updates: int = 1, pre_train: bool = True, update_actor: bool = True, offline_mask=None,
    ) -> InfoDict:
        """`batch`, when given, is the critic's own externally-pre-sampled
        ChunkedBatch, stacked `num_updates` deep along a LEADING axis (one
        full batch per update iteration - the caller's job, e.g.
        train_parallel_chunking.py's own) - matches xqc_learner.py's own
        _do_multiple_updates convention exactly: that function's own
        `batches` argument is likewise pre-sampled `num_updates` deep by
        the caller and sliced one iteration at a time
        (jax.tree.map(lambda x: jnp.take(x, i, axis=1), batches)) rather
        than critic and actor getting a different number of updates per
        call - the only difference is axis: axis=1 there skips over
        xqc_learner.py's own num_seeds vmap axis (axis 0), which this
        class has no equivalent of (single-seed only - see conversation
        history), so the slice axis here is 0. critic_infos and
        actor_infos are flat-merged into one InfoDict, same as
        xqc_learner.py's own _update ({**critic_info, **actor_info, ...}).

        `pre_train` selects which actor update runs, matching
        xqc_learner.py's own _update exactly (`update_actor_bc` when True,
        `update_actor` when False): pre_train=True samples self.dataset
        internally for a BC step (unaffected by `batch` - can't share it
        with the critic, different windowing convention); pre_train=False
        instead runs the RL actor update on THIS SAME externally-passed
        batch's own current observation (no reference-matching windowing to
        preserve once there's no BC target to match - see conversation
        history), and requires critic_name to have been set (there's
        nothing for update_actor to query otherwise). `batch=None` (the
        default) skips the critic step entirely - every existing agent
        config (no critic_name) is unaffected.

        Actor and critic now update together in ONE call to the
        module-level _update function above (critic first, then the actor
        using that SAME-STEP fresh critic) - matching xqc_learner.py's own
        _update exactly, see that function's own docstring for the full
        rationale. This method itself stays a thin per-iteration Python
        loop (num_updates calls to _update) - unlike XQCLearner.update(),
        which makes ONE call covering every iteration via
        _do_multiple_updates - see conversation history: that loop-fusion
        is a deliberately separate, later change.

        update_actor=False: skip the actor's own gradient step entirely
        this call (identity pass-through - see _update's own docstring) -
        lets the driver loop keep calling update() past the actor's own
        actor_pre_training_steps budget purely to let the critic keep
        training toward ITS OWN independent critic_pre_training_steps (see
        conversation history: the two can now differ, e.g. when resuming a
        run to let the critic catch up while the actor stays frozen). Also
        skips sampling self.dataset for actor_batch entirely (nothing would
        use it) rather than wastefully drawing and discarding one.

        offline_mask: (num_updates, batch) or None - wasserstein_offline_only's
        own masking input (see actor.py's own update_actor docstring), built
        by train_parallel_chunking.py at its own online/offline concat site
        and sliced one iteration at a time below, same convention as `batch`
        itself. None (default, and always for pre_train=True/BC calls, which
        never touch wasserstein_regularization at all) is passed straight
        through to _update() unsliced."""
        if not pre_train and self.critic is None:
            raise NotImplementedError(self._rl_not_implemented_msg())
        if not pre_train and batch is None:
            raise ValueError("update(pre_train=False) needs a batch - it's the actor's own RL update input, not just the critic's.")
        if batch is not None and self.critic is None:
            raise RuntimeError(f"{type(self).__name__} was built without critic_name - no critic to update.")

        infos: InfoDict = {}
        for i in range(num_updates):
            batch_i = jax.tree.map(lambda x: x[i], batch) if batch is not None else None
            offline_mask_i = offline_mask[i] if offline_mask is not None else None
            if batch_i is not None:
                batch_i = batch_i._replace(
                    actions=(batch_i.actions - self.action_offset) / self.action_scale,
                    next_actions=(batch_i.next_actions - self.action_offset) / self.action_scale,
                )

            if pre_train and update_actor:
                # gamma=1.0: rewards/discount are RL-only fields the actor
                # never reads (see conversation history) - a harmless
                # placeholder.
                actor_batch = self.dataset.sample(
                    self.batch_size, To=self.To, h=self.execution_horizon, gamma=1.0,
                    camera_keys=self.camera_keys, rng=self._dataset_rng, Ta=self.Ta,
                )
                actor_batch = actor_batch._replace(
                    observations=jnp.asarray(actor_batch.observations),
                    actions=jnp.asarray(actor_batch.actions),
                    # rescale_images: self.dataset stores raw uint8 [0,255]
                    # (chunked_qc_buffer.py's own convention) - encode_condition
                    # expects float32 [-1,1] (see its own docstring). The
                    # critic's own path already does this via rescale_images;
                    # the actor's offline BC batch was missing it (see
                    # conversation history) - live rollouts are unaffected,
                    # ChunkExecutor already rescales images itself.
                    images=rescale_images(actor_batch.images),
                )
            else:
                actor_batch = None

            # critic_bootstrap_use_ema (explicit request - see conversation
            # history and this class's own __init__ docstring for the full
            # rationale): built fresh each inner iteration (self.network_ema_*
            # itself updates at the end of THIS loop body, below) rather than
            # once per update() call, same granularity as self.network/
            # self.vision/self.obs_encoder themselves. .replace() only swaps
            # params/batch_stats - apply_fn/tx/opt_state/step stay the live
            # Model's own (_update's own bootstrap_network_model fallback
            # logic, and bootstrap_next_actions itself, never read those).
            # None (default, toggle off) reproduces today's behavior exactly
            # - _update's own `critic_bootstrap_network_model if ... is not
            # None else network_model` falls back to the live network_model/
            # vision_models/obs_encoder_model, unchanged.
            if self.critic_bootstrap_use_ema:
                critic_bootstrap_network_model = self.network.replace(
                    params=self.network_ema_params, batch_stats=self.network_ema_batch_stats
                )
                critic_bootstrap_vision_models = {
                    cam: self.vision[cam].replace(params=self.vision_ema_params[cam]) for cam in self.vision
                }
                critic_bootstrap_obs_encoder_model = self.obs_encoder.replace(params=self.obs_encoder_ema_params)
            else:
                critic_bootstrap_network_model = None
                critic_bootstrap_vision_models = None
                critic_bootstrap_obs_encoder_model = None

            (
                self.rng, self.network, self.vision, self.obs_encoder,
                self.critic, self.target_critic, self.critic_vision, self.critic_target_vision,
                self.critic_obs_encoder, self.critic_target_obs_encoder,
                infos,
            ) = _update(
                self.rng,
                self.network, self.vision, self.obs_encoder,
                self.critic, self.target_critic, self.critic_vision, self.critic_target_vision,
                self.critic_obs_encoder, self.critic_target_obs_encoder,
                actor_batch, batch_i,
                obs_offset=self.obs_offset, obs_scale=self.obs_scale,
                critic_tau=self.critic_tau if self.critic is not None else 0.0,
                critic_target_net_update_freq=self.critic_target_net_update_freq if self.critic is not None else 1,
                update_actor_bc_fn=self.update_actor_bc_fn,
                update_actor_fn=getattr(self, "update_actor_fn", None),
                update_critic_fn=getattr(self, "update_critic_fn", None),
                critic_name=self.critic_name, pre_train=pre_train,
                camera_keys=self.camera_keys, Ta=self.Ta,
                execution_horizon=self.execution_horizon, act_dim=self.act_dim,
                update_actor=update_actor,
                pretrain_use_sarsa=self.pretrain_use_sarsa,
                # wasserstein_regularization's own teacher snapshot - see
                # actor.py's own update_actor docstring. Read fresh off
                # self each call (not bound via functools.partial anywhere)
                # since these get replaced once, at the offline->online
                # transition, and getattr defaults keep this safe for
                # agents built before that attribute existed (old
                # checkpoints - see chunked_learner.py's own load()).
                prior_network_model=getattr(self, "prior_network", None),
                prior_vision_models=getattr(self, "prior_vision", None),
                prior_obs_encoder_model=getattr(self, "prior_obs_encoder", None),
                offline_mask=offline_mask_i,
                critic_bootstrap_network_model=critic_bootstrap_network_model,
                critic_bootstrap_vision_models=critic_bootstrap_vision_models,
                critic_bootstrap_obs_encoder_model=critic_bootstrap_obs_encoder_model,
                critic_bootstrap_best_of_n=self.critic_bootstrap_best_of_n,
            )

            if "weight_norm" in infos:
                # Ratio-to-init diagnostics - needs self._critic_init_* (plain
                # Python floats, computed once at construction), so this
                # stays in Python rather than moving inside _update.
                infos["weight_norm_ratio"] = float(infos["weight_norm"]) / self._critic_init_weight_norm
                if "dense_norm" in infos:
                    infos["dense_norm_ratio"] = float(infos["dense_norm"]) / max(self._critic_init_dense_norm, 1e-8)
                    infos["batchnorm_norm_ratio"] = float(infos["batchnorm_norm"]) / max(self._critic_init_batchnorm_norm, 1e-8)
                infos.update(layer_norm_ratios(infos, self._critic_init_layer_norms))

            if "actor_weight_norm" in infos:
                infos["actor_weight_norm_ratio"] = float(infos["actor_weight_norm"]) / self._actor_init_weight_norm
                infos["actor_dense_norm_ratio"] = float(infos["actor_dense_norm"]) / max(self._actor_init_dense_norm, 1e-8)
                infos["actor_batchnorm_norm_ratio"] = (
                    float(infos["actor_batchnorm_norm"]) / max(self._actor_init_batchnorm_norm, 1e-8)
                )
                infos.update(layer_norm_ratios(infos, self._actor_init_layer_norms, prefix="actor_layer_norm"))

            if self.ema_rate < 1:
                decay = _ema_decay(int(self.network.step), self.ema_schedule, self.ema_rate, self.ema_power, self.ema_max_value)
                self.network_ema_params = _ema_update(self.network_ema_params, self.network.params, decay)
                if self.network.batch_stats is not None:
                    self.network_ema_batch_stats = _ema_update(
                        self.network_ema_batch_stats, self.network.batch_stats, decay
                    )
                self.vision_ema_params = {
                    cam: _ema_update(self.vision_ema_params[cam], self.vision[cam].params, decay) for cam in self.vision
                }
                self.obs_encoder_ema_params = _ema_update(self.obs_encoder_ema_params, self.obs_encoder.params, decay)
        return infos

    def save(self, directory: str) -> None:
        """One orbax checkpoint per component (network, each vision camera,
        obs_encoder, and the EMA copies), plus a small meta.npz for rng and
        normalization stats - matches xqc_learner.py's own per-named-Model
        save() pattern rather than the single combined tree used before
        this refactor. Old-format checkpoints need
        scripts/convert_chunked_checkpoint_to_model_format.py first.

        Also saves the critic (critic/target_critic/critic_vision/
        critic_target_vision/critic_obs_encoder/critic_target_obs_encoder),
        when self.critic is not None - xqc_learner.py's own
        _NETWORK_NAMES includes "critic"/"target_critic" right alongside
        "actor" unconditionally (xqc always has a critic); previously this
        method saved the actor only, so resuming a critic-enabled run from
        a checkpoint silently reinitialized the critic from scratch - see
        conversation history."""
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        checkpointer = ocp.PyTreeCheckpointer()
        for name in _MODEL_NAMES:
            checkpointer.save(str(directory / name), _model_checkpoint_dict(getattr(self, name)), force=True)
        for cam, model in self.vision.items():
            checkpointer.save(str(directory / f"vision_{cam}"), _model_checkpoint_dict(model), force=True)
        for cam, model in self.prior_vision.items():
            checkpointer.save(str(directory / f"prior_vision_{cam}"), _model_checkpoint_dict(model), force=True)
        checkpointer.save(
            str(directory / "ema"),
            {
                "network": self.network_ema_params,
                "network_batch_stats": self.network_ema_batch_stats,
                "vision": self.vision_ema_params,
                "obs_encoder": self.obs_encoder_ema_params,
            },
            force=True,
        )
        if self.critic is not None:
            for name in _CRITIC_MODEL_NAMES:
                checkpointer.save(str(directory / name), _model_checkpoint_dict(getattr(self, name)), force=True)
            for cam, model in self.critic_vision.items():
                checkpointer.save(str(directory / f"critic_vision_{cam}"), _model_checkpoint_dict(model), force=True)
            for cam, model in self.critic_target_vision.items():
                checkpointer.save(str(directory / f"critic_target_vision_{cam}"), _model_checkpoint_dict(model), force=True)
        np.savez(
            directory / "meta.npz",
            rng=np.asarray(self.rng),
            obs_offset=np.asarray(self.obs_offset),
            obs_scale=np.asarray(self.obs_scale),
            action_offset=np.asarray(self.action_offset),
            action_scale=np.asarray(self.action_scale),
            **(
                {
                    "critic_obs_offset": np.asarray(self.critic_obs_offset),
                    "critic_obs_scale": np.asarray(self.critic_obs_scale),
                }
                if self.critic is not None else {}
            ),
        )

    def load(self, directory: str) -> None:
        directory = Path(directory).resolve()
        checkpointer = ocp.PyTreeCheckpointer()
        for name in _MODEL_NAMES:
            # prior_network/prior_obs_encoder are new (see conversation
            # history) - a checkpoint saved before this change simply won't
            # have them on disk. Skip restoring (leaving __init__'s freshly-
            # initialized value in place) rather than crash, same defensive
            # pattern critic_checkpoint_exists below already uses for the
            # critic's own backward-compatible case.
            if not (directory / name).exists():
                continue
            model = getattr(self, name)
            restored = checkpointer.restore(str(directory / name), item=_model_checkpoint_dict(model))
            setattr(self, name, model.replace(**restored))
        for cam in self.vision:
            restored_vision = checkpointer.restore(str(directory / f"vision_{cam}"), item=_model_checkpoint_dict(self.vision[cam]))
            self.vision[cam] = self.vision[cam].replace(**restored_vision)
        for cam in self.prior_vision:
            if not (directory / f"prior_vision_{cam}").exists():
                continue
            restored_prior_vision = checkpointer.restore(
                str(directory / f"prior_vision_{cam}"), item=_model_checkpoint_dict(self.prior_vision[cam])
            )
            self.prior_vision[cam] = self.prior_vision[cam].replace(**restored_prior_vision)
        restored_ema = checkpointer.restore(
            str(directory / "ema"),
            item={
                "network": self.network_ema_params,
                "network_batch_stats": self.network_ema_batch_stats,
                "vision": self.vision_ema_params,
                "obs_encoder": self.obs_encoder_ema_params,
            },
        )
        self.network_ema_params = restored_ema["network"]
        self.network_ema_batch_stats = restored_ema["network_batch_stats"]
        self.vision_ema_params = restored_ema["vision"]
        self.obs_encoder_ema_params = restored_ema["obs_encoder"]

        # self.critic is not None means THIS agent was constructed with a
        # critic - but the checkpoint on disk might predate that (e.g.
        # resuming a pure-BC checkpoint saved before critic_name was ever
        # set, to attach a fresh critic for RL fine-tuning without redoing
        # BC pretraining - see conversation history). Check the checkpoint
        # ITSELF has critic data (directory/"critic" exists) before trying
        # to restore it, rather than assuming self.critic is not None means
        # it does - otherwise this crashes on a checkpoint that never saved
        # a critic at all, instead of just leaving the freshly-initialized
        # one (from __init__) in place.
        critic_checkpoint_exists = self.critic is not None and (directory / "critic").exists()
        if critic_checkpoint_exists:
            for name in _CRITIC_MODEL_NAMES:
                model = getattr(self, name)
                restored = checkpointer.restore(str(directory / name), item=_model_checkpoint_dict(model))
                setattr(self, name, model.replace(**restored))
            for cam in self.critic_vision:
                restored_cv = checkpointer.restore(str(directory / f"critic_vision_{cam}"), item=_model_checkpoint_dict(self.critic_vision[cam]))
                self.critic_vision[cam] = self.critic_vision[cam].replace(**restored_cv)
            for cam in self.critic_target_vision:
                restored_ctv = checkpointer.restore(str(directory / f"critic_target_vision_{cam}"), item=_model_checkpoint_dict(self.critic_target_vision[cam]))
                self.critic_target_vision[cam] = self.critic_target_vision[cam].replace(**restored_ctv)

        meta = np.load(directory / "meta.npz")
        self.rng = jnp.array(meta["rng"])
        self.obs_offset = jnp.array(meta["obs_offset"])
        self.obs_scale = jnp.array(meta["obs_scale"])
        self.action_offset = jnp.array(meta["action_offset"])
        self.action_scale = jnp.array(meta["action_scale"])
        if critic_checkpoint_exists:
            self.critic_obs_offset = jnp.array(meta["critic_obs_offset"])
            self.critic_obs_scale = jnp.array(meta["critic_obs_scale"])
        if self.critic is not None:
            # Rebuild compute_q_fn/update_critic_fn - against the just-
            # restored critic_obs_offset/critic_obs_scale if the checkpoint
            # had them, or the freshly-initialized __init__ ones otherwise
            # (critic_checkpoint_exists False - see above) - BEFORE
            # _build_actor_bound_fns below (update_actor_fn needs the fresh
            # compute_q_fn) - same ordering as __init__ itself.
            self._build_critic_bound_fns()
        # obs_offset/obs_scale/action_offset/action_scale just changed -
        # rebuild the functions bound to their PREVIOUS values (see
        # _build_actor_bound_fns's own docstring) so they don't stay stale.
        self._build_actor_bound_fns()

    def _data_dependent_network_kwargs(self, network_extra_kwargs: dict, normalized_actions: np.ndarray) -> dict:
        """Hook for subclasses whose network needs fields derived from the
        dataset (e.g. MFLearner's squash_scale). normalized_actions: every
        transition's action, (N, act_dim), already normalized exactly as the
        network sees it. Default: network_extra_kwargs unchanged."""
        return network_extra_kwargs

    def _maybe_delay_tx(self, tx: optax.GradientTransformation, undelayed_until: int = 0) -> optax.GradientTransformation:
        """Wraps `tx` in optax.conditionally_mask (TD3-style delayed actor
        updates - see self.policy_delay's own __init__ docstring) only when
        policy_delay>1 - the default (1, every step) returns `tx` completely
        UNWRAPPED rather than wrapped-but-numerically-a-no-op, specifically
        so every existing agent's actor/encoder opt_state pytree shape (and
        hence checkpoint compatibility - see load()) stays exactly what it
        was before policy_delay existed.

        undelayed_until (explicit request - see conversation history: a
        real regression caught before it shipped - see this method's own
        git history/conversation history for the full story of what went
        wrong first): 0 (default) means the delay is active from the very
        first call, matching XQCFDLearner's own reference convention
        (TD3-style delay active during pre-training too, not just online -
        see conf/actor/xqcfd_state.yaml's own policy_delay field comment).
        Every OTHER agent (MF included) wants pre-training completely
        undelayed on purpose (every offline BC step is a real actor
        update, regardless of policy_delay) and the configured delay
        active once online RL starts - __init__ passes
        undelayed_until=actor_pre_training_steps for those. This is ONE
        optax transformation, built once, that already knows how to do
        both - no separate rebuild at the offline->online transition
        needed (confirmed directly: optax.conditionally_mask's own
        predicate is called with a step counter that increments on EVERY
        call, masked-off ones included, continuously from construction -
        never resets - so `step < undelayed_until` correctly identifies
        "still inside the known offline budget" using the exact same
        continuous counter the LR schedule's own _schedule_span already
        relies on, see this file's own module docstring for that
        decision)."""
        if self.policy_delay <= 1:
            return tx
        if undelayed_until <= 0:
            return optax.conditionally_mask(tx, lambda step: jnp.mod(step, self.policy_delay) == 0)
        return optax.conditionally_mask(
            tx, lambda step: jnp.logical_or(step < undelayed_until, jnp.mod(step, self.policy_delay) == 0)
        )

    def reset_optimizers(self, online_steps: Optional[int] = None) -> None:
        """NOT CALLED from train_parallel_chunking.py's offline->online
        transition any more, for ANY agent (explicit request - see
        conversation history: superseded by __init__'s own _schedule_span,
        which sizes every agent's lr_schedule_fn to its correct combined
        offline+online budget from construction, so there's nothing left
        that NEEDS resetting at the transition). Kept, fully working,
        purely so this can be reinstated later without rewriting it from
        scratch if that decision gets revisited - the rest of this
        docstring describes what it actually does, unchanged.

        Resets every live (non-target) Model's own optimizer STATE
        (momentum/variance accumulators) back to a fresh, zero-initialized
        start, while leaving its trained params exactly as they are -
        matches xqc_learner.py's own reset_optimizers, which re-runs
        init_models(...) from scratch and keeps only the freshly-built
        .tx object off each Model, discarding the fresh params it also
        produced.

        online_steps=None (default): just re-inits opt_state on each
        Model's OWN EXISTING tx object - an optax.GradientTransformation is
        stateless config (see conversation history), so this produces a
        fresh, zeroed opt_state for the ALREADY-trained params without
        rebuilding tx or computing a wasted fresh param init just to throw
        it away. The schedule embedded in that tx (if any) keeps whatever
        decay_steps it was originally built with.

        online_steps=<int>: additionally REBUILDS the ACTOR's own tx
        (ONLY - see this method's own critic-section comment further down
        for why the critic is now always left alone instead, explicit
        request) with a fresh schedule whose decay/transition span is this
        value instead of actor_pre_training_steps - for the offline->online
        transition specifically. See conversation history for why this
        matters: actor_pre_training_steps is the OFFLINE budget: reusing
        that same schedule (same decay_steps) for online fine-tuning
        conflates two unrelated quantities - online might run for far fewer
        (LR barely decays, stays near-peak the whole time) or far more (LR
        bottoms out early, actor stops learning for the remainder) updates
        than the offline phase took. Passing the actual, known online
        update budget here instead makes the restart (still to step 0 - see
        conversation history, that part was never the problem) decay over
        the RIGHT span.

        The ONLINE actor schedule shape is optax.linear_schedule (matching
        xqc_learner.py's own online schedule on the xqcfd/main branch),
        even though the actor's OFFLINE (BC pre-training) schedule stays
        warmup_cosine_decay_schedule - explicit request, see conversation
        history: offline keeps cosine, online switches to linear.
        Constant-mode actors (lr_schedule="constant") have no decay_steps
        to resize, so this only rebuilds anything for cosine_warmup mode.
        XQCFDLearner's own actor never reaches this branch at all (see this
        method's own early return below).

        Only touches Models that actually own an optimizer -
        target_critic/critic_target_vision/critic_target_obs_encoder never
        have a tx at all (same as xqc's own target_critic - Polyak-updated
        only, never gradient-updated directly), so there's nothing to
        reset there.

        XQCFDLearner (explicit request - see conversation history): this
        whole method is a NO-OP for this agent, regardless of online_steps -
        empirically verified (see conversation history) that xqc_learner.py's
        own reset_optimizers() - `self.actor = self.actor.replace(tx=
        actor.tx)` - does NOT reset opt_state at all (flax.struct.dataclass.
        replace only touches the field named, `tx` here; opt_state,
        including Adam's own momentum AND the schedule's own internal step
        count, carries over completely untouched), and rebuilds a "fresh"
        tx that's formula-IDENTICAL to the one already in use (init_models()
        is a deterministic function of the same config) - so calling it
        changes nothing observable. Reproducing that exactly means doing
        NOTHING here for this agent: the SAME num_interactions-spanning
        linear schedule built once in __init__ (see its own docstring)
        keeps counting straight through the offline->online transition,
        already partway decayed by whatever actor_pre_training_steps/
        critic_pre_training_steps consumed, exactly like the reference.

        See self.actor_optimizer_resets_online's own construction-site
        comment (__init__) for why this early-return condition is exposed
        as a plain attribute too, not just inlined here - a caller logging
        lr_schedule_fn's current value from outside this class needs to
        know, without duplicating this exact check, whether to query it at
        update_count (meaningful only once opt_state has actually been
        reset to 0, i.e. this method didn't early-return for this agent)
        or at agent.network.step (the real, never-reset counter - the
        right one for XQCFDLearner, and for the critic's own schedule too,
        see this method's own critic-section comment below: the critic is
        NEVER reset, for ANY agent, not just XQCFDLearner)."""
        if not self.actor_optimizer_resets_online:
            return
        if online_steps is not None and self._lr_schedule_mode == "cosine_warmup":
            transition_steps = max(online_steps, 1)
            self.lr_schedule_fn = optax.linear_schedule(
                init_value=self._actor_peak_lr, end_value=0.0, transition_steps=transition_steps,
            )
            actor_tx = self._maybe_delay_tx(optax.adamw(self.lr_schedule_fn, b1=self._actor_beta1, b2=self._actor_beta2, weight_decay=self._actor_weight_decay))
            encoder_tx = self._maybe_delay_tx(optax.adamw(self.lr_schedule_fn, b1=self._actor_beta1, b2=self._actor_beta2, weight_decay=self._actor_weight_decay))
            self.network = self.network.replace(tx=actor_tx, opt_state=actor_tx.init(self.network.params))
            self.vision = {cam: m.replace(tx=encoder_tx, opt_state=encoder_tx.init(m.params)) for cam, m in self.vision.items()}
            self.obs_encoder = self.obs_encoder.replace(tx=encoder_tx, opt_state=encoder_tx.init(self.obs_encoder.params))
        else:
            self.network = self.network.replace(opt_state=self.network.tx.init(self.network.params))
            self.vision = {cam: m.replace(opt_state=m.tx.init(m.params)) for cam, m in self.vision.items()}
            self.obs_encoder = self.obs_encoder.replace(opt_state=self.obs_encoder.tx.init(self.obs_encoder.params))
        # Critic: NO-OP, always, for every agent (explicit request - see
        # conversation history and this method's own XQCFDLearner-branch
        # docstring above for the full empirical rationale) - was
        # previously reset (online_steps=None: opt_state re-init; else:
        # tx+opt_state rebuilt scoped to online_steps) same as the actor
        # still is just above. Now matches xqc_learner.py's own
        # reset_optimizers exactly for the critic side too: its own
        # critic_lr_schedule_fn (built once in __init__, spanning
        # num_interactions*updates_per_step) just keeps counting straight
        # through the offline->online transition, already partway decayed
        # by whatever critic_pre_training_steps consumed - see __init__'s
        # own critic_lr_schedule_fn construction. The actor above is
        # UNCHANGED (still gets a genuine reset+fresh online-scoped
        # schedule) - explicit request: only the critic's own optimizer
        # behavior was asked to match the reference, not the actor's.

    def compute_val_loss(self, max_batches: Optional[int] = None) -> Optional[float]:
        """Mean loss over the held-out episodes' windows (self.val_dataset -
        a separate Dataset instance, see xqc/replay_buffer/
        chunked_qc_buffer.py's load_robomimic_flat/Dataset.get_val_batch),
        using the LIVE (non-EMA) params, no gradient step - matches the
        reference's own per-epoch val_loss metric exactly (see
        actor.val_loss's own docstring). Returns None if there are no
        held-out windows (val_ratio disabled). max_batches caps how many
        val batches get averaged in (None = all of them, matching the
        reference's own max_val_steps=None default) - the held-out set is
        normally small (~2% of the training set) so this is cheap either
        way."""
        val_batch = self.val_dataset.get_val_batch(
            To=self.To, h=self.execution_horizon, gamma=1.0, camera_keys=self.camera_keys, Ta=self.Ta,
        )
        val_size = val_batch.observations.shape[0]
        if val_size == 0:
            return None
        # Each batch's loss is weighted by how many windows count toward it:
        # with mask_reference_region the loss is a masked mean
        # (loss_masking.masked_mean), and a batch whose windows are all
        # outside the reference region (e.g. the last, partial batch of a
        # held-out episode's tail) has an all-zero mask - its loss is 0/0 =
        # NaN, which used to make every val_loss NaN. Such batches are
        # skipped; the weighted average is then exactly the masked mean over
        # the whole held-out set (a plain mean of batch means would also
        # over-weight a small last batch).
        losses, weights = [], []
        n_batches = -(-val_size // self.batch_size)  # ceil div
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)
        vision_params = {cam: m.params for cam, m in self.vision.items()}
        for i in range(n_batches):
            start, stop = i * self.batch_size, min((i + 1) * self.batch_size, val_size)
            weight = (
                float(np.sum(val_batch.in_reference_region[start:stop])) if self.mask_reference_region
                else float(stop - start)
            )
            if weight == 0:
                continue
            batch = ChunkedBatch(
                observations=jnp.asarray(val_batch.observations[start:stop]),
                actions=jnp.asarray(val_batch.actions[start:stop]),
                next_observations=jnp.asarray(val_batch.next_observations[start:stop]),
                next_actions=jnp.asarray(val_batch.next_actions[start:stop]),
                rewards=jnp.asarray(val_batch.rewards[start:stop]),
                masks=jnp.asarray(val_batch.masks[start:stop]),
                discount=jnp.asarray(val_batch.discount[start:stop]),
                valid=jnp.asarray(val_batch.valid[start:stop]),
                in_reference_region=jnp.asarray(val_batch.in_reference_region[start:stop]),
                # rescale_images: see update()'s own actor_batch construction
                # comment above - val_batch.images/next_images are also raw
                # uint8 from self.val_dataset's own storage.
                images=rescale_images({key: arr[start:stop] for key, arr in val_batch.images.items()}),
                next_images=rescale_images({key: arr[start:stop] for key, arr in val_batch.next_images.items()}),
            )
            self.rng, val_rng = jax.random.split(self.rng)
            loss = self.val_loss_fn(
                self.network, self.vision, self.obs_encoder,
                self.network.params, vision_params, self.obs_encoder.params,
                batch, rng=val_rng,
            )
            losses.append(float(loss))
            weights.append(weight)
        if not losses:
            return None
        return float(np.average(losses, weights=weights))

    def evaluate_critic(self, batch: ChunkedBatch) -> InfoDict:
        """No-gradient pass over a batch, using the critic's own
        chunked-SARSA target (batch.next_actions, not an actor-bootstrap
        target - there's no actor_network_model kwarg passed here, matching
        chunked_learner_sarsa.py's own evaluate() exactly, which has no
        actor at all). Same action normalization/dropout-key handling as
        that method - see its own docstring. Requires critic_eval_fn to
        already exist (built in _build_critic_bound_fns, only when
        critic_name is set)."""
        batch = batch._replace(
            actions=(batch.actions - self.action_offset) / self.action_scale,
            next_actions=(batch.next_actions - self.action_offset) / self.action_scale,
        )
        self.rng, crop_key, dropout_key = jax.random.split(self.rng, 3)
        dropout_kwarg = {"dropout_key": dropout_key} if self.critic_name == "transformer" else {}
        return self.critic_eval_fn(
            self.critic, self.target_critic, self.critic_vision, self.critic_target_vision,
            self.critic_obs_encoder, self.critic_target_obs_encoder, batch,
            crop_key=crop_key,
            **dropout_kwarg,
        )

    def compute_critic_val_loss(self, discount: float) -> Optional[InfoDict]:
        """Critic-side counterpart of compute_val_loss above - the full
        held-out set (self.val_dataset) run through evaluate_critic in one
        no-gradient pass, matching train_critic_sarsa.py's own
        `learner.evaluate(val_dataset.get_val_batch(...))` exactly (same
        Dataset.get_val_batch call, same critic_eval_fn underneath - see
        conversation history: this is what makes the two pipelines'
        train/val critic curves directly comparable). discount is passed
        in by the caller (train_parallel_chunking.py's own heuristic
        discount, computed after this agent is constructed - see its own
        get_discount call) rather than stored on self, since nothing else
        on this class currently needs it. Returns None if there's no
        critic, or no held-out windows (val_ratio disabled)."""
        if self.critic is None:
            return None
        val_batch = self.val_dataset.get_val_batch(
            To=self.To, h=self.execution_horizon, gamma=discount, camera_keys=self.critic_camera_keys,
        )
        if val_batch.observations.shape[0] == 0:
            return None
        return self.evaluate_critic(val_batch)

    def _inference_params(self, use_ema: bool, ema_rate: Optional[float] = None):
        """Returns (network_params, network_batch_stats, vision_params,
        obs_encoder_params). network_batch_stats (explicit request - see
        conversation history and self.network_ema_batch_stats's own
        construction-site comment): paired with whichever network_params
        this call returns, so a caller combining EMA params with EMA
        batch_stats (or live with live) never mixes two different-speed
        summaries of the network's history - see every caller of this
        method, all updated to thread network_batch_stats through instead
        of reaching for self.network.batch_stats directly.

        ema_rate (explicit request - see conversation history and
        self.ema_rate_online's own construction-site comment): None
        (default) gates off self.ema_rate, unchanged. A caller may pass a
        different rate here (e.g. sample_actions's own online-rollout call
        site passing self.ema_rate_online) to gate THIS call only, without
        touching self.ema_rate or the EMA buffer's own maintenance."""
        effective_ema_rate = self.ema_rate if ema_rate is None else ema_rate
        if use_ema and effective_ema_rate < 1:
            return self.network_ema_params, self.network_ema_batch_stats, self.vision_ema_params, self.obs_encoder_ema_params
        return (
            self.network.params, self.network.batch_stats,
            {cam: m.params for cam, m in self.vision.items()}, self.obs_encoder.params,
        )

    def compute_action_mse(self, obs_window: np.ndarray, action_window: np.ndarray, images: Dict[str, np.ndarray]) -> float:
        """MSE between a fully-sampled (generative, not 1-step-noise-
        prediction) action chunk and the ground-truth action_window, both in
        real/unnormalized units - matches the reference's own
        train_action_mse_error exactly (train_diffusion_unet_lowdim_workspace.py's
        run(): `policy.predict_action(obs_dict)` on a FIXED training batch
        captured once at the start of training, EMA params if use_ema,
        compared against that same batch's raw `action` via plain MSE -
        train_parallel.py's own training loop captures and reuses one fixed
        batch the same way, passed in here as obs_window/action_window/
        images each call). Uses EMA params when self.ema_rate<1, matching
        the reference's own `policy = self.ema_model if use_ema`."""
        # rescale_images: `images` here is drawn straight from self.dataset
        # (see train_parallel_chunking.py's own _train_sampling_batch) - raw
        # uint8, same fix as update()'s own actor_batch construction above.
        images_jnp = rescale_images(images)
        act_shape = action_window.shape
        network_params, network_batch_stats, vision_params, obs_encoder_params = self._inference_params(use_ema=True)
        self.rng, infer_rng = jax.random.split(self.rng)
        action_pred = self.infer_fn(
            self.network, self.vision, self.obs_encoder,
            network_params, vision_params, obs_encoder_params,
            jnp.asarray(obs_window), images_jnp,
            act_shape=act_shape, rng=infer_rng,
            network_batch_stats=network_batch_stats,
        )
        return float(jnp.mean((jnp.asarray(action_pred) - jnp.asarray(action_window)) ** 2))

    def compute_action_range(
        self, obs_window: np.ndarray, images: Dict[str, np.ndarray], ema_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Plain output-range diagnostic (mean/max/min, NORMALIZED/network-
        output units - explicit request, see conversation history: the
        point is to see the network's own raw output scale directly, to
        catch a scaling issue in the network itself, which real-unit
        rescaling would mask - self.infer_fn's own action*action_scale+
        action_offset rescale (see its own docstring) is inverted right
        back out below rather than avoiding infer_fn altogether, since
        infer_fn/EMA-params is still the right, cheap, already-proven
        diagnostic call (matches compute_action_mse just above) - only the
        UNITS of what gets reported differ) on a FIXED offline training
        batch - the offline-phase counterpart of update_actor's own
        offline_action_{mean,max,min} (xqc/agents/chunking/actor.py, same
        normalized-units convention), sharing the SAME bare wandb key
        names via train_parallel_chunking.py's own epoch_log block (same
        "MSE"/offline_prior_mse bare-key-merge convention already used
        there) so the curve reads as one continuous trace: the action
        range imitation learning established while offline, then how far
        online RL moves it once training continues. No offline/online
        split needed here (unlike update_actor's own version) - every
        sample IS offline by construction during BC pretraining, there is
        no online buffer yet. Same EMA-params/infer_fn convention as
        compute_action_mse just above (this is a genuinely separate,
        once-per-epoch diagnostic call - no existing differentiable
        forward pass to reuse the way update_actor's own online-phase
        version does, since update_actor_bc's own loss() operates on a
        noised version of the recorded action, never a full sample).

        ema_rate (explicit request - see conversation history): forwarded
        straight to _inference_params - None (default) reproduces today's
        behavior exactly (self.ema_rate). Pass 1.0 to force LIVE weights
        instead, for a direct, same-batch comparison against the EMA'd
        call above - e.g. to check whether the live network's own raw
        output is saturating/collapsing in a way the EMA'd one isn't,
        rather than inferring that indirectly from eval success rate."""
        images_jnp = rescale_images(images)
        act_shape = (obs_window.shape[0], self.Ta, self.act_dim)
        network_params, network_batch_stats, vision_params, obs_encoder_params = self._inference_params(
            use_ema=True, ema_rate=ema_rate,
        )
        self.rng, infer_rng = jax.random.split(self.rng)
        action_pred_real = jnp.asarray(self.infer_fn(
            self.network, self.vision, self.obs_encoder,
            network_params, vision_params, obs_encoder_params,
            jnp.asarray(obs_window), images_jnp,
            act_shape=act_shape, rng=infer_rng,
            network_batch_stats=network_batch_stats,
        ))
        action_pred = (action_pred_real - self.action_offset) / self.action_scale
        return {
            "mean": float(action_pred.mean()),
            "max": float(action_pred.max()),
            "min": float(action_pred.min()),
        }

    @property
    def supports_policy_loglik(self) -> bool:
        """True when the actor network has a flow_map (MeanFlow: actor.py's
        policy_loglik, via its Jacobian) or is an explicit distribution
        (XQCFD: actor.py's distribution_policy_loglik, exact)."""
        return hasattr(self.network.apply_fn, "flow_map") or hasattr(self.network.apply_fn, "distribution")

    def compute_policy_loglik(
        self, obs_window: np.ndarray, images: Dict[str, np.ndarray], rng: jax.Array,
        num_spread_samples: int = 8, jac_chunk: int = 16, use_prior: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Per-state log-likelihood/spread diagnostic - see actor.py's
        policy_loglik (flow_map networks) / distribution_policy_loglik
        (explicit-distribution networks, jac_chunk unused). Uses exactly the
        params the ONLINE rollout samples with (ema_rate=self.ema_rate_online,
        as train_parallel_chunking.py's own rollout call), so it measures the
        policy actually being run. rng is passed in (not split off self.rng)
        so enabling this diagnostic leaves the training RNG stream untouched.

        use_prior (explicit request - see conversation history): False
        (default) is the behavior above, unchanged. True swaps in
        self.prior_network/prior_vision/prior_obs_encoder's own params/
        batch_stats instead - the frozen BC-pretrained snapshot taken once
        at the offline->online transition (train_parallel_chunking.py's
        own first_online_transition block), never gradient-updated after
        that (tx=None - see this class's own __init__). Same exact
        diagnostic computation either way, just evaluated at a different,
        fixed set of params - this is the policy wasserstein_regularization/
        kl_regularization's own teacher snapshot is pulled toward (or
        measured against), and the one we'd hope stays stationary (its own
        p(own sample|state) shouldn't drift, since nothing ever updates it)
        across online rollout - any drift here would point at the
        snapshot/batch_stats plumbing, not the policy itself, since a
        truly frozen set of params has no mechanism to change its own
        likelihood of its own samples over time."""
        if use_prior:
            network_params = self.prior_network.params
            network_batch_stats = self.prior_network.batch_stats
            vision_params = {cam: m.params for cam, m in self.prior_vision.items()}
            obs_encoder_params = self.prior_obs_encoder.params
        else:
            network_params, network_batch_stats, vision_params, obs_encoder_params = self._inference_params(
                use_ema=True, ema_rate=self.ema_rate_online,
            )
        if not hasattr(self.network.apply_fn, "flow_map"):
            out = distribution_policy_loglik(
                self.network, self.vision, self.obs_encoder,
                network_params, vision_params, obs_encoder_params,
                jnp.asarray(obs_window), rescale_images(images), self.obs_offset, self.obs_scale, rng,
                camera_keys=self.camera_keys, random_crop=self.random_crop, crop_shape=self.crop_shape,
                num_spread_samples=num_spread_samples, network_batch_stats=network_batch_stats,
            )
            return {k: np.asarray(v) for k, v in out.items()}
        out = policy_loglik(
            self.network, self.vision, self.obs_encoder,
            network_params, vision_params, obs_encoder_params,
            jnp.asarray(obs_window), rescale_images(images), self.obs_offset, self.obs_scale, rng,
            camera_keys=self.camera_keys, random_crop=self.random_crop, crop_shape=self.crop_shape,
            act_shape=(self.Ta, self.act_dim), num_spread_samples=num_spread_samples, jac_chunk=jac_chunk,
            network_batch_stats=network_batch_stats,
        )
        return {k: np.asarray(v) for k, v in out.items()}

    def reset_policy(self) -> None:
        """Full reset of chunk-execution state. Call once at the start of a
        rollout (ParallelEnv.evaluate() calls this right after env.reset())."""
        self._executor.reset()

    def reset_policy_idx(self, done: np.ndarray) -> None:
        """Per-env reset for envs whose episode just ended mid-rollout - see
        ChunkExecutor.reset_idx's own docstring for the full explanation
        (including the eval_episodes>1 bug this replaced)."""
        self._executor.reset_idx(done)

    def sample_actions(
        self,
        observations: np.ndarray,
        temperature: float = 1.0,
        images: Optional[Dict[str, np.ndarray]] = None,
        use_ema: bool = True,
        ema_rate: Optional[float] = None,
        best_of_n: int = 1,
    ) -> np.ndarray:
        """Denoising/flow-map solve, executed as a receding-horizon action
        chunk (bookkeeping in ChunkExecutor; this method's own job is only
        to supply the actual network inference as a callback).

        ema_rate: forwarded straight to _inference_params - see its own
        ema_rate param docstring and self.ema_rate_online's construction-
        site comment. None (default) changes nothing for any existing
        caller.

        temperature (explicit request - see conversation history, and
        xqc/agents/chunking/actor.py's own infer() docstring for the full
        rationale): now genuinely threaded through to infer_fn - matches
        xqc/agents/xqc/xqc_learner.py's own sample_actions_with_log_probs(
        temperature), whose default is ALSO 1.0 (used, unoverridden, by
        train_parallel.py's own online rollout loop) with xqc/envs/envs.py's
        own eval_env.evaluate() explicitly overriding to 0.0 at eval time -
        this default was previously 0.0 here, silently WRONG for that same
        online-rollout call site (train_parallel_chunking.py's own
        `agent.sample_actions(observations)`, no override) even though it
        had zero effect before this parameter was wired to anything. Only
        XQCFD's own network actually uses this value (network_accepts_
        temperature, bound below) - every other actor in this family has a
        deterministic .sample() with no temperature concept, so this stays
        a complete no-op for DP/FM/MF/MIP/NF/BC/AF exactly as before.

        best_of_n (explicit request - see conversation history: the
        rollout-side counterpart of ChunkedPolicyLearner's own
        critic_bootstrap_best_of_n/_update's actor-bootstrap ranking - same
        "optimal policy", no actor gradient update involved, just sample N
        candidate chunks from the actor and keep the one the LIVE critic
        (self.critic/self.critic_vision/self.critic_obs_encoder, NOT the
        target critic - rollout should always act on the most up-to-date Q
        estimate, unlike the TD target which deliberately uses the frozen
        target) scores highest. 1 (default) is a true no-op - score_fn is
        never built, infer_fn takes the exact single-.sample() path it
        always took. >1 needs self.critic to exist (raises otherwise - no
        sane ranking without one). Scoring reuses self.compute_q_fn
        (already bound by _build_critic_bound_fns, the SAME function
        update_actor's own Q-maximization term calls) at the SAME obs
        window/images this call's own actor encoding already uses - safe
        since this codebase has exactly one shared self.To for both actor
        and critic (no separate critic-side window length - see
        self.critic_camera_keys's own construction-site comment), scored
        on the execution_horizon-sliced prefix of each candidate (matching
        what the critic was actually trained on - see ChunkedBatch's own
        actions field), min-over-ensemble (TD3-style pessimism, same
        convention bootstrap_next_actions's own score_fn callers use)."""

        def infer_chunk_fn(obs_history: np.ndarray, image_history: Dict[str, np.ndarray]) -> np.ndarray:
            images_jnp = {key: jnp.asarray(arr) for key, arr in image_history.items()}
            act_shape = (obs_history.shape[0], self.Ta, self.act_dim)
            network_params, network_batch_stats, vision_params, obs_encoder_params = self._inference_params(use_ema=use_ema, ema_rate=ema_rate)
            self.rng, infer_rng = jax.random.split(self.rng)
            score_fn = None
            if best_of_n > 1:
                if self.critic is None:
                    raise ValueError("sample_actions: best_of_n>1 needs a critic to rank candidates against.")
                # Tiled by best_of_n to match infer()'s own flat
                # (B*best_of_n, ...) candidate batch (explicit fix - see
                # infer()'s own docstring: no jax.vmap any more) - obs_window/
                # images must have the SAME leading batch size as the
                # candidate action this gets paired with below.
                obs_history_tiled = jnp.repeat(jnp.asarray(obs_history), best_of_n, axis=0)
                images_tiled = {k: jnp.repeat(v, best_of_n, axis=0) for k, v in images_jnp.items()}

                def score_fn(flat_candidates):
                    candidate_slice = flat_candidates[:, :self.execution_horizon]
                    q = self.compute_q_fn(
                        self.critic, self.critic_vision, self.critic_obs_encoder,
                        obs_window=obs_history_tiled, images=images_tiled, action=candidate_slice,
                    )
                    return jnp.min(q, axis=0)

            return self.infer_fn(
                self.network, self.vision, self.obs_encoder,
                network_params, vision_params, obs_encoder_params,
                jnp.asarray(obs_history), images_jnp,
                act_shape=act_shape, rng=infer_rng, temperature=temperature,
                network_batch_stats=network_batch_stats,
                best_of_n=best_of_n, score_fn=score_fn,
            )

        return self._executor.sample_actions(observations, images, infer_chunk_fn)

    def sample_actions_with_log_probs(self, observations: np.ndarray, temperature: float = 1.0):
        raise NotImplementedError(self._rl_not_implemented_msg())
