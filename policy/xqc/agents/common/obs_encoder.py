"""The observation-encoding pipeline shared by every ChunkedPolicyLearner-
based agent (xqc/agents/chunking/chunked_learner.py) - everything that turns
raw observations+images into what the prediction network actually
consumes, in the order it actually runs:

  1. encode_condition: raw obs_window + raw per-camera images -> a single
     "condition" vector (low-dim obs concatenated with per-camera vision
     features, run through each camera's own vision encoder). Deliberately
     kept here rather than in xqc/agents/chunking/actor.py (see conversation
     history): it's not action-prediction logic, it's condition
     construction - a future critic needs the exact same function, not a
     copy, so it lives in the neutral module both depend on rather than in
     either's own file.
  2. normalization (obs_offset/obs_scale - done inline by whichever caller
     needs it, e.g. actor.py's update_actor_bc/val_loss/infer).
  3. IdentityObsEncoder (below): the final stage feeding ChiUNet's
     `condition` input.

IdentityObsEncoder is a no-op - matches the reference Diffusion Policy
implementation this whole port was verified against, which has NO learned
compression stage here at all (raw, uncompressed obs-encoder output feeds
FiLM conditioning directly at every ChiUNet residual block - see
xqc/agents/common/network.py's ChiUNet.project_condition field docstring).
Deliberately applied to every agent for a fair, apples-to-apples comparison
across the different training objectives/algorithms this repo implements,
all sharing one verified prediction network.

This module used to also carry StateMLPEncoder/VisionObsEncoder (ports of
MIP's own original obs_type="state"/"image" encoders, xqc/networks/mip/
network_utils.py:get_encoder) for architectural parity with MIP's own
upstream reference - removed as dead code once every agent moved onto
IdentityObsEncoder above and nothing constructed them anymore (see git
history if MIP's own original encoder choice is ever needed again).
"""
from typing import Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from xqc.agents.common.vision import build_vision_encoder, crop_images as _crop_images
from xqc.networks.common import Model


def encode_condition(
    vision_apply_fn, vision_params, vision_batch_stats, obs_window, camera_keys, images, train,
    random_crop=False, crop_shape=(76, 76), crop_key=None,
):
    """obs_window: (B, To, obs_dim). images: {key: (B, To, H, W, 3) float32
    in [-1,1]}. Returns (condition, new_batch_stats): condition is (B, To,
    obs_dim + len(camera_keys)*feature_dim) = low-dim obs concatenated with
    per-camera vision features. vision_params/vision_batch_stats are dicts
    keyed by camera - one independently-initialized encoder per camera
    (matches robomimic's ObservationEncoder, which never shares weights
    between obs keys by default). random_crop/crop_shape/crop_key: optional
    crop augmentation, each camera independently split - see
    xqc/agents/common/vision.py's crop_images docstring. ReLU is applied to
    each camera's raw vision features right after the encoder, matching
    robomimic's ObservationEncoder.forward feature_activation default.

    Only ever receives plain apply-function/params/batch_stats arguments,
    never assumes anything about how the caller stores or builds them - see
    conversation history for why this matters for future critic reuse."""
    if not camera_keys:
        return obs_window, vision_batch_stats
    batch_size, To = obs_window.shape[:2]
    feats = [obs_window] # low dim obs. 
    batch_stats = dict(vision_batch_stats)
    if random_crop and train:
        camera_crop_keys = jax.random.split(crop_key, len(camera_keys))
    else:
        camera_crop_keys = [None] * len(camera_keys)
    for key, cam_crop_key in zip(camera_keys, camera_crop_keys):
        imgs = images[key]
        if random_crop:
            imgs = _crop_images(imgs, crop_shape, train, cam_crop_key)
        imgs = imgs.reshape((batch_size * To,) + imgs.shape[2:])
        variables = {"params": vision_params[key], "batch_stats": batch_stats[key]}
        if train:
            feat, new_model_state = vision_apply_fn(variables, imgs, train=True, mutable=["batch_stats"])
            batch_stats[key] = new_model_state["batch_stats"]
        else:
            feat = vision_apply_fn(variables, imgs, train=False)
        feat = jax.nn.relu(feat) # images encoding. 
        feats.append(feat.reshape(batch_size, To, -1))
    return jnp.concatenate(feats, axis=-1), batch_stats


def encode_observation_from_models(
    vision_models, vision_params, obs_encoder_model, obs_encoder_params,
    obs_window, images, obs_offset, obs_scale, camera_keys, train,
    random_crop=False, crop_shape=(76, 76), crop_key=None,
):
    """encode_condition -> normalize -> obs_encoder - the ONE shared
    implementation of "raw obs+images -> what the network actually sees",
    used by everything: the actor (xqc/agents/chunking/actor.py's own
    update_actor_bc/update_actor/val_loss/infer) and every critic variant
    (xqc/agents/critics/{mlp,xqc_style,transformer}/critic.py) - see
    conversation history: this used to be inlined separately four times in
    actor.py (in two different shapes - one taking Model objects, one
    taking pre-extracted apply_fns) plus duplicated identically three more
    times across critic.py's own variants (each explicitly documented as
    "not an import, deliberately self-contained"). Collapsing all of that
    into one function/one calling convention is the point - living in this
    neutral module (same reasoning as encode_condition above) means
    actor.py and critic.py both depend on this instead of on each other.

    Takes `vision_models`/`obs_encoder_model` as actual Model objects (for
    their `.apply_fn`/`.batch_stats`) but `vision_params`/
    `obs_encoder_params` as SEPARATE arguments rather than reading
    `vision_models[cam].params` directly - this is what makes one calling
    convention serve every caller: gradient computation (update_actor_bc/
    update_actor/every critic loss function) needs the params to be an
    externally-tracked argnum, not read off a Model attribute; inference
    (val_loss/infer, called with either live or EMA params) just uses this
    same slot to pass whichever params it wants evaluated - the Model
    objects themselves (architecture/apply_fn/batch_stats) never differ
    between live and EMA. `images` must already be in the encoder's
    expected float32 [-1,1] range - the actor's own dataset rescales at
    load time; critics reading raw lazy uint8 batches must call
    xqc/agents/critics/vision_utils.py's rescale_images first (a
    critic-specific lazy-loading detail, not a generic encoding concern,
    so it isn't baked in here)."""
    vision_apply_fn = next(iter(vision_models.values())).apply_fn.apply if vision_models else None
    vision_batch_stats = {cam: (m.batch_stats or {}) for cam, m in vision_models.items()}
    condition, new_vision_batch_stats = encode_condition(
        vision_apply_fn, vision_params, vision_batch_stats, obs_window, camera_keys, images, train=train,
        random_crop=random_crop, crop_shape=crop_shape, crop_key=crop_key,
    )
    obs_norm = (condition - obs_offset) / obs_scale
    obs = obs_encoder_model.apply_fn.apply({"params": obs_encoder_params}, obs_norm)
    return obs, new_vision_batch_stats


def build_vision_and_obs_encoder(
    camera_keys: Sequence[str],
    raw_image_shape: Dict[str, Tuple[int, int]],
    vision_encoder: str,
    vision_norm_type: str,
    spatial_softmax_num_kp: int,
    spatial_softmax_feature_dim: int,
    vit_image_size: int,
    random_crop: bool,
    crop_shape: Tuple[int, int],
    condition_dim: int,
    To: int,
    vision_key: jnp.ndarray,
    obs_encoder_key: jnp.ndarray,
    tx: Optional[optax.GradientTransformation],
) -> Tuple[Dict[str, Model], Model]:
    """Builds the per-camera vision encoders + the (currently always
    identity) obs_encoder feeding a prediction network's `condition` input -
    an exact, behavior-preserving relocation of the block that used to live
    inline in xqc/agents/chunking/chunked_learner.py's own __init__ (see
    conversation history: this is a pure move, no logic changes, so it
    needs no re-verification against the DP reference the way a genuine
    rewrite would).

    Called once there for the actor's own self.vision/self.obs_encoder, and
    again (different camera_keys/keys/tx) for a critic's own SEPARATE
    vision stack - "separate vision for actor and critic" is the
    deliberately simple starting point (see conversation history); shared
    or frozen-actor-vision modes are a later generalization, not
    implemented here. `condition_dim` = low_dim_obs_dim + vision_dim - the
    caller's job to compute (needs vision_feature_dims[vision_encoder],
    which this function doesn't take, to stay agnostic to how vision_dim
    was derived).

    `tx` is the SAME optax.GradientTransformation object reused for every
    Model built here (every camera's own vision encoder, plus obs_encoder) -
    not a fresh instance per Model. This is safe: a GradientTransformation
    is stateless config (just (init_fn, update_fn) closures); the actual
    per-Model optimizer STATE lives entirely in each Model's own separately-
    built opt_state (tx.init(params), called once per Model.create), so
    reusing one tx object is numerically equivalent to building a fresh one
    per Model - see conversation history. Pass `tx=None` (e.g. for a
    target/EMA-only copy that never trains) to skip optimizer state
    entirely, same as Model.create's own tx=None default."""
    vision: Dict[str, Model] = {}
    if camera_keys:
        # One independently-initialized encoder instance per camera, not
        # one shared across cameras - matches robomimic's
        # ObservationEncoder (register_obs_key with share_net_from=None).
        vision_def = build_vision_encoder(
            vision_encoder, norm_type=vision_norm_type,
            spatial_softmax_num_kp=spatial_softmax_num_kp,
            spatial_softmax_feature_dim=spatial_softmax_feature_dim,
            vit_image_size=vit_image_size,
        )
        camera_vision_keys = jax.random.split(vision_key, len(camera_keys))
        for cam_key, cam_vision_key in zip(camera_keys, camera_vision_keys):
            raw_h, raw_w = raw_image_shape[cam_key]
            if random_crop:
                assert crop_shape[0] <= raw_h and crop_shape[1] <= raw_w, (
                    f"crop_shape {crop_shape} does not fit within camera "
                    f"{cam_key!r}'s raw image shape {(raw_h, raw_w)}"
                )
                dummy_img_shape = crop_shape
            else:
                dummy_img_shape = (raw_h, raw_w)
            dummy_img = jnp.zeros((1,) + dummy_img_shape + (3,))
            vision[cam_key] = Model.create(vision_def, [cam_vision_key, dummy_img, False], tx=tx)

    # No learned projection here - matches the reference exactly (the raw,
    # uncompressed obs-encoder output is what feeds ChiUNet - see
    # xqc/agents/common/network.py's project_condition field docstring).
    obs_encoder_def = IdentityObsEncoder()
    dummy_condition = jnp.zeros((1, To, condition_dim))
    obs_encoder = Model.create(obs_encoder_def, [obs_encoder_key, dummy_condition], tx=tx)

    return vision, obs_encoder


class IdentityObsEncoder(nn.Module):
    """No-op - see module docstring. Zero parameters, so it costs nothing in
    the shared params["obs_encoder"] tree structure (callers should use
    `.get("params", {})` rather than `["params"]` when calling `.init()` on
    this, since a parameterless module produces no "params" collection at
    all)."""

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        return obs
