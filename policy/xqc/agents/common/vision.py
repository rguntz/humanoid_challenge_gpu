"""Plain (non-pretrained) ResNet18 image encoder, matching the original MIP
reference exactly: xqc/networks/mip/encoders.py:get_resnet(name, weights=None)
-> torchvision.models.resnet18(weights=None) with resnet.fc replaced by
Identity. Random init, standard BatchNorm, classifier stripped - no external
checkpoint, no h5 loading. Simpler than an earlier pretrained-ResNet18
attempt (which needed a frozen/trainable BatchNorm split to preserve
checkpoint statistics) - for a first, fair comparison against MIP's own
default config, which does not use pretrained weights either.

BatchNorm's running stats (batch_stats) are a real Flax mutable collection,
updated normally during training (mutable=["batch_stats"]) - matching how
MIP's own torchvision resnet trains from scratch (no reason to freeze
statistics that were never pretrained in the first place).

This class itself is just the per-camera architecture definition - the
per-camera instantiation lives in the caller. ChunkedPolicyLearner.__init__
(xqc/agents/chunking/chunked_learner.py) builds one independently-initialized
Model per camera key (a fresh PRNG key and a fresh optimizer per camera, via
build_vision_encoder + Model.create in a loop over camera_keys), and
update_actor_bc (xqc/agents/chunking/actor.py) computes and applies a
separate gradient per camera - i.e. no weights are shared between
agentview_image and robot0_eye_in_hand_image, matching robomimic's own
ObservationEncoder (register_obs_key with share_net_from=None, the default),
which the reference Diffusion Policy implementation reuses verbatim.

Per-encoder pooled-feature width (needed by callers, e.g.
ChunkedPolicyLearner, to size raw_condition_dim/obs_offset/obs_scale before
any image has actually been encoded), the SpatialSoftmax keypoint count/
feature width, and the ViT checkpoint's required input resolution are all
config values now (conf/agent/training/vision.yaml's vision_feature_dims/
spatial_softmax_num_kp/spatial_softmax_feature_dim/vit_image_size), not
module constants here - see build_vision_encoder below for where they're
threaded through. "resnet"'s own feature width is num_kp(32, default)*2
spatial-softmax keypoint coords, projected through a final Linear(64,
default) - see SpatialSoftmaxPool/PlainResNet18Encoder below, ported from
robomimic's VisualCore(pool_class="SpatialSoftmax", pool_kwargs.num_kp=32,
feature_dimension=64) default BC-RNN image config, which the reference
Diffusion Policy implementation reuses verbatim
(get_robomimic_config('bc_rnn', 'image', ...) in
diffusion_policy/common/robomimic_config_util.py). Was a plain 512-dim
global average pool (no spatial structure) - reverted (see git history)
once found to diverge from the reference during a training-vs-reference
debug pass; SpatialSoftmax's explicit 2D keypoint localization is a much
stronger inductive bias for precise manipulation than a global average.
"""
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class SpatialSoftmaxPool(nn.Module):
    """Port of robomimic.models.base_nets.SpatialSoftmax (num_kp=32,
    temperature=1.0 fixed/non-learnable, noise_std=0.0 - robomimic's own
    BC-RNN image defaults, see generate_paper_configs.py's
    modify_config_for_default_image_exp), followed by the Linear(num_kp*2,
    feature_dimension) VisualCore always appends when feature_dimension is
    set (both 64 here, but still a real learned layer, not a no-op reshape).

    x: (B, H, W, C_in) -> (B, feature_dim). A 1x1 conv projects C_in ->
    num_kp channels, each channel's H*W activations are softmax-normalized
    into a spatial probability distribution, and the expected (x, y)
    position under that distribution becomes one keypoint - i.e. "where is
    the most salient point for this learned feature", not just "how much of
    it is present" (what a global average pool would give)."""

    num_kp: int
    feature_dim: int
    temperature: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, _ = x.shape
        x = nn.Conv(self.num_kp, (1, 1))(x)  # (B, H, W, K)
        x = x.transpose(0, 3, 1, 2).reshape(b * self.num_kp, h * w)  # (B*K, H*W)
        attention = jax.nn.softmax(x / self.temperature, axis=-1)

        # [-1, 1] pixel-coordinate grid, row-major flattened to match the
        # (B*K, H*W) attention layout above - same convention as robomimic's
        # own np.meshgrid(linspace(-1,1,W), linspace(-1,1,H)).reshape(1,H*W).
        pos_x, pos_y = jnp.meshgrid(jnp.linspace(-1.0, 1.0, w), jnp.linspace(-1.0, 1.0, h))
        pos_x = pos_x.reshape(1, h * w)
        pos_y = pos_y.reshape(1, h * w)
        expected_x = jnp.sum(pos_x * attention, axis=1, keepdims=True)
        expected_y = jnp.sum(pos_y * attention, axis=1, keepdims=True)
        keypoints = jnp.concatenate([expected_x, expected_y], axis=1)  # (B*K, 2)
        keypoints = keypoints.reshape(b, self.num_kp * 2)  # (B, K*2), [x0,y0,x1,y1,...]

        return nn.Dense(self.feature_dim)(keypoints)


def _norm_layer(norm_type: str, train: bool, channels: int) -> nn.Module:
    """"batch" -> nn.BatchNorm (running stats, mutable "batch_stats"), "group"
    -> nn.GroupNorm (no running stats, no mutable state - normalizes each
    example independently using its own within-group statistics). GroupNorm
    is what the reference diffusion_policy implementation swaps to for its
    vision encoder specifically because EMA-ing BatchNorm's running stats
    separately from its affine params causes a train/eval mismatch (see
    xqc/agents/actors/dp/dp_learner.py's docstring) - GroupNorm has no running stats
    to mismatch in the first place.

    num_groups = channels // 16 (fixed GROUP SIZE of 16 channels, so the
    group COUNT varies per layer width) - matches the reference exactly
    (`nn.GroupNorm(num_groups=x.num_features//16, num_channels=x.num_features)`
    in diffusion_unet_hybrid_image_policy.py's replace_submodules call). Was
    a fixed num_groups=16 (fixed group COUNT, varying group SIZE) here,
    incorrectly - only coincidentally equivalent at exactly 256 channels
    (ResNet18's 3rd stage); wrong at every other stage (64/128/512 channels)
    - see git history."""
    if norm_type == "batch":
        return nn.BatchNorm(use_running_average=not train)
    if norm_type == "group":
        return nn.GroupNorm(num_groups=max(1, channels // 16))
    raise ValueError(f"Unknown norm_type {norm_type!r}, expected 'batch' or 'group'")


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3x3 convs + norm layer, with a
    projection shortcut when the stride or channel count changes."""
    features: int
    stride: int = 1
    norm_type: str = "batch"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = x

        out = nn.Conv(self.features, (3, 3), strides=(self.stride, self.stride), padding=[(1, 1), (1, 1)], use_bias=False)(x)
        out = _norm_layer(self.norm_type, train, self.features)(out)
        out = nn.relu(out)

        out = nn.Conv(self.features, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)], use_bias=False)(out)
        out = _norm_layer(self.norm_type, train, self.features)(out)

        if self.stride != 1 or residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), strides=(self.stride, self.stride), use_bias=False)(residual)
            residual = _norm_layer(self.norm_type, train, self.features)(residual)

        return nn.relu(out + residual)


class PlainResNet18Encoder(nn.Module):
    """Input: (B, H, W, 3) RGB in [-1, 1]. Output: (B, 64) pooled features.

    Standard resnet18 layout: stem (7x7 conv, stride 2, maxpool) -> 4 stages
    of 2 BasicBlocks each (64, 128, 256, 512 channels; first block of each
    stage after the first uses stride 2) -> SpatialSoftmaxPool (see above;
    matches robomimic's ResNet18Conv+SpatialSoftmax, which the reference
    Diffusion Policy implementation reuses verbatim). No classifier head
    (matches resnet.fc = Identity()).

    norm_type: "batch" (default, unchanged for every existing caller) or
    "group" - see _norm_layer's docstring. Opt-in per caller via
    build_vision_encoder(norm_type=...) below. spatial_softmax_num_kp/
    spatial_softmax_feature_dim: SpatialSoftmaxPool's own num_kp/feature_dim
    (conf/agent/training/vision.yaml) - this class's own output width is
    exactly spatial_softmax_feature_dim, so it must match whatever
    vision_feature_dims["resnet"] the caller separately configured.
    """
    spatial_softmax_num_kp: int
    spatial_softmax_feature_dim: int
    norm_type: str = "batch"
    _STAGES = [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)], use_bias=False)(x)
        x = _norm_layer(self.norm_type, train, 64)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

        for features, stride in self._STAGES:
            x = BasicBlock(features, stride, self.norm_type)(x, train)

        return SpatialSoftmaxPool(num_kp=self.spatial_softmax_num_kp, feature_dim=self.spatial_softmax_feature_dim)(x)


class PretrainedViTEncoder:
    """Wraps HuggingFace's `transformers.FlaxViTModel` (default checkpoint:
    google/vit-base-patch16-224) to duck-type the same init(rng, dummy_img,
    train)/apply(variables, imgs, train, mutable) interface PlainResNet18Encoder
    exposes (see vision.py's module docstring and mf_learner.py/mip_learner.py's
    _encode_condition), so it drops into the existing vision_def slot without
    touching the call sites.

    Downloads the checkpoint from the Hugging Face Hub on first use (cached
    under ~/.cache/huggingface afterwards - no local checkpoint file needed,
    and no authentication needed for this public model).

    Input: (B, H, W, 3) RGB float32 in [-1, 1] - same format the dataset
    provides for PlainResNet18Encoder (xqc/agents/common/dataset.py already
    rescales to [-1, 1], matching this checkpoint's own mean=std=0.5
    preprocessing convention exactly - see _VIT_MEAN/_VIT_STD above - so no
    further rescale happens here, only the resize to 224x224).

    Output: (B, 768) - the pretrained CLS-token embedding from the last hidden
    state, not the classification pooler output (that head is randomly
    initialized by transformers, never pretrained, so using it here would
    silently mix untrained weights into an otherwise-pretrained encoder).

    No BatchNorm/running-stats (ViT uses LayerNorm), so unlike the ResNet
    encoder this never populates batch_stats - the mutable=["batch_stats"]
    call path exists purely so _encode_condition's train/eval branching does
    not need to special-case the encoder type.

    Params are seeded from the pretrained checkpoint (not random) and then
    live in the ordinary state.params["vision"] slot, so they get fine-tuned
    jointly with the rest of the network by the existing AdamW optimizer,
    exactly like PlainResNet18Encoder's weights - there is no freezing here.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224", image_size: int = 224):
        from transformers import FlaxViTModel

        self.model_name = model_name
        # google/vit-base-patch16-224's own required square input resolution
        # (conf/agent/training/vision.yaml's vit_image_size) - not tunable
        # without a different checkpoint entirely.
        self._image_size = image_size
        self._hf_model = FlaxViTModel.from_pretrained(model_name, dtype=jnp.float32, add_pooling_layer=False)

    def init(self, rng, dummy_img: jnp.ndarray, train: bool = False):
        del rng, dummy_img, train  # weights come from the pretrained checkpoint, not from random init
        return {"params": self._hf_model.params}

    def apply(self, variables, imgs: jnp.ndarray, train: bool = False, mutable=None):
        del train
        # imgs already arrive in [-1, 1] (xqc/agents/common/dataset.py) - the
        # checkpoint's own mean=std=0.5 preprocessing is exactly that [0, 1]
        # -> [-1, 1] affine map, so it's already satisfied; only the resize
        # is needed here, no further rescale (re-applying (x-0.5)/0.5 to
        # already-[-1,1] input would double-transform it out of range).
        pixel_values = jax.image.resize(
            imgs, (imgs.shape[0], self._image_size, self._image_size, imgs.shape[-1]), method="bilinear"
        )
        out = self._hf_model.module.apply(
            {"params": variables["params"]}, pixel_values=pixel_values, deterministic=True
        )
        feat = out.last_hidden_state[:, 0, :]  # CLS token
        return (feat, {"batch_stats": {}}) if mutable else feat


def crop_images(
    imgs: jnp.ndarray, crop_shape: Tuple[int, int], train: bool, crop_key: Optional[jax.random.PRNGKey],
) -> jnp.ndarray:
    """imgs: (B, To, H, W, 3). Independent random crop offset per (batch
    item, To frame) - matches robomimic's own CropRandomizer.forward_in,
    which "samples N random crops for each input in the batch": the
    reference's own obs-encoder call flattens (B, To) into one batch
    dimension before invoking CropRandomizer (see
    DiffusionUnetHybridImagePolicy.predict_action's
    `x[:,:To,...].reshape(-1,*x.shape[2:])`), so CropRandomizer cannot tell
    timesteps apart from batch items - every observation timestep gets its
    own independent crop, not one shared across a sample's To-history.
    train=True: random offset (needs crop_key), sampled from [0, max_y) /
    [0, max_x) - matches robomimic's own sample_random_image_crops exactly
    (`(max_sample_h * torch.rand(...)).long()`: a continuous [0,1) draw
    scaled by max_sample_h then floored, so max_sample_h itself is never
    reached - i.e. the bottom/right-most crop offset is excluded). train=
    False: deterministic center crop (no RNG).

    Shared by every ChiUNet-backed learner (dp/fm/mf/mf_2_step/
    mf_invertible/nf/mip/mip_flow/mip_flow_injection) - was defined
    separately (and only) in dp_learner.py, and only wired into DP's own
    _encode_condition - see git history for the parity gap this closes."""
    bs, To, h, w, c = imgs.shape
    ch, cw = crop_shape
    max_y, max_x = h - ch, w - cw
    if train:
        key_y, key_x = jax.random.split(crop_key)
        offsets_y = jax.random.randint(key_y, (bs, To), 0, max_y)
        offsets_x = jax.random.randint(key_x, (bs, To), 0, max_x)
    else:
        offsets_y = jnp.full((bs, To), max_y // 2)
        offsets_x = jnp.full((bs, To), max_x // 2)

    def crop_one(img, oy, ox):
        return jax.lax.dynamic_slice(img, (oy, ox, 0), (ch, cw, c))

    return jax.vmap(jax.vmap(crop_one))(imgs, offsets_y, offsets_x)


def build_vision_encoder(
    vision_encoder: str,
    norm_type: str = "batch",
    spatial_softmax_num_kp: int = 32,
    spatial_softmax_feature_dim: int = 64,
    vit_image_size: int = 224,
):
    """Factory used by the learners: "resnet" -> PlainResNet18Encoder() (random
    init, trained from scratch), "vit" -> PretrainedViTEncoder() (downloads
    google/vit-base-patch16-224 on first use, fine-tuned jointly thereafter).
    Both expose the same init/apply duck-typed interface. norm_type/
    spatial_softmax_num_kp/spatial_softmax_feature_dim are only meaningful
    for "resnet" (see PlainResNet18Encoder/_norm_layer/SpatialSoftmaxPool);
    vit_image_size only for "vit" - every caller passes these explicitly
    from conf/agent/training/vision.yaml now (see
    ChunkedPolicyLearner.__init__)."""
    if vision_encoder == "resnet":
        return PlainResNet18Encoder(
            norm_type=norm_type,
            spatial_softmax_num_kp=spatial_softmax_num_kp,
            spatial_softmax_feature_dim=spatial_softmax_feature_dim,
        )
    if vision_encoder == "vit":
        return PretrainedViTEncoder(image_size=vit_image_size)
    raise ValueError(f"Unknown vision_encoder {vision_encoder!r}, expected 'resnet' or 'vit'")
