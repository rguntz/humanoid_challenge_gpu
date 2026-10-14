#!/usr/bin/env bash
# train_bc_flow_matching_jax_match.sh with the two remaining differences from policy/'s flow
# matching (agent=fm_vision_abs) removed too:
#   - full fp32 training (PRECISION=fp32), as policy/ (no mixed precision; its float32 matrix
#     products use TF32 on this GPU), instead of bf16 mixed precision (bf16 keeps ~3 significant
#     digits, and flow matching integrates the predicted velocity over 10 Euler steps). Slower,
#     ~1.6-1.9x per step.
#   - the image encoder initialized as policy/'s (vision_init=flax: Flax's LeCun normal,
#     truncated at 2 standard deviations, biases 0) instead of torchvision's
#     kaiming_normal_(mode="fan_out"): e.g. the 7x7 stem starts with weights ~3x larger
# Everything else exactly as train_bc_flow_matching_jax_match.sh (see its header): Unet widths
# (512, 1024, 2048), chunk from the current step, crop 116 per sample and frame, no colour jitter,
# images in [-1, 1], full episodes, padded actions trained on with at most 7 per chunk, no
# clipping, flow time unscaled, 10 Euler steps, batch 64, To 2, Ta 16, 8 actions executed,
# agentview, EMA, policy/'s optimizer and schedule, 100000 steps. On GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_jax_match_fp32_flaxinit.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_full_absolute_6d. Writes
# outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per evaluation) and logs
# to the W&B project below. Each evaluation draws new random start states (EVAL_LAYOUTS=fresh), as
# policy/ does. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_6d_agentview_To2_Ta16_h8_ema_legacyopt_jaxmatch_fullep_cropps_nojitter_fp32_flaxinit" \
GPU=1 \
POLICY=flow_matching \
DATASET=pick_place_bottle_full \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
CAMERAS="agentview" \
N_OBS_STEPS=2 \
CHUNK_SIZE=16 \
N_ACTION_STEPS=8 \
BATCH_SIZE=64 \
OPTIMIZER=legacy \
GRAD_CLIP_NORM=0 \
EMA=1 \
PRECISION=fp32 \
IMAGE_NORM=minus_one_one \
COLOR_JITTER=0 \
RANDOM_SHIFT=0 \
STEPS=100000 \
EVAL_LAYOUTS=fresh \
EXTRA_ARGS="--policy_kwargs down_dims=[512,1024,2048],actions_from_current_step=true,crop_shape=[116,116],do_mask_loss_for_padding=false,clip_sample=false,flow_time_scale=1,crop_per_sample=true,max_padded_actions=7,vision_init=flax" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
