#!/usr/bin/env bash
# train_bc_flow_matching_jax_match.sh with one change: the image encoder initialized as policy/'s
# JAX encoder (vision_init=flax: Flax's LeCun normal, truncated at 2 standard deviations, biases 0)
# instead of torchvision's kaiming_normal_(mode="fan_out"). Still bf16 mixed precision.
# Everything else exactly as train_bc_flow_matching_jax_match.sh (see its header), which reached
# 43.5 % on 200 fresh layouts (eval_bc.sh, seed 2000; policy/'s FM: 46 %). The run with both
# fp32 and vision_init=flax (train_bc_flow_matching_jax_match_fp32_flaxinit.sh) reached 33.0 %:
# this script and its sibling change one each, to find which one costs the 10 points.
# On GPU 1. Run from anywhere (the two one after the other):
#     bash resfit/lerobot/shell/train_bc_flow_matching_jax_match_fp32.sh && \
#         bash resfit/lerobot/shell/train_bc_flow_matching_jax_match_flaxinit.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_full_absolute_6d. Writes
# outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per evaluation) and logs
# to the W&B project below. Each evaluation draws new random start states (EVAL_LAYOUTS=fresh), as
# policy/ does. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_6d_agentview_To2_Ta16_h8_ema_legacyopt_jaxmatch_fullep_cropps_nojitter_flaxinit" \
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
IMAGE_NORM=minus_one_one \
COLOR_JITTER=0 \
RANDOM_SHIFT=0 \
STEPS=100000 \
EVAL_LAYOUTS=fresh \
EXTRA_ARGS="--policy_kwargs down_dims=[512,1024,2048],actions_from_current_step=true,crop_shape=[116,116],do_mask_loss_for_padding=false,clip_sample=false,flow_time_scale=1,crop_per_sample=true,max_padded_actions=7,vision_init=flax" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
