#!/usr/bin/env bash
# Flow matching set up like the earlier JAX pipeline's (policy/, agent=fm_vision_abs), which did
# much better than train_bc_flow_matching_legacy_recipe.sh. Differences with that script:
#   - Unet widths (512, 1024, 2048), ~252M parameters (the JAX ChiUNet's model_dim 512 with
#     dim_mult (1, 2, 2); this pipeline's default is (256, 512, 1024), ~65M)
#   - the chunk starts at the current step (actions t .. t + 15, as JAX) instead of Diffusion
#     Policy's t - 1 .. t + 14
#   - random crop 116 x 116 (as JAX and legacy_bc; default here: 115)
#   - the copy-padded actions past an episode's end are trained on (JAX repeats the last action
#     and keeps it in the loss), not masked out, but only in chunks with at most 7 of them
#     (max_padded_actions=7, N_ACTION_STEPS - 1): JAX's mask_reference_region leaves the samples
#     starting in an episode's last 8 frames out of the loss
#   - the sampled chunk is not clipped to [-1, 1] (JAX does not clip)
#   - image augmentation exactly as JAX's: a random crop (116 x 116) with its own offset for every
#     sample and every observation frame (crop_per_sample; this pipeline's default crops the
#     whole batch at one offset), no colour jitter (COLOR_JITTER=0), no random shift
#   - images scaled to [-1, 1] (x * 2 - 1, IMAGE_NORM=minus_one_one), as JAX, instead of each
#     channel's dataset mean and standard deviation
#   - the demos' full episodes, the frames after the first success included, as JAX trains on
#     them (dataset pick_place_bottle_full, see below)
# Kept as JAX had them (and as before): flow time t in [0, 1] given unscaled to the timestep
# embedding (flow_time_scale 1), t ~ U(0, 1) in training, 10 Euler steps to sample, batch 64,
# To 2, Ta 16, 8 actions executed, agentview, EMA, JAX's optimizer and schedule, 100000 steps.
# On the absolute-action dataset with 6D rotations, normalized per dimension, on GPU 1.
# The full-episode dataset is made once with (reads videos/task/dataset_sim/policy/
# pick_place_bottle_base.hdf5, writes videos/task/dataset_sim/lerobot/pick_place_bottle_full_absolute_6d/):
#     uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py --full-episodes \
#         --name pick_place_bottle_full --modes absolute_6d
# Then run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_jax_match.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_full_absolute_6d. Writes
# outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per evaluation) and logs
# to the W&B project below. Each evaluation draws new random start states (EVAL_LAYOUTS=fresh), as
# policy/ does. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_6d_agentview_To2_Ta16_h8_ema_legacyopt_jaxmatch_fullep_cropps_nojitter" \
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
EXTRA_ARGS="--policy_kwargs down_dims=[512,1024,2048],actions_from_current_step=true,crop_shape=[116,116],do_mask_loss_for_padding=false,clip_sample=false,flow_time_scale=1,crop_per_sample=true,max_padded_actions=7" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
