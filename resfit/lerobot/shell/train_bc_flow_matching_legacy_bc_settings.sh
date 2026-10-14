#!/usr/bin/env bash
# Flow matching with exactly the settings of train_bc_legacy_bc_absolute_6d.sh: only the policy
# differs, flow matching on the Unet of train_bc_flow_matching_jax_match.sh (widths (512, 1024,
# 2048), the earlier JAX pipeline's, flow time unscaled, 10 Euler steps to sample) instead of
# legacy_bc's MLP head. So, as legacy_bc here:
#   - dataset pick_place_bottle (episodes end at the demo's first success)
#   - random crop 116 x 116 at one offset for the whole batch, train_bc's colour jitter, no
#     random shift
#   - images normalized with each channel's dataset mean and standard deviation
#   - the chunk starts at the current step (actions t .. t + 15)
#   - the copy-padded actions past an episode's end are left out of the loss, every sample kept
#     (do_mask_loss_for_padding, the default)
#   - the sampled chunk is clipped to [-1, 1] (clip_sample, the default)
#   - To 2, Ta 16, 8 actions executed, agentview, per-dimension range normalization, the earlier
#     JAX pipeline's optimizer and schedule, EMA, no gradient clipping, batch 64, 100000 steps
# (train_bc_flow_matching_jax_match.sh instead sets all of these as policy/ has them.)
# On the absolute-action dataset with 6D rotations, on GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_legacy_bc_settings.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_absolute_6d. Writes
# outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per evaluation) and logs
# to the W&B project below. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_6d_agentview_To2_Ta16_h8_ema_legacy_bc_settings" \
GPU=0 \
POLICY=flow_matching \
DATASET=pick_place_bottle \
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
STEPS=100000 \
EXTRA_ARGS="--policy_kwargs down_dims=[512,1024,2048],actions_from_current_step=true,crop_shape=[116,116]" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
