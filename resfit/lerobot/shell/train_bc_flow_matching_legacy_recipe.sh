#!/usr/bin/env bash
# Flow matching (Diffusion Policy's Unet and encoder, flow-matching objective, 10 Euler steps)
# with the training recipe of the earlier JAX pipeline (see train_bc_legacy_bc_absolute_6d.sh):
# EMA of the weights, its optimizer (betas 0.95/0.999, 500-step warmup + cosine), no gradient
# clipping, agentview camera only. The chunk alignment (Diffusion Policy's) and the crop (115)
# stay this pipeline's own. On the absolute-action dataset with 6D rotations, normalized per
# dimension, on GPU 1. Batch 64, as legacy_bc. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_legacy_recipe.sh
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_6d_agentview_To2_Ta16_h8_ema_legacyopt_bs64" \
GPU=1 \
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
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
