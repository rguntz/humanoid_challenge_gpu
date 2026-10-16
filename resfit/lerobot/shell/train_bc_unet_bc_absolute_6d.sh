#!/usr/bin/env bash
# Plain behaviour cloning (unet_bc: MSE on the action chunk, one forward pass, no denoising;
# see train_bc.py), with Diffusion Policy's encoder (ResNet18 from scratch, GroupNorm, spatial
# softmax, random crop) and the observation/action windows, optimizer and EMA of the earlier
# JAX pipeline's own plain-BC agent. On the absolute-action dataset with 6D rotations,
# normalized per dimension, on GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_unet_bc_absolute_6d.sh
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_unet_bc_6d_agentview" \
GPU=1 \
POLICY=unet_bc \
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
