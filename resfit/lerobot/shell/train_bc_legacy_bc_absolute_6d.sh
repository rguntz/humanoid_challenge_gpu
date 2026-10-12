#!/usr/bin/env bash
# The earlier JAX pipeline's plain BC (policy/train.py, agent=bc_vision_abs), in this training
# script: its MLP head on Diffusion Policy's encoder (ResNet18 from scratch, GroupNorm, spatial
# softmax, random crop 116), its cameras, observation/action windows, optimizer and EMA. On the
# absolute-action dataset with 6D rotations (its abs_action), normalized per dimension, on GPU 1.
# Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_legacy_bc_absolute_6d.sh
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_legacy_bc_6d_agentview_To2_Ta16_h8_ema" \
GPU=1 \
POLICY=legacy_bc \
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
