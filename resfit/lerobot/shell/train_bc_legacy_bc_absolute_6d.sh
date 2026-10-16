#!/usr/bin/env bash
# Plain behaviour cloning (legacy_bc: an MLP 256-256 head on
# Diffusion Policy's encoder, ResNet18 from scratch, MSE on the action chunk, images randomly
# cropped to 116 x 116; see train_bc.py), 2 observation frames, chunks of 16 (8 executed),
# the legacy optimizer and EMA. On the absolute-action dataset with 6D rotations, normalized per dimension, on GPU 1.
# The 48 % run (2026-10-11_22-34-14_legacy_bc_6d_agentview_To2_Ta16_h8_ema). Run from anywhere:
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
PRECISION=bf16 \
IMAGE_NORM=mean_std \
COLOR_JITTER=1 \
RANDOM_SHIFT=0 \
NUM_WORKERS=12 \
STEPS=100000 \
EVAL_FREQ=10000 \
EVAL_EPISODES=64 \
EVAL_NUM_ENVS=16 \
EVAL_LAYOUTS=fixed \
SEED=0 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
