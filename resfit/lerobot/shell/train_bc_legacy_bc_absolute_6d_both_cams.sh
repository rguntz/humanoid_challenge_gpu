#!/usr/bin/env bash
# train_bc_legacy_bc_absolute_6d.sh with both cameras (agentview and wrist) instead of agentview
# only, everything else identical: tells whether the wrist camera is what lowers the success
# rate. On GPU 2. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_legacy_bc_absolute_6d_both_cams.sh
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_legacy_bc_6d_agentview_wrist_To2_Ta16_h8_ema" \
GPU=2 \
POLICY=legacy_bc \
DATASET=pick_place_bottle \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
CAMERAS="agentview wrist" \
N_OBS_STEPS=2 \
CHUNK_SIZE=16 \
N_ACTION_STEPS=8 \
BATCH_SIZE=64 \
OPTIMIZER=legacy \
GRAD_CLIP_NORM=0 \
EMA=1 \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
