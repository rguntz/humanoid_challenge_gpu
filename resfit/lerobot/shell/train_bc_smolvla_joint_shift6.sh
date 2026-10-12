#!/usr/bin/env bash
# train_bc_smolvla_joint.sh with a random image shift of up to 6 pixels (RANDOM_SHIFT=6: padded
# by 6, cropped back to 128 x 128 at a random offset), the geometric augmentation that
# train_bc_legacy_bc_absolute_6d.sh gets from its random 116 x 116 crop; everything else identical.
# Tells whether SmolVLA's gap to legacy_bc on new object placements is the augmentation. On GPU 0.
# Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_smolvla_joint_shift6.sh
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_smolvla_joint_agentview_chunk50_shift6" \
GPU=0 \
POLICY=smolvla \
PRETRAINED=lerobot/smolvla_base \
DATASET=pick_place_bottle \
ACTION_MODE=joint \
NORMALIZER=mean_std \
CAMERAS="agentview" \
CHUNK_SIZE=50 \
N_ACTION_STEPS=50 \
BATCH_SIZE=64 \
OPTIMIZER=policy \
GRAD_CLIP_NORM=10 \
EMA=0 \
RANDOM_SHIFT=6 \
STEPS=30000 \
EVAL_FREQ=5000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
