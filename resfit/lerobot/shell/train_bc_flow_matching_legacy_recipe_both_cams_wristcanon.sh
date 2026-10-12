#!/usr/bin/env bash
# train_bc_flow_matching_legacy_recipe.sh with both cameras (agentview and wrist), on the
# dataset where every demo grasps with the gripper turned the same way
# (pick_place_bottle_wristcanon: regenerate_dataset.py --canonical-wrist); everything else
# identical. On GPU 3. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_legacy_recipe_both_cams_wristcanon.sh
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_6d_wristcanon_agentview_wrist_To2_Ta16_h8_ema_legacyopt" \
GPU=3 \
POLICY=flow_matching \
DATASET=pick_place_bottle_wristcanon \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
CAMERAS="agentview wrist" \
N_OBS_STEPS=2 \
CHUNK_SIZE=16 \
N_ACTION_STEPS=8 \
BATCH_SIZE=256 \
OPTIMIZER=legacy \
GRAD_CLIP_NORM=0 \
EMA=1 \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
