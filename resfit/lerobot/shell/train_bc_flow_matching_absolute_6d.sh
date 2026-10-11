#!/usr/bin/env bash
# BC pre-training of Diffusion Policy's Unet trained by flow matching (flow_matching: velocity
# loss, 10 Euler steps to sample; see train_bc.py) on the absolute-action dataset with 6D
# rotations, normalized per dimension, on GPU 0. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_absolute_6d.sh
# Images are randomly cropped by the policy itself, so RANDOM_SHIFT stays 0.
# Other settings: see train_bc.sh.
GPU=1 \
POLICY=flow_matching \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
