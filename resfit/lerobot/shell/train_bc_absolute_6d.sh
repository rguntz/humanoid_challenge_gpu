#!/usr/bin/env bash
# BC pre-training on the absolute-action dataset with 6D rotations, normalized per
# dimension, on GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_absolute_6d.sh
# Other settings: see train_bc.sh.
GPU=1 \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
