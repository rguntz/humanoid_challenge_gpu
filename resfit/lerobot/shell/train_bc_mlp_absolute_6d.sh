#!/usr/bin/env bash
# BC pre-training of the MLP policy (ACT's vision, an MLP instead of the transformer:
# resfit/lerobot/policies/mlp/) on the absolute-action dataset with 6D rotations,
# normalized per dimension, on GPU 2. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_mlp_absolute_6d.sh
# Other settings: see train_bc.sh.
GPU=2 \
POLICY=mlp \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
