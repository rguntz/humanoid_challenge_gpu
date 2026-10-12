#!/usr/bin/env bash
# BC pre-training of the MLP policy (ACT's vision, an MLP instead of the transformer:
# resfit/lerobot/policies/mlp/) on the home-pose dataset (every episode starts from the same
# home pose: resfit/pick_place/regenerate_dataset.py --home), absolute actions with 6D rotations,
# normalized per dimension, on GPU 3. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_mlp_home_absolute_6d.sh
# Other settings: see train_bc.sh.
GPU=1 \
POLICY=mlp \
DATASET=pick_place_bottle_home \
ACTION_MODE=absolute_6d \
NORMALIZER=per_dim_range \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
