#!/usr/bin/env bash
# BC pre-training on the absolute-action dataset, on GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_absolute.sh
# Sized to run alongside train_bc_delta.sh (the other GPU, 24 loader
# workers each on the 64 cores). Other settings: see train_bc.sh.
GPU=1 \
ACTION_MODE=absolute \
STEPS=100000 \
NUM_WORKERS=24 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
