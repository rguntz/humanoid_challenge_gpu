#!/usr/bin/env bash
# BC pre-training on the delta-action dataset, on GPU 0. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_delta.sh
# Sized to run alongside train_bc_absolute.sh (the other GPU, 24 loader
# workers each on the 64 cores). Other settings: see train_bc.sh.
GPU=0 \
ACTION_MODE=delta \
STEPS=100000 \
NUM_WORKERS=24 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
