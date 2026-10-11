#!/usr/bin/env bash
# BC pre-training on the delta-action dataset, on GPU 0. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_delta.sh
# Other settings: see train_bc.sh.
GPU=0 \
ACTION_MODE=delta \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
