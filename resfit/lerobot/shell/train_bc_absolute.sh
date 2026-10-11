#!/usr/bin/env bash
# BC pre-training on the absolute-action dataset, on GPU 2. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_absolute.sh
# Other settings: see train_bc.sh.
GPU=2 \
ACTION_MODE=absolute \
STEPS=100000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
