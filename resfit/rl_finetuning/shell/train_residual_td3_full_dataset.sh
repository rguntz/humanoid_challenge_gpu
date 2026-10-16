#!/usr/bin/env bash
# train_residual_td3.sh with the offline buffer filled from the full demos (the dataset the base
# policy was trained on: the demos continue after the first success, to the end of the recording)
# instead of the demos cut at their first success. Every other setting is train_residual_td3.sh's
# (set them on the command line as there, e.g. GPU=0 TERMINATE_ON_FALL=1). The W&B run name
# starts with fulldata. Run from anywhere:
#     bash resfit/rl_finetuning/shell/train_residual_td3_full_dataset.sh
# Note: the offline reward is 1 on each demo's last frame only, so here the frames where the
# bottle already sits in the box get reward 0 and are not terminal (online, the episode ends
# with reward 1 at the first success).
set -euo pipefail

export DATASET="${DATASET:-pick_place_bottle_full_absolute_6d}"
export EXTRA_ARGS="wandb.name=fulldata ${EXTRA_ARGS:-}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/train_residual_td3.sh"
