#!/usr/bin/env bash
# train_bc_smolvla_joint.sh with the vision-language model trained too, so its vision encoder can
# adapt to these 128 x 128 renders (pretrained on real photos): freeze_vision_encoder=false and
# train_expert_only=false. SmolVLA cannot train the vision encoder alone: train_expert_only=true
# freezes the whole vision-language model, so this also trains its language layers (all but the
# last): 403M trained parameters instead of 100M, at SmolVLA's learning rate (1e-4).
# Everything else identical. On GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_smolvla_joint_unfrozen.sh
# GPU memory, measured on an RTX 3090: 11.6 GB at batch 16, 19.8 GB at batch 32; batch 64 needs
# ~36 GB (fine on an H200, not on a 24 GB card: use BATCH_SIZE=32 there).
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_smolvla_joint_agentview_chunk50_unfrozen" \
GPU=1 \
POLICY=smolvla \
PRETRAINED=lerobot/smolvla_base \
DATASET=pick_place_bottle \
ACTION_MODE=joint \
NORMALIZER=mean_std \
CAMERAS="agentview" \
CHUNK_SIZE=50 \
N_ACTION_STEPS=50 \
BATCH_SIZE=64 \
OPTIMIZER=policy \
GRAD_CLIP_NORM=10 \
EMA=0 \
STEPS=30000 \
EVAL_FREQ=5000 \
EXTRA_ARGS="--policy_kwargs freeze_vision_encoder=false,train_expert_only=false" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
