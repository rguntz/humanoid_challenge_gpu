#!/usr/bin/env bash
# SmolVLA (LeRobot's vision-language-action model, resfit/lerobot/policies/smolvla/), started from
# the pretrained lerobot/smolvla_base and fine-tuned on absolute end-effector actions with 6D
# rotations, as train_bc_unet_bc_absolute_6d.sh (pick_place_bottle_absolute_6d: position,
# rotation as two rows of its matrix, gripper; state: end-effector position, quaternion and
# fingers), with the agentview camera and the task sentence. Otherwise the same as
# train_bc_smolvla_joint.sh: one observation frame, chunks of 50 actions all executed, SmolVLA's
# optimizer and schedule, gradient clipping 10, mean/std normalization, the vision-language model
# frozen and the action expert trained. On GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_smolvla_absolute_6d.sh
# ~0.64 s per step at batch 64 on an RTX 3090 (~9 GB of GPU memory): ~5.5 h for 30000 steps.
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_smolvla_6d_agentview_chunk50" \
GPU=2 \
POLICY=smolvla \
PRETRAINED=lerobot/smolvla_base \
DATASET=pick_place_bottle \
ACTION_MODE=absolute_6d \
NORMALIZER=mean_std \
CAMERAS="agentview" \
CHUNK_SIZE=50 \
N_ACTION_STEPS=50 \
BATCH_SIZE=64 \
OPTIMIZER=policy \
GRAD_CLIP_NORM=10 \
EMA=0 \
STEPS=100000 \
EVAL_FREQ=10000 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
