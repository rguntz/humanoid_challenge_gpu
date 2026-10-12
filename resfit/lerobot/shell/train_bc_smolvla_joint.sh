#!/usr/bin/env bash
# SmolVLA (LeRobot's vision-language-action model, resfit/lerobot/policies/smolvla/), started from
# the pretrained lerobot/smolvla_base and fine-tuned in joint space: the arm and finger joints as
# state, the demos' recorded joint targets and gripper as actions (pick_place_bottle_joint), the
# agentview camera and the task sentence. SmolVLA's own settings: one observation frame, chunks
# of 50 actions all executed, its optimizer and schedule (AdamW betas 0.9/0.95, 1000-step warmup,
# cosine decay to 2.5e-6 over 30000 steps), gradient clipping 10, mean/std normalization, the
# vision-language model frozen and the action expert trained. On GPU 2. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_smolvla_joint.sh
# ~0.64 s per step at batch 64 on an RTX 3090 (~9 GB of GPU memory): ~5.5 h for 30000 steps.
# The W&B run (and the run folder, outputs/resfit_bc/<RUN_NAME>) is named below; the date keeps
# every launch's name new. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_smolvla_joint_agentview_chunk50" \
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
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
