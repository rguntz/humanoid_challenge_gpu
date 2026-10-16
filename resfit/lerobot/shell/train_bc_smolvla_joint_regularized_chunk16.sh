#!/usr/bin/env bash
# train_bc_smolvla_joint_regularized.sh with one change: chunk size and execution horizon both
# 16 instead of 50/10 (CHUNK_SIZE == N_ACTION_STEPS, every predicted action executed, matching
# how PI0/SmolVLA are typically run - see train_bc_pi0_absolute_6d.sh's header). Everything else
# identical (see train_bc_smolvla_joint_regularized.sh's header for the regularization details:
# random shift 6, EMA, 4000-step schedule evaluated every 500, unfrozen vision-language model).
# On GPU 0. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_smolvla_joint_regularized_chunk16.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_full_joint and lerobot/smolvla_base
# (pick_place_bottle_full, matching train_bc_smolvla_joint_regularized.sh's own dataset
# version). Writes outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per
# evaluation) and logs to the W&B project below. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_smolvla_joint_agentview_chunk16_unfrozen_regularized" \
GPU=0 \
POLICY=smolvla \
PRETRAINED=lerobot/smolvla_base \
DATASET=pick_place_bottle_full \
ACTION_MODE=joint \
NORMALIZER=mean_std \
CAMERAS="agentview" \
CHUNK_SIZE=16 \
N_ACTION_STEPS=16 \
BATCH_SIZE=64 \
OPTIMIZER=policy \
GRAD_CLIP_NORM=10 \
EMA=1 \
RANDOM_SHIFT=6 \
STEPS=4000 \
EVAL_FREQ=500 \
EXTRA_ARGS="--policy_kwargs freeze_vision_encoder=false,train_expert_only=false,scheduler_decay_steps=4000" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
