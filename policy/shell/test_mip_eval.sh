#!/usr/bin/env bash
# Quick smoke test of the offline eval path of policy/train_parallel_chunking.py
# with the same MIP agent as policy/shell/train_mip.sh: 500 actor BC steps
# (no critic update, no online RL), then ONE eval at step 500 - EMA eval with
# video + live-weights eval (ema_rate=1.0, agent.ema_rate=0.995 < 1), checkpoint
# save. A small eval (2 replicas x 1 episode) so it finishes in minutes.
#
# Writes: checkpoints/MIP_test/<RUN_NAME>/ (step_500/, best/, eval_videos/step_500/*.mp4,
#         train.log). W&B in offline mode (logged under /tmp/wandb, nothing uploaded).
#
# Run from anywhere:
#     bash policy/shell/test_mip_eval.sh            # GPU 2
#     GPU=1 bash policy/shell/test_mip_eval.sh
# Training (JAX) and every env worker (eval rendering, MuJoCo EGL) use only GPU $GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-2}"
SEED=0
STEPS=500  # actor BC steps; the single eval runs at the last one

RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_test_mip_eval_steps${STEPS}_seed${SEED}"
SAVE_DIR="checkpoints/MIP_test/${RUN_NAME}"
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"

uv run policy/train_parallel_chunking.py \
    agent=mip_vision_abs \
    agent.squash_actions=true \
    agent.squash_scale=standardize \
    agent.actor_pre_training_steps=$STEPS \
    agent.critic_kwargs.critic_pre_training_steps=0 \
    max_steps=0 \
    eval_interval=$STEPS \
    eval_at_start=false \
    num_seeds=1 \
    eval_num_seeds=2 \
    eval_episodes=1 \
    eval_video=true \
    eval_video_local=true \
    eval_video_episodes=1 \
    seed=$SEED \
    wandb.mode=offline \
    wandb.name="$RUN_NAME" \
    save_dir="$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/train.log"
