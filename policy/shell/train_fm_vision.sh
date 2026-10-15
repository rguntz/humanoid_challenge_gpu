#!/usr/bin/env bash
# Flow matching (conditional flow-matching BC policy, xqc/agents/actors/fm/)
# on the pick-and-place task, VISION observations (camera_keys=[agentview_image],
# trainable ResNet18 encoder - see conf/agent/fm_vision_abs.yaml), comparison
# run against MIP (see train_mip_no_squash_vision.sh/train_mip_tanh_vision.sh)
# and against the state-only versions (train_fm_state.sh and friends,
# wandb project humanoid_challenge_state): offline BC pretraining only - FM
# has no online RL path at all (xqc/agents/fm/flow_matching.py raises
# NotImplementedError if max_steps>0), so this always stops after
# actor_pre_training_steps.
#
# SINGLE seed/GPU (matches the state-only scripts' current setup - a 4-GPU
# node allocation took too long to get scheduled on workq). Run this again
# with a different SEED if you want more than one seed.
#
# Reads:  videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5 (+ its env_scene.xml)
# Writes: checkpoints/FM/<RUN_NAME>/ (step_<N>/ and best/ checkpoints,
#         eval_videos/step_<N>/*.mp4, train.log) and a W&B run (project
#         humanoid_challenge_vision) named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_fm_vision.sh            # GPU 0
#     GPU=1 bash policy/shell/train_fm_vision.sh
#     SEED=1 bash policy/shell/train_fm_vision.sh
# Training (JAX) and every env worker (training + eval rendering, MuJoCo EGL)
# use only GPU $GPU: JAX sees just that device via CUDA_VISIBLE_DEVICES, and
# EGL renders on it via MUJOCO_EGL_DEVICE_ID (conf/config.yaml's env_set sets
# the same value; the spawned env workers inherit both).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
SEED="${SEED:-0}"

# RUN_TAG (no seed) is the wandb run name - re-running this script with a
# different SEED at the same moment would otherwise get its own name; this
# way separate seeds launched close together can still be grouped on wandb.
# SAVE_DIR still gets the seed suffix so checkpoints never collide.
RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)_fm_vision_bc_only"
SAVE_DIR="checkpoints/FM/${RUN_TAG}_seed${SEED}"
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"

uv run policy/train_parallel_chunking.py \
    agent=fm_vision_abs \
    agent.actor_pre_training_steps=100_000 \
    max_steps=0 \
    eval_interval=10_000 \
    eval_at_start=true \
    num_seeds=1 \
    eval_num_seeds=10 \
    eval_episodes=10 \
    eval_video=true \
    eval_video_local=true \
    seed=$SEED \
    wandb.project_name=humanoid_challenge_vision \
    wandb.name="$RUN_TAG" \
    save_dir="$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/train.log"
