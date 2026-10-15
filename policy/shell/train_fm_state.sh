#!/usr/bin/env bash
# Flow matching (conditional flow-matching BC policy, xqc/agents/actors/fm/)
# on the pick-and-place task, STATE OBSERVATIONS ONLY (no cameras/vision),
# comparison run against MIP (see train_mip_no_squash_state.sh/
# train_mip_tanh_state.sh): offline BC pretraining only - FM has no online
# RL path at all (xqc/agents/fm/flow_matching.py raises NotImplementedError
# if max_steps>0), so this always stops after actor_pre_training_steps.
#
# agent=fm_state_abs (conf/agent/fm_state_abs.yaml): camera_keys=[] makes
# xqc/envs/pick_place.py construct the env with camera_names=None, which
# flips on use_object_obs - the policy gets robot0_eef_pos/quat/gripper_qpos
# PLUS "object" (bottle + container positions) as plain low-dim state (see
# task.py's world_lowdim_obs/flat_obs), no images, no camera rendering, no
# ResNet encoder.
#
# SINGLE seed/GPU (not 4-in-parallel - explicit request: a 4-GPU node
# allocation took too long to get scheduled on workq, 1 GPU is much faster
# to get). Run this 3+ times with different SEED values if you want the
# noise-vs-signal comparison back later.
#
# Reads:  videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5 (+ its env_scene.xml)
# Writes: checkpoints/FM/<RUN_NAME>/ (step_<N>/ and best/ checkpoints,
#         eval_videos/step_<N>/*.mp4, train.log) and a W&B run named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_fm_state.sh            # GPU 0
#     GPU=1 bash policy/shell/train_fm_state.sh
#     SEED=1 bash policy/shell/train_fm_state.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
SEED="${SEED:-0}"

# RUN_TAG (no seed) is the wandb run name - re-running this script with a
# different SEED at the same moment would otherwise get its own name; this
# way separate seeds launched close together can still be grouped on wandb.
# SAVE_DIR still gets the seed suffix so checkpoints never collide.
RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)_fm_state_bc_only"
SAVE_DIR="checkpoints/FM/${RUN_TAG}_seed${SEED}"
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"

uv run policy/train_parallel_chunking.py \
    agent=fm_state_abs \
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
    wandb.project_name=humanoid_challenge_state \
    wandb.name="$RUN_TAG" \
    save_dir="$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/train.log"
