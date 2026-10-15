#!/usr/bin/env bash
# MIP WITH tanh squashing (deterministic 2-step flow-map policy,
# xqc/agents/actors/mip/) on the pick-and-place task, STATE OBSERVATIONS ONLY
# (no cameras/vision), comparison run against FM (train_fm_state.sh) and
# MIP-WITHOUT-tanh (train_mip_no_squash_state.sh):
#   1. offline: BC pretraining of the actor + SARSA pretraining of the
#      xqc_style critic on the demos (agent.critic_kwargs.critic_pre_training_steps
#      =100_000, the "full run" recipe - critic gets real offline convergence
#      before online RL starts, unlike policy/shell/train_mip.sh's current
#      CRITIC_PRE_STEPS=0 debug setting);
#   2. online RL (max_steps=200_000): the actor maximizes the critic's Q,
#      regularized toward the frozen BC-pretrained MIP by
#      PRIOR_MSE * ||a_live - a_prior||^2 (agent.wasserstein_regularization).
#      At the transition both the live actor and the prior start from the EMA
#      weights (reset_to_ema_at_transition=true); Adam state and the LR
#      schedule carry on untouched.
#
# WITH squashing: tanh (agent.squash_actions=true) with squash-space
# standardization (agent.squash_scale=standardize, computed from the
# dataset) - the MIP trains on safe_arctanh(action) and sample() returns
# tanh(...), bounded to (-1, 1). See train_mip_no_squash_state.sh for the
# WITHOUT-tanh counterpart.
#
# agent=mip_state_abs (conf/agent/mip_state_abs.yaml): camera_keys=[] makes
# xqc/envs/pick_place.py construct the env with camera_names=None, which
# flips on use_object_obs - the policy gets robot0_eef_pos/quat/gripper_qpos
# PLUS "object" (bottle + container positions) as plain low-dim state, no
# images/camera rendering/ResNet encoder.
#
# SINGLE seed/GPU (not 4-in-parallel - explicit request: a 4-GPU node
# allocation took too long to get scheduled on workq, 1 GPU is much faster
# to get). Run this 3+ times with different SEED values if you want the
# noise-vs-signal comparison back later.
#
# Reads:  videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5 (+ its env_scene.xml)
# Writes: checkpoints/MIP/<RUN_NAME>/ (step_<N>/ and best/ checkpoints,
#         eval_videos/step_<N>/*.mp4, train.log) and a W&B run named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_mip_tanh_state.sh            # GPU 0
#     GPU=1 bash policy/shell/train_mip_tanh_state.sh
#     SEED=1 bash policy/shell/train_mip_tanh_state.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
SEED="${SEED:-0}"

PRIOR_MSE=3.0  # coefficient of the live-vs-prior MIP MSE (0 = off)
POLICY_DELAY=3  # actor step every POLICY_DELAY critic steps (online)
CRITIC_PRE_STEPS=100_000
MAX_STEPS=200_000

# RUN_TAG (no seed) is the wandb run name - re-running this script with a
# different SEED at the same moment would otherwise get its own name; this
# way separate seeds launched close together can still be grouped on wandb.
# SAVE_DIR still gets the seed suffix so checkpoints never collide.
RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)_mip_state_squash_std_critic${CRITIC_PRE_STEPS}_online${MAX_STEPS}_xqc_style_prior_mse${PRIOR_MSE}_delay${POLICY_DELAY}"
SAVE_DIR="checkpoints/MIP/${RUN_TAG}_seed${SEED}"
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"

uv run policy/train_parallel_chunking.py \
    agent=mip_state_abs \
    agent.squash_actions=true \
    agent.squash_scale=standardize \
    agent.actor_pre_training_steps=100_000 \
    agent.critic_kwargs.critic_pre_training_steps=$CRITIC_PRE_STEPS \
    agent.pretrain_use_sarsa=true \
    agent.offline_data_ratio=0.5 \
    agent.policy_delay=$POLICY_DELAY \
    agent.ema_rate_online=1.0 \
    agent.wasserstein_regularization=$PRIOR_MSE \
    agent.wasserstein_offline_only=false \
    agent.bc_regularization=null \
    agent.critic_bootstrap_use_ema=false \
    agent.critic_bootstrap_best_of_n=1 \
    agent.rollout_best_of_n=1 \
    agent.fix_reward_normalization_scale_jump=true \
    reset_to_ema_at_transition=true \
    max_steps=$MAX_STEPS \
    start_training_critic=0 \
    start_training_actor=0 \
    rollout_warmup_steps=10_000 \
    updates_per_step=2 \
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
