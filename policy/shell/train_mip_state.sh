#!/usr/bin/env bash
# MIP (deterministic 2-step flow-map policy, xqc/agents/actors/mip/) on the
# pick-and-place task, STATE OBSERVATIONS ONLY (no cameras/vision), in ONE
# run of policy/train_parallel_chunking.py:
#   1. offline: BC pretraining of the actor + SARSA pretraining of the
#      xqc_style critic on the demos;
#   2. online RL: the actor maximizes the critic's Q, regularized toward the
#      frozen BC-pretrained MIP by PRIOR_MSE * ||a_live - a_prior||^2
#      (agent.wasserstein_regularization; PRIOR_MSE=0 = no regularization).
#      At the transition both the live actor and the prior start from the EMA
#      weights (reset_to_ema_at_transition=true); Adam state and the LR
#      schedule carry on untouched.
#
# agent=mip_state_abs (conf/agent/mip_state_abs.yaml): camera_keys=[] makes
# xqc/envs/pick_place.py construct the env with camera_names=None, which
# flips on use_object_obs - the policy gets robot0_eef_pos/quat/gripper_qpos
# PLUS "object" (bottle + container positions) as plain low-dim state (see
# task.py's world_lowdim_obs/flat_obs), no images, no camera rendering, no
# ResNet encoder. Meant for fast pipeline iteration/debugging - much cheaper
# per step than policy/shell/train_mip_no_squash.sh's vision run - before
# committing to a long vision run.
#
# No squashing (matches train_mip_no_squash.sh, not train_mip.sh): no tanh
# (agent.squash_actions=false) and no squash-space standardization
# (agent.squash_scale=none).
#
# Reads:  videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5 (+ its env_scene.xml)
# Writes: checkpoints/MIP/<RUN_NAME>/ (step_<N>/ and best/ checkpoints, eval_videos/step_<N>/*.mp4,
#         train.log) and a W&B run named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_mip_state.sh            # GPU 2
#     GPU=1 bash policy/shell/train_mip_state.sh
# Training (JAX) and every env worker (training + eval rendering, MuJoCo EGL
# - only used for eval_video here, no obs images) use only GPU $GPU: JAX
# sees just that device via CUDA_VISIBLE_DEVICES, and EGL renders on it via
# MUJOCO_EGL_DEVICE_ID (conf/config.yaml's env_set sets the same value; the
# spawned env workers inherit both).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-2}"
SEED=0
PRIOR_MSE=3.0  # coefficient of the live-vs-prior MIP MSE (0 = off)
POLICY_DELAY=3  # actor step every POLICY_DELAY critic steps (online)
# Actor-BC-only test (default here, for fast pipeline iteration):
# CRITIC_PRE_STEPS=0 (no critic update) and MAX_STEPS=0 (no online RL).
# Full run: CRITIC_PRE_STEPS=100_000, MAX_STEPS=200_000.
CRITIC_PRE_STEPS=0
MAX_STEPS=0

RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_mip_state_critic${CRITIC_PRE_STEPS}_online${MAX_STEPS}_xqc_style_prior_mse${PRIOR_MSE}_delay${POLICY_DELAY}_seed${SEED}"
SAVE_DIR="checkpoints/MIP/${RUN_NAME}"
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"

uv run policy/train_parallel_chunking.py \
    agent=mip_state_abs \
    agent.squash_actions=false \
    agent.squash_scale=none \
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
    num_seeds=1 \
    eval_num_seeds=10 \
    eval_episodes=10 \
    eval_video=true \
    eval_video_local=true \
    seed=$SEED \
    wandb.project_name=humanoid_challenge_state \
    wandb.name="$RUN_NAME" \
    save_dir="$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/train.log"
