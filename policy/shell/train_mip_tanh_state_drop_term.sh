#!/usr/bin/env bash
# Ablation run: MIP WITH tanh squashing, STATE observations (no cameras/
# vision - fast to check), same recipe as train_mip_tanh_state.sh, EXCEPT
# it resumes straight into online fine-tuning from that run's own already-
# completed checkpoint (step_100000 - the exact offline->online boundary)
# instead of redoing offline pretraining from scratch (same pretrained
# actor/critic either way, agent.actor_pre_training_steps/critic_kwargs.
# critic_pre_training_steps are left at 100_000 unchanged so the offline
# loop's own step range is empty - see train_parallel_chunking.py's own
# resume_step<=offline_end_step logic), and turns on
# env.terminate_on_drop=true for the online phase: an episode now
# terminates the moment the bottle's cap touches the table (tipped/fallen,
# checked even if the bottle never leaves the table - xqc/envs/pick_place.py's
# _cap_on_table) instead of running out the full horizon with a dropped
# bottle sitting uselessly in the replay buffer for the rest of the episode.
#
# Online-phase-only change: offline data/pretraining is identical to the
# baseline (the dataset has no notion of "dropped" - it's all successful
# demos), so this can only affect what happens once online RL starts
# collecting its own rollouts.
#
# Compare against: the original train_mip_tanh_state.sh run this resumes
# from (checkpoints/MIP/2026-10-14_20-28-06_..._seed0/, a completed
# 0->300_000-step run, diverging only from step_100000).
#
# Reads:  the step_100000 checkpoint below (+ the same dataset as the baseline
#         for the offline buffer mix, agent.offline_data_ratio=0.5)
# Writes: checkpoints/MIP/<RUN_NAME>/ (step_<N>/ and best/ checkpoints,
#         eval_videos/step_<N>/*.mp4, train.log) and a W&B run (project
#         humanoid_challenge_state) named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_mip_tanh_state_drop_term.sh            # GPU 0
#     GPU=1 bash policy/shell/train_mip_tanh_state_drop_term.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
SEED="${SEED:-0}"

PRIOR_MSE=3.0  # coefficient of the live-vs-prior MIP MSE (0 = off)
POLICY_DELAY=3  # actor step every POLICY_DELAY critic steps (online)
CRITIC_PRE_STEPS=100_000
MAX_STEPS=200_000
RESUME_FROM="checkpoints/MIP/_resume_sources/mip_tanh_state_step_100000"
RESUME_STEP=100000

RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)_mip_state_squash_std_dropterm_critic${CRITIC_PRE_STEPS}_online${MAX_STEPS}_xqc_style_prior_mse${PRIOR_MSE}_delay${POLICY_DELAY}"
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
    resume_from="$RESUME_FROM" \
    resume_step=$RESUME_STEP \
    env.terminate_on_drop=true \
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
