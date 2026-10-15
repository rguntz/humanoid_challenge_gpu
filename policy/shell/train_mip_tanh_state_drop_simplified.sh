#!/usr/bin/env bash
# Ablation run: MIP WITH tanh squashing, STATE observations (no cameras/
# vision - fast to check), same recipe as train_mip_tanh_state.sh, EXCEPT
# it resumes straight into online fine-tuning from that run's own already-
# completed checkpoint (step_100000 - the exact offline->online boundary)
# instead of redoing offline pretraining from scratch (same pretrained
# actor/critic either way - see train_mip_tanh_state_drop_term.sh's own
# header comment for the full resume mechanism), and turns on BOTH
# env.terminate_on_drop=true AND env.simplified_init=true for the online
# phase - the combination of train_mip_tanh_state_drop_term.sh and
# train_mip_tanh_state_simplified_init.sh's own single changes:
#   - terminate_on_drop: episode ends the moment the bottle's cap touches
#     the table (tipped/fallen), instead of running out the full horizon
#     with a dropped bottle sitting uselessly in the replay buffer.
#   - simplified_init: robot start pose narrowed to a small neighborhood
#     around one canonical pose, container position FIXED - only the
#     bottle keeps moving across its full range.
#
# Online-phase-only change: offline data/pretraining is identical to the
# baseline.
#
# Compare against: the original train_mip_tanh_state.sh run this resumes
# from, AND the other two single-change ablations
# (train_mip_tanh_state_drop_term.sh / train_mip_tanh_state_simplified_init.sh) -
# this is the one where if BOTH changes help, their combined effect should
# show up; if only one of the two actually matters, this run's curve should
# track that one ablation's, not the other.
#
# Reads:  the step_100000 checkpoint below (+ the same dataset as the baseline
#         for the offline buffer mix, agent.offline_data_ratio=0.5)
# Writes: checkpoints/MIP/<RUN_NAME>/ (step_<N>/ and best/ checkpoints,
#         eval_videos/step_<N>/*.mp4, train.log) and a W&B run (project
#         humanoid_challenge_state) named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_mip_tanh_state_drop_simplified.sh            # GPU 0
#     GPU=1 bash policy/shell/train_mip_tanh_state_drop_simplified.sh
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

RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)_mip_state_squash_std_dropterm_simplifiedinit_critic${CRITIC_PRE_STEPS}_online${MAX_STEPS}_xqc_style_prior_mse${PRIOR_MSE}_delay${POLICY_DELAY}"
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
    env.simplified_init=true \
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
