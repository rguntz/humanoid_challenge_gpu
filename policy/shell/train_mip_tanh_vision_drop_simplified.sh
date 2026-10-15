#!/usr/bin/env bash
# Ablation run: MIP WITH tanh squashing, VISION observations
# (camera_keys=[agentview_image], trainable ResNet18 encoder - see
# conf/agent/mip_vision_abs.yaml), same recipe as train_mip_tanh_vision.sh,
# EXCEPT it resumes straight into online fine-tuning from that run's own
# already-completed BC-pretrained checkpoint (step_100000 - see
# train_mip_tanh_vision_drop_term.sh's own header comment for the full
# resume mechanism), and turns on BOTH env.terminate_on_drop=true AND
# env.simplified_init=true for the online phase:
#   - terminate_on_drop: episode ends the moment the bottle's cap touches
#     the table (tipped/fallen), instead of running out the full horizon
#     with a dropped bottle sitting uselessly in the replay buffer.
#   - simplified_init: robot start pose narrowed to a small neighborhood
#     around one canonical pose, container position FIXED ("the new home
#     box position") - only the bottle keeps moving across its full range.
#
# This is the VISION counterpart of train_mip_tanh_state_drop_simplified.sh.
# On the STATE task, the simplified_init change (alone, and combined with
# terminate_on_drop) started near 1% success rate right after resuming -
# the very first eval post-resume, not a converged number, and those two
# state jobs were cancelled rather than waiting to see if it recovered (see
# conversation history). This vision run is launched anyway (explicit
# request) to check the combination directly on the actual task that
# prompted this investigation - watch its first few evals (step 110_000
# onward) the same way before concluding anything from an early low number.
#
# Online-phase-only change: offline data/pretraining is identical to the
# baseline.
#
# Compare against: the original train_mip_tanh_vision.sh run this resumes
# from, and against train_mip_tanh_vision_drop_term.sh (terminate_on_drop
# alone).
#
# Reads:  checkpoints/MIP/_resume_sources/mip_tanh_vision_step_100000 (+ the
#         same dataset as the baseline for the offline buffer mix,
#         agent.offline_data_ratio=0.5)
# Writes: checkpoints/MIP/<RUN_NAME>/ (step_<N>/ and best/ checkpoints,
#         eval_videos/step_<N>/*.mp4, train.log) and a W&B run (project
#         humanoid_challenge_vision) named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_mip_tanh_vision_drop_simplified.sh            # GPU 0
#     GPU=1 bash policy/shell/train_mip_tanh_vision_drop_simplified.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
SEED="${SEED:-0}"

PRIOR_MSE=3.0  # coefficient of the live-vs-prior MIP MSE (0 = off)
POLICY_DELAY=3  # actor step every POLICY_DELAY critic steps (online)
CRITIC_PRE_STEPS=100_000
MAX_STEPS=200_000
RESUME_FROM="checkpoints/MIP/_resume_sources/mip_tanh_vision_step_100000"
RESUME_STEP=100000

RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)_mip_vision_squash_std_dropterm_simplifiedinit_critic${CRITIC_PRE_STEPS}_online${MAX_STEPS}_xqc_style_prior_mse${PRIOR_MSE}_delay${POLICY_DELAY}"
SAVE_DIR="checkpoints/MIP/${RUN_TAG}_seed${SEED}"
mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU"

uv run policy/train_parallel_chunking.py \
    agent=mip_vision_abs \
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
    wandb.project_name=humanoid_challenge_vision \
    wandb.name="$RUN_TAG" \
    save_dir="$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/train.log"
