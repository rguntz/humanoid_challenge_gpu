#!/usr/bin/env bash
# MIP (deterministic 2-step flow-map policy, xqc/agents/actors/mip/) on the
# pick-and-place task, in ONE run of policy/train_parallel_chunking.py:
#   1. offline: BC pretraining of the actor + SARSA pretraining of the
#      xqc_style_vision critic on the demos;
#   2. online RL: the actor maximizes the critic's Q, regularized toward the
#      frozen BC-pretrained MIP by PRIOR_MSE * ||a_live - a_prior||^2
#      (agent.wasserstein_regularization; PRIOR_MSE=0 = no regularization).
#      At the transition both the live actor and the prior start from the EMA
#      weights (reset_to_ema_at_transition=true); Adam state and the LR
#      schedule carry on untouched. Same transition settings as xqcfd's
#      scripts/isambard/mf_chunked_stationary_unet_abs_ema_reset_online_rl_sweep.sbatch.
#
# Reads:  videos/task/dataset_sim/policy/pick_place_bottle_base.hdf5 (+ its env_scene.xml)
# Writes: checkpoints/MIP/<RUN_NAME>/ (step_<N>/ and best/ checkpoints, eval_videos/step_<N>/*.mp4,
#         train.log) and a W&B run named <RUN_NAME>.
#
# Run from anywhere:
#     bash policy/shell/train_mip.sh            # GPU 0
#     GPU=1 bash policy/shell/train_mip.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU="${GPU:-0}"
SEED=0
PRIOR_MSE=3.0  # coefficient of the live-vs-prior MIP MSE (0 = off)
POLICY_DELAY=3  # actor step every POLICY_DELAY critic steps (online)

RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_mip_xqc_style_prior_mse${PRIOR_MSE}_delay${POLICY_DELAY}_seed${SEED}"
SAVE_DIR="checkpoints/MIP/${RUN_NAME}"
mkdir -p "$SAVE_DIR"

CUDA_VISIBLE_DEVICES="$GPU" uv run policy/train_parallel_chunking.py \
    agent=mip_vision_abs \
    agent.actor_pre_training_steps=100_000 \
    agent.critic_kwargs.critic_pre_training_steps=100_000 \
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
    max_steps=200_000 \
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
    wandb.name="$RUN_NAME" \
    save_dir="$SAVE_DIR" \
    2>&1 | tee -a "$SAVE_DIR/train.log"
