#!/usr/bin/env bash
# train_residual_td3.sh with one change: a delta-action base policy and offline dataset,
# instead of absolute_6d. Residual RL (ResFiT's residual TD3) on top of a train_bc.py base
# policy, on the bottle pick-and-place task. The residual agent sees both cameras, the state
# and the base policy's action, and adds a small correction to that action (at most
# ACTION_SCALE of each normalized dimension); the base policy is frozen. Offline transitions
# come from the demos (episodes ending at the first success, which is rewarded). Edit the
# settings below, then run from anywhere:
#     bash resfit/rl_finetuning/shell/train_residual_td3_delta.sh
# Reads BASE_POLICY (a policy folder saved by train_bc.py, default: the delta flow-matching
# run's best/, 35 % success at step 10000, see
# resfit/lerobot/shell/train_bc_flow_matching_jax_match_fp32_delta.sh) and DATASET (the LeRobot
# dataset: offline transitions, action/state normalization, and the env's action mode and
# scene - pick_place_bottle_delta, not the _full_ variant, since episodes must end at the
# first success for the offline reward to be correct). Writes outputs/resfit_residual/<run
# name>/: models/best and models/latest (the residual agent, agent.pt, with the evaluation
# that saved it), outputs/ (one evaluation video per evaluation, every episode, and Q-value
# plots) and config.json; the replay-buffer caches in outputs/resfit_residual/cache/. Logs to
# the W&B project below (videos stay local).
# Debug mode first (INFERENCE_ONLY=1): only the base policy runs, in the training env's rollout
# loop (zero residual, no noise; nothing trained, collected or saved), for INFERENCE_EPISODES
# episodes; training/success_rate_last50 should converge to the base checkpoint's eval_bc.sh
# score:
#     INFERENCE_ONLY=1 bash resfit/rl_finetuning/shell/train_residual_td3_delta.sh
# Other settings: see train_residual_td3.sh.
set -euo pipefail

# ---- settings -------------------------------------------------------------------
GPU="${GPU:-1}"                                   # GPU for the agent, the base policy and the rendering
BASE_POLICY="${BASE_POLICY:-outputs/resfit_bc/2026-10-14_19-34-32_flow_matching_delta_agentview_To2_Ta16_h8_ema_legacyopt_jaxmatch_fullep_cropps_nojitter_fp32/best/policy}"  # flow matching, delta actions: 35 % at step 10000
DATASET="${DATASET:-pick_place_bottle_delta}"     # the base policy's action mode; episodes end at the first success
TOTAL_STEPS="${TOTAL_STEPS:-300000}"              # environment steps (one training env)
EVAL_EVERY="${EVAL_EVERY:-10000}"                 # evaluate, save models/latest (and models/best) every N steps
EVAL_EPISODES="${EVAL_EPISODES:-50}"              # per evaluation (multiple of EVAL_NUM_ENVS)
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-25}"              # simulators run in parallel during evaluation
ACTION_SCALE="${ACTION_SCALE:-0.1}"               # largest residual, as a fraction of each normalized action dimension
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pick-place-residual-td3}"
WANDB_MODE="${WANDB_MODE:-online}"                # online | offline | disabled
INFERENCE_ONLY="${INFERENCE_ONLY:-0}"             # 1: debug mode, base policy only (see above)
INFERENCE_EPISODES="${INFERENCE_EPISODES:-200}"   # episodes in debug mode
EXTRA_ARGS="${EXTRA_ARGS:-}"                      # other Hydra overrides, e.g. "algo.n_step=5 agent.actor_lr=1e-5"
# ---------------------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
# EGL numbers the GPUs in its own order (not $GPU's): find the one that is this GPU
MUJOCO_EGL_DEVICE_ID="$(uv run python resfit/pick_place/egl_device.py)"
export MUJOCO_EGL_DEVICE_ID
export CACHE_DIR="$REPO/outputs/resfit_residual/cache"   # offline/online replay-buffer caches

# hydra.run.dir=. and hydra.output_subdir=null: no Hydra output folder (the run folder is written
# by the script itself, in outputs/resfit_residual/)
# shellcheck disable=SC2086  # EXTRA_ARGS is split on purpose
uv run resfit/rl_finetuning/scripts/train_residual_td3.py \
    base_policy.path="$BASE_POLICY" \
    offline_data.name="local/$DATASET" \
    offline_data.root="videos/task/dataset_sim/lerobot/$DATASET" \
    algo.total_timesteps="$TOTAL_STEPS" \
    eval_interval_every_steps="$EVAL_EVERY" \
    eval_num_episodes="$EVAL_EPISODES" \
    eval_num_envs="$EVAL_NUM_ENVS" \
    agent.actor.action_scale="$ACTION_SCALE" \
    seed="$SEED" \
    wandb.project="$WANDB_PROJECT" \
    wandb.mode="$WANDB_MODE" \
    inference_only="$([[ "$INFERENCE_ONLY" == 1 ]] && echo true || echo false)" \
    inference_only_episodes="$INFERENCE_EPISODES" \
    hydra.run.dir=. hydra.output_subdir=null \
    $EXTRA_ARGS
