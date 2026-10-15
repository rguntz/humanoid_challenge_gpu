#!/usr/bin/env bash
# Residual RL (ResFiT's residual TD3) on top of a train_bc.py base policy, on the bottle
# pick-and-place task. The residual agent sees both cameras, the state and the base policy's action,
# and adds a small correction to that action (at most ACTION_SCALE of each normalized dimension);
# the base policy is frozen. Offline transitions come from the demos (episodes ending at the first
# success, which is rewarded). Edit the settings below, then run from anywhere:
#     bash resfit/rl_finetuning/shell/train_residual_td3.sh
# Reads BASE_POLICY (a policy folder saved by train_bc.py) and DATASET (the LeRobot dataset: offline
# transitions, action/state normalization, and the env's action mode and scene). Writes
# outputs/resfit_residual/<run name>/: models/best and models/latest (the residual agent,
# agent.pt, with the evaluation that saved it), outputs/ (one evaluation video per evaluation, every
# episode, and Q-value plots) and config.json; the replay-buffer caches in outputs/resfit_residual/cache/.
# Logs to the W&B project below (videos stay local).
# Debug mode first (INFERENCE_ONLY=1): only the base policy runs, in the training env's rollout loop
# (zero residual, no noise; nothing trained, collected or saved), for INFERENCE_EPISODES episodes;
# training/success_rate_last50 should converge to the base checkpoint's eval_bc.sh score:
#     INFERENCE_ONLY=1 bash resfit/rl_finetuning/shell/train_residual_td3.sh
set -euo pipefail

# ---- settings -------------------------------------------------------------------
GPU="${GPU:-1}"                                   # GPU for the agent, the base policy and the rendering
BASE_POLICY="${BASE_POLICY:-outputs/resfit_bc/2026-10-13_18-47-18_flow_matching_6d_agentview_To2_Ta16_h8_ema_legacyopt_jaxmatch_fullep_cropps_nojitter_fp32/checkpoints/step_0040000/policy}"  # flow matching, step 40k: 49 % on 100 fresh layouts during training
DATASET="${DATASET:-pick_place_bottle_absolute_6d}"   # the base policy's action mode; episodes end at the first success
TOTAL_STEPS="${TOTAL_STEPS:-300000}"              # environment steps (one training env)
EVAL_EVERY="${EVAL_EVERY:-10000}"                 # evaluate, save models/latest (and models/best) every N steps
EVAL_EPISODES="${EVAL_EPISODES:-50}"              # per evaluation (multiple of EVAL_NUM_ENVS)
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-25}"              # simulators run in parallel during evaluation
ACTION_SCALE="${ACTION_SCALE:-0.1}"               # largest residual, as a fraction of each normalized action dimension
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pick-place-residual-td3}"
WANDB_MODE="${WANDB_MODE:-online}"                # online | offline | disabled
# Task options (resfit/pick_place/env.py), for the training and evaluation envs; defaults: the original task
TERMINATE_ON_FALL="${TERMINATE_ON_FALL:-0}"       # 1: end an episode (reward 0) when the bottle falls outside the container
ROBOT_START="${ROBOT_START:-demo}"                # demo (a random demo's start) | near_home (home pose +- ROBOT_START_NOISE_DEG)
ROBOT_START_NOISE_DEG="${ROBOT_START_NOISE_DEG:-5}"
BOX_POSITION="${BOX_POSITION:-random}"            # random | fixed (one container position for every episode)
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
    pick_place.terminate_on_fall="$([[ "$TERMINATE_ON_FALL" == 1 ]] && echo true || echo false)" \
    pick_place.robot_start="$ROBOT_START" \
    pick_place.robot_start_noise_deg="$ROBOT_START_NOISE_DEG" \
    pick_place.box_position="$BOX_POSITION" \
    inference_only_episodes="$INFERENCE_EPISODES" \
    hydra.run.dir=. hydra.output_subdir=null \
    $EXTRA_ARGS
