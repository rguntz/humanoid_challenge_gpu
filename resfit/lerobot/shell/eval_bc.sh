#!/usr/bin/env bash
# Evaluate a train_bc.py checkpoint at several execution horizons (actions executed from each
# predicted chunk before the policy is called again), on GPU 0 only (training and rendering).
# Set up as the overfitting test: the SmolVLA unfrozen run's best/, 10 actions per chunk, 50
# episodes started from the layouts of the training demos (compare with the 20 % it gets on
# random layouts at 10 actions). Edit the settings below, then run from anywhere:
#     bash resfit/lerobot/shell/eval_bc.sh
# Reads <RUN>/<CHECKPOINT>/policy and the run's dataset (for the env). Writes, in <RUN>/evals_offline/:
# one video per horizon (<CHECKPOINT>_n<N>_<LAYOUTS>_seed<SEED>.mp4), a line per horizon in
# results.jsonl (success rate, also per round) and a line per episode in episodes.jsonl.
set -euo pipefail

# ---- settings -------------------------------------------------------------------
GPU="${GPU:-0}"
RUN="${RUN:-outputs/resfit_bc/2026-10-12_16-17-59_smolvla_joint_agentview_chunk50_unfrozen}"
CHECKPOINT="${CHECKPOINT:-best}"                 # best | latest | checkpoints/step_<N> (kept with SAVE_EVERY_EVAL=1)
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"           # one or more, e.g. "50 25 10 8"; each <= the run's chunk size
EPISODES="${EPISODES:-50}"                       # per horizon (multiple of NUM_ENVS)
NUM_ENVS="${NUM_ENVS:-25}"                       # simulators in parallel: 2 episodes each
SEED="${SEED:-1000}"                             # 1000: the start states of train_bc's evaluations
LAYOUTS="${LAYOUTS:-demo}"                       # random | demo (the recorded demos' layouts, at most 59 episodes)
# ---------------------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false   # SmolVLA's tokenizer: silences its warning when the env processes start
# EGL numbers the GPUs in its own order (not $GPU's): find the one that is this GPU
MUJOCO_EGL_DEVICE_ID="$(uv run python resfit/pick_place/egl_device.py)"
export MUJOCO_EGL_DEVICE_ID

# shellcheck disable=SC2086  # one argument per horizon
uv run resfit/lerobot/scripts/eval_bc.py --run "$RUN" --checkpoint "$CHECKPOINT" \
    --n_action_steps $N_ACTION_STEPS --episodes "$EPISODES" --num_envs "$NUM_ENVS" --seed "$SEED" --layouts "$LAYOUTS"
