#!/usr/bin/env bash
# Pre-train the base policy (behaviour cloning) on a pick-and-place LeRobot dataset.
# Edit the variables below (or set them on the command line), then run from anywhere:
#     bash resfit/lerobot/shell/train_bc.sh
#     GPU=1 ACTION_MODE=delta bash resfit/lerobot/shell/train_bc.sh
# Each run writes to outputs/resfit_bc/<run name>/ (checkpoints latest/ and best/,
# evaluation videos) and logs to W&B (run `uv run wandb login` once per machine).
set -euo pipefail

# ---- settings -------------------------------------------------------------------
GPU="${GPU:-0}"                      # GPU used for training, evaluation rendering included
ACTION_MODE="${ACTION_MODE:-absolute}"       # absolute | delta: which LeRobot dataset to train on
POLICY="${POLICY:-act}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"              # actions predicted per policy call (16 = ~1.1 s at 15 Hz)
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"           # actions executed from each chunk before the next call
STEPS="${STEPS:-100000}"               # total optimisation steps
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-32}"             # data loading processes (more gives nothing on 64 cores)
EVAL_FREQ="${EVAL_FREQ:-10000}"            # evaluate, save latest/ (and best/ if better) every N steps
EVAL_EPISODES="${EVAL_EPISODES:-64}"           # episodes per evaluation (multiple of EVAL_NUM_ENVS)
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-16}"           # simulators run in parallel during evaluation
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pick-place-bc}"
WANDB_ENTITY="${WANDB_ENTITY:-}"            # empty: your default W&B entity
RUN_NAME="${RUN_NAME:-}"                # empty: <date>_<policy>_<dataset>
RESUME="${RESUME:-}"                  # a run folder (outputs/resfit_bc/<run name>) to continue it;
                           # the run's own settings are used, except STEPS
EXTRA_ARGS="${EXTRA_ARGS:-}"              # anything else, e.g. "--policy_kwargs dim_model=256"
# ---------------------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"   # EGL counts every GPU, regardless of CUDA_VISIBLE_DEVICES

ARGS=(--steps "$STEPS")
if [[ -n "$RESUME" ]]; then
    ARGS+=(--resume "$RESUME")
else
    ARGS+=(
        --dataset "videos/task/dataset_sim/lerobot/pick_place_bottle_${ACTION_MODE}"
        --policy "$POLICY"
        --chunk_size "$CHUNK_SIZE"
        --n_action_steps "$N_ACTION_STEPS"
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
        --eval_freq "$EVAL_FREQ"
        --eval_episodes "$EVAL_EPISODES"
        --eval_num_envs "$EVAL_NUM_ENVS"
        --seed "$SEED"
        --wandb_project "$WANDB_PROJECT"
    )
    [[ -n "$WANDB_ENTITY" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")
    [[ -n "$RUN_NAME" ]] && ARGS+=(--run_name "$RUN_NAME")
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is split on purpose
uv run resfit/lerobot/scripts/train_bc.py "${ARGS[@]}" $EXTRA_ARGS
