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
DATASET="${DATASET:-pick_place_bottle}"       # pick_place_bottle | pick_place_bottle_home (one home start) | pick_place_bottle_wristcanon
                                     # (one grasp orientation) | pick_place_bottle_home_wristcanon: regenerate_dataset.py
ACTION_MODE="${ACTION_MODE:-absolute}"       # absolute | delta | absolute_6d | joint (joint targets, joint state)
NORMALIZER="${NORMALIZER:-mean_std}"        # mean_std (LeRobot) | per_dim_range (state and actions, see train_bc.py)
POLICY="${POLICY:-act}"                    # act | mlp (ACT's vision + MLP head) | diffusion | unet_bc | flow_matching
                                     # (diffusion's Unet) | legacy_bc (the earlier JAX pipeline's plain BC) | smolvla
PRETRAINED="${PRETRAINED:-}"               # smolvla: checkpoint to start from, e.g. lerobot/smolvla_base (empty: none)
CAMERAS="${CAMERAS:-}"                     # cameras the policy gets: "agentview wrist" | agentview | wrist (empty: all)
N_OBS_STEPS="${N_OBS_STEPS:-}"                 # To: observation frames (empty: the policy's own; act and mlp: 1)
CHUNK_SIZE="${CHUNK_SIZE:-16}"              # Ta: actions predicted per policy call (16 = ~1.1 s at 15 Hz)
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"           # execution horizon: actions executed from each chunk before the next call
STEPS="${STEPS:-100000}"               # total optimisation steps
BATCH_SIZE="${BATCH_SIZE:-256}"
OPTIMIZER="${OPTIMIZER:-adamw}"            # adamw (constant lr) | legacy (betas 0.95/0.999, 500-step warmup + cosine)
                                     # | policy (the policy's own optimizer and schedule)
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10}"         # 0: no clipping
EMA="${EMA:-0}"                         # 1: evaluate and save an EMA of the weights (warmup up to 0.9999)
NUM_WORKERS="${NUM_WORKERS:-12}"             # data loading processes: 12 give a batch every ~50 ms, faster than
                                     # a training step; each costs ~1 GB of RAM (too many: oomd kills the session)
IMAGE_NORM="${IMAGE_NORM:-mean_std}"         # mean_std (each channel's dataset mean/std) | minus_one_one (x*2-1, as policy/)
COLOR_JITTER="${COLOR_JITTER:-1}"           # LeRobot's colour jitter augmentation: 1 on, 0 off (policy/ has none)
RANDOM_SHIFT="${RANDOM_SHIFT:-0}"           # random image shift augmentation, in pixels (0: off; 4 is usual)
PRECISION="${PRECISION:-bf16}"             # bf16 (mixed precision, ~1.6-1.9x faster) | fp32
EVAL_FREQ="${EVAL_FREQ:-10000}"            # evaluate, save latest/ (and best/ if better) every N steps
EVAL_EPISODES="${EVAL_EPISODES:-64}"           # episodes per evaluation (multiple of EVAL_NUM_ENVS)
SAVE_EVERY_EVAL="${SAVE_EVERY_EVAL:-0}"      # 1: keep the evaluated policy of every evaluation (checkpoints/step_<N>/)
EVAL_LAYOUTS="${EVAL_LAYOUTS:-fixed}"        # fixed (same start states every evaluation) | fresh (new ones each time, as policy/)
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
# EGL numbers the GPUs in its own order (not $GPU's): find the one that is this GPU
MUJOCO_EGL_DEVICE_ID="$(uv run python resfit/pick_place/egl_device.py)"
export MUJOCO_EGL_DEVICE_ID

ARGS=(--steps "$STEPS")
if [[ -n "$RESUME" ]]; then
    ARGS+=(--resume "$RESUME")
else
    ARGS+=(
        --dataset "videos/task/dataset_sim/lerobot/${DATASET}_${ACTION_MODE}"
        --policy "$POLICY"
        --normalizer "$NORMALIZER"
        --precision "$PRECISION"
        --image_norm "$IMAGE_NORM"
        --color_jitter "$COLOR_JITTER"
        --random_shift "$RANDOM_SHIFT"
        --chunk_size "$CHUNK_SIZE"
        --n_action_steps "$N_ACTION_STEPS"
        --batch_size "$BATCH_SIZE"
        --optimizer "$OPTIMIZER"
        --grad_clip_norm "$GRAD_CLIP_NORM"
        --ema "$EMA"
        --num_workers "$NUM_WORKERS"
        --eval_freq "$EVAL_FREQ"
        --eval_episodes "$EVAL_EPISODES"
        --eval_num_envs "$EVAL_NUM_ENVS"
        --eval_layouts "$EVAL_LAYOUTS"
        --save_every_eval "$SAVE_EVERY_EVAL"
        --seed "$SEED"
        --wandb_project "$WANDB_PROJECT"
    )
    [[ -n "$WANDB_ENTITY" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")
    [[ -n "$RUN_NAME" ]] && ARGS+=(--run_name "$RUN_NAME")
    [[ -n "$N_OBS_STEPS" ]] && ARGS+=(--n_obs_steps "$N_OBS_STEPS")
    [[ -n "$PRETRAINED" ]] && ARGS+=(--pretrained "$PRETRAINED")
    # shellcheck disable=SC2206  # one argument per camera
    [[ -n "$CAMERAS" ]] && ARGS+=(--cameras $CAMERAS)
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is split on purpose
uv run resfit/lerobot/scripts/train_bc.py "${ARGS[@]}" $EXTRA_ARGS
