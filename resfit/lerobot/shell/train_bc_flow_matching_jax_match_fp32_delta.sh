#!/usr/bin/env bash
# train_bc_flow_matching_jax_match_fp32.sh with one change: delta end-effector actions
# (ACTION_MODE=delta) instead of absolute_6d. Everything else exactly as
# train_bc_flow_matching_jax_match_fp32.sh (see its header).
# On GPU 1. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_flow_matching_jax_match_fp32_delta.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_full_delta (generated via:
#     uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py \
#         --full-episodes --name pick_place_bottle_full --modes delta
# ). Writes outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per
# evaluation) and logs to the W&B project below. Each evaluation draws new random start
# states (EVAL_LAYOUTS=fresh), as policy/ does, over 100 episodes (25 simulators, 4 each:
# about +-5 points instead of +-6 with 64). Every evaluation's policy is kept in
# checkpoints/step_<N>/ (SAVE_EVERY_EVAL=1, ~1 GB each, ~10 GB for the run), to choose the
# base policy of the residual RL afterwards with eval_bc.sh, e.g.
#     CHECKPOINT=checkpoints/step_0040000 RUN=outputs/resfit_bc/<run> bash resfit/lerobot/shell/eval_bc.sh
# Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_flow_matching_delta_agentview_To2_Ta16_h8_ema_legacyopt_jaxmatch_fullep_cropps_nojitter_fp32" \
GPU="${GPU:-1}" \
POLICY=flow_matching \
DATASET=pick_place_bottle_full \
ACTION_MODE=delta \
NORMALIZER=per_dim_range \
CAMERAS="agentview" \
N_OBS_STEPS=2 \
CHUNK_SIZE=16 \
N_ACTION_STEPS=8 \
BATCH_SIZE=64 \
OPTIMIZER=legacy \
GRAD_CLIP_NORM=0 \
EMA=1 \
PRECISION=fp32 \
IMAGE_NORM=minus_one_one \
COLOR_JITTER=0 \
RANDOM_SHIFT=0 \
STEPS="${STEPS:-100000}" \
EVAL_LAYOUTS=fresh \
EVAL_EPISODES=100 \
EVAL_NUM_ENVS=25 \
SAVE_EVERY_EVAL=1 \
EXTRA_ARGS="--policy_kwargs down_dims=[512,1024,2048],actions_from_current_step=true,crop_shape=[116,116],do_mask_loss_for_padding=false,clip_sample=false,flow_time_scale=1,crop_per_sample=true,max_padded_actions=7" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
