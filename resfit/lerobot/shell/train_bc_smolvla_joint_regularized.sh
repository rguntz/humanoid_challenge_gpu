#!/usr/bin/env bash
# train_bc_smolvla_joint_unfrozen.sh, regularized against overfitting the 59 demos (its best/
# succeeded 66 % from the demos' own layouts but 20 % from random ones, best at the first
# evaluation, step 5000):
#   - random image shift of up to 6 pixels (DrQ), the size of legacy_bc's random crop (116 of 128)
#   - EMA of the weights (evaluated and saved in best/)
#   - 4000 steps instead of 30000, the learning rate decaying over those 4000 (SmolVLA's
#     schedule otherwise always decays over 30000), evaluated every 500 to catch the peak:
#     success fell after the first evaluation in every unfrozen run (here, every 2000 steps:
#     29.7 %, 21.9 %, 18.8 %), so best/ (the best evaluation) is the policy to use
#   - 10 actions executed per chunk instead of 50 (no difference measured, slightly better)
#   - the policy's weights in fp32 (now always, policies/smolvla/modeling_smolvla.py)
# Everything else identical (joint actions, agentview, chunk 50, batch 64, vision-language model
# trained). On GPU 0. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_smolvla_joint_regularized.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_joint and lerobot/smolvla_base. Writes
# outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/ with one video per evaluation) and logs
# to the W&B project below. Other settings: see train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_smolvla_joint_agentview_chunk50_unfrozen_regularized" \
GPU=0 \
POLICY=smolvla \
PRETRAINED=lerobot/smolvla_base \
DATASET=pick_place_bottle \
ACTION_MODE=joint \
NORMALIZER=mean_std \
CAMERAS="agentview" \
CHUNK_SIZE=50 \
N_ACTION_STEPS=10 \
BATCH_SIZE=64 \
OPTIMIZER=policy \
GRAD_CLIP_NORM=10 \
EMA=1 \
RANDOM_SHIFT=6 \
STEPS=4000 \
EVAL_FREQ=500 \
EXTRA_ARGS="--policy_kwargs freeze_vision_encoder=false,train_expert_only=false,scheduler_decay_steps=4000" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
