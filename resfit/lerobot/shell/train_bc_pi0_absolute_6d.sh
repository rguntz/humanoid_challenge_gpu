#!/usr/bin/env bash
# PI0 (LeRobot's VLA: PaliGemma VLM backbone + flow-matching action expert,
# resfit/lerobot/policies/pi0/), started from the pretrained lerobot/pi0 and fine-tuned on
# absolute end-effector actions with 6D rotations, matching
# train_bc_flow_matching_absolute_6d.sh (DATASET=pick_place_bottle_full,
# ACTION_MODE=absolute_6d, CAMERAS=agentview, same EVAL_LAYOUTS=fresh/EVAL_EPISODES=100/
# EVAL_NUM_ENVS=25/SAVE_EVERY_EVAL=1) for a comparison against that run (43.5 % on 200 fresh
# layouts) - same action/observation space, same eval methodology, EXCEPT the action chunking
# (see below): same action/observation space, same eval methodology.
#
# CHUNK_SIZE=16, N_ACTION_STEPS=16 (NOT 8, unlike the flow_matching baseline's Ta=16/execution-
# horizon=8 split) - explicit choice, to stay canonical with how PI0 is actually designed/
# supported: LeRobot's own PI0FlowMatching.embed_suffix() silently assumes n_action_steps ==
# chunk_size everywhere (its action-token attention mask is built from config.n_action_steps,
# while the actual embedded action-chunk length is chunk_size - the two only coincide when they're
# set equal) - this is also what PI0Config's own defaults (50/50) and SmolVLA's own launch
# script use, i.e. the setting Physical Intelligence's own checkpoint was actually trained/used
# with. Setting them unequal (to mirror the flow_matching baseline's own held-out-tail chunking
# exactly) hits that assumption and crashes (confirmed directly: RuntimeError, mismatched
# attention-mask tensor sizes, exactly N_ACTION_STEPS short) - fixable with a one-line patch to
# the installed lerobot package, but deliberately NOT done here (explicit choice): every
# predicted action in the chunk is executed, matching PI0's own validated usage, rather than
# patching third-party code to replicate a chunking convention PI0 was never used with.
#
# Other deliberate differences from the flow_matching baseline, all architectural constraints
# or explicit choices (see resfit/lerobot/policies/pi0/modeling_pi0.py's own docstring):
#   - N_OBS_STEPS is NOT set (PI0Config.__post_init__ hard-rejects anything but 1 observation
#     frame - "Multiple observation steps not handled yet" - so this can't match the flow_matching
#     run's N_OBS_STEPS=2).
#   - PRECISION is NOT set (stays at train_bc.sh's own bf16 default): LeRobot's own PI0 already
#     converts the PaliGemma + action-expert weights to bf16 in its own __init__
#     (to_bfloat16_like_physical_intelligence), matching how Physical Intelligence's own
#     checkpoint was trained/released - this script does not re-derive SmolVLA's own all-fp32
#     choice (SmolVLA's own modeling_smolvla.py docstring explains why THAT policy needed it);
#     the pretrained checkpoint's own precision is the reference point here, left untouched.
#   - NORMALIZER stays mean_std (LeRobot's own, PI0Config's default for state/action) and
#     IMAGE_NORM stays mean_std too (a no-op here: PI0Config's own VISUAL normalization is
#     IDENTITY by default - PI0's prepare_images already rescales images itself, same as
#     SmolVLA) - the flow_matching run's per_dim_range/minus_one_one choices were specific to
#     matching the original JAX pipeline's own Diffusion-Policy-style encoder, not applicable
#     to PI0's own architecture.
#   - OPTIMIZER=policy (PI0Config's own AdamW preset, lr=2.5e-5, betas=(0.9,0.95), cosine decay
#     with warmup) and EMA=0 (PI0Config has no EMA support) - same choices SmolVLA's own
#     train_bc_smolvla_absolute_6d.sh already made for the same reason (a VLM-backbone policy
#     uses its own pretrained optimizer recipe, not the legacy JAX-matching one).
#   - GRAD_CLIP_NORM=10: SmolVLA's own choice, carried over as the closest architectural
#     precedent (both are large pretrained-backbone VLA policies).
#   - BATCH_SIZE=16, NOT 64 (SmolVLA's own choice, tried first): confirmed via a real run -
#     PI0's own VRAM footprint (optimizer state for ~3B trainable params, tens of GB on its own,
#     independent of batch size, on top of PaliGemma+expert's own activations) left only ~0.5GB
#     free on a 95GB GH200 at batch 64 mid-forward-pass, and a real production run OOMed
#     (torch.OutOfMemoryError, "Tried to allocate 608.00 MiB ... 86.43 GiB memory in use").
#     16 leaves real headroom - SmolVLA's own smaller backbone apparently fits at 64 where
#     PI0's larger PaliGemma-3B does not.
#
# On GPU 2. Run from anywhere:
#     bash resfit/lerobot/shell/train_bc_pi0_absolute_6d.sh
# Reads videos/task/dataset_sim/lerobot/pick_place_bottle_full_absolute_6d (same dataset the
# flow_matching baseline reads). Writes outputs/resfit_bc/<RUN_NAME>/ (latest/, best/, evals/
# with one video per evaluation) and logs to the W&B project below. Other settings: see
# train_bc.sh.
WANDB_PROJECT=pick-place-bc \
RUN_NAME="$(date +%Y-%m-%d_%H-%M-%S)_pi0_6d_agentview_Ta16_h16_policyopt" \
GPU="${GPU:-2}" \
POLICY=pi0 \
PRETRAINED=lerobot/pi0 \
DATASET=pick_place_bottle_full \
ACTION_MODE=absolute_6d \
NORMALIZER=mean_std \
CAMERAS="agentview" \
CHUNK_SIZE=16 \
N_ACTION_STEPS=16 \
BATCH_SIZE=16 \
OPTIMIZER=policy \
GRAD_CLIP_NORM=10 \
EMA=0 \
STEPS="${STEPS:-100000}" \
EVAL_LAYOUTS=fresh \
EVAL_EPISODES=100 \
EVAL_NUM_ENVS=25 \
SAVE_EVERY_EVAL=1 \
bash "$(dirname "${BASH_SOURCE[0]}")/train_bc.sh"
