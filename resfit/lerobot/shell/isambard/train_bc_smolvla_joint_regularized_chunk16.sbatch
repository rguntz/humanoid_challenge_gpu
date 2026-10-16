#!/usr/bin/env bash
# Isambard-AI sbatch wrapper for
# resfit/lerobot/shell/train_bc_smolvla_joint_regularized_chunk16.sh.
# Follows the account/queueing conventions in ~/Documents/isambard_slurm_jobs.md
# and ~/Documents/humanoid_challenge_gpu_isambard.md: normal workq queue, 1 GH200
# GPU, uv-managed venv already synced on the login node (no per-job reinstall
# needed). No --mem override, same reasoning as the chunk50 sbatch: SmolVLA's
# 450M parameters are far smaller than PI0's PaliGemma-3B backbone.
#
# Submit from the repo root:
#     sbatch resfit/lerobot/shell/isambard/train_bc_smolvla_joint_regularized_chunk16.sbatch
# Fast-start fallback if still PENDING after ~10 min (see the doc, section 1):
#     sbatch --partition=interactive --reservation=interactive --time=08:00:00 \
#       resfit/lerobot/shell/isambard/train_bc_smolvla_joint_regularized_chunk16.sbatch
#
#SBATCH --job-name=smolvla_joint_reg_c16
#SBATCH --account=brics.u6vu
#SBATCH --qos=normal
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=resfit/lerobot/shell/isambard/logs/%x-%j.out
#SBATCH --error=resfit/lerobot/shell/isambard/logs/%x-%j.out

set -euo pipefail

# Non-interactive sbatch shells don't source ~/.bashrc automatically -
# WANDB_API_KEY lives there (see isambard_slurm_jobs.md section 2.5).
source ~/.bashrc

# Slurm copies the submitted script to a per-job spool file, so
# dirname "${BASH_SOURCE[0]}" does NOT point back at the repo - use the
# directory sbatch was invoked from instead. (Submitting the plain .sh script
# directly to sbatch fails on exactly this: its own dirname "${BASH_SOURCE[0]}"
# resolves to the spool copy, not the repo, so it can't find train_bc.sh.)
cd "$SLURM_SUBMIT_DIR"

# Same TMPDIR gotcha as the other sbatch wrappers in this repo: an agent
# session's scratchpad path doesn't exist on the compute node this job lands on.
unset TMPDIR

export TOKENIZERS_PARALLELISM=false

# The script's own GPU=0 is already correct here: only one GPU is allocated to
# this job, always device 0 from the job's own point of view.
bash resfit/lerobot/shell/train_bc_smolvla_joint_regularized_chunk16.sh
