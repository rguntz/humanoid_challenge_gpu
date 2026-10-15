#!/usr/bin/env bash
# Isambard-AI sbatch wrapper for policy/shell/train_mip_tanh_state_drop_term.sh.
# Follows the account/queueing conventions in ~/Documents/isambard_slurm_jobs.md
# (xqcfd/diffusion_policy jobs): normal workq queue, 1 GH200 GPU, uv-managed
# venv already synced on the login node (no per-job reinstall needed).
#
# Resumes straight into online fine-tuning from an already-completed
# checkpoint (see the .sh script's own header) - no offline pretraining to
# redo, so this job's own wall-clock is just the 200_000-step online phase.
#
# Submit from the repo root:
#     sbatch policy/shell/isambard/train_mip_tanh_state_drop_term.sbatch
# Fast-start fallback if still PENDING after ~10 min (see the doc, section 1):
#     sbatch --partition=interactive --reservation=interactive --time=08:00:00 \
#       policy/shell/isambard/train_mip_tanh_state_drop_term.sbatch
#
#SBATCH --job-name=mip_tanh_state_dropterm
#SBATCH --account=brics.u6vu
#SBATCH --qos=normal
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=policy/shell/isambard/logs/%x-%j.out
#SBATCH --error=policy/shell/isambard/logs/%x-%j.out

set -euo pipefail

# Non-interactive sbatch shells don't source ~/.bashrc automatically -
# WANDB_API_KEY lives there (see doc section 2.5).
source ~/.bashrc

# Slurm copies the submitted script to a per-job spool file, so
# dirname "${BASH_SOURCE[0]}" does NOT point back at the repo - use the
# directory sbatch was invoked from instead.
cd "$SLURM_SUBMIT_DIR"

# Only one GPU is allocated to this job; it's always device 0 from the job's
# own point of view regardless of the physical GPU Slurm picked.
GPU=0 bash policy/shell/train_mip_tanh_state_drop_term.sh
