#!/usr/bin/env bash
# Isambard-AI sbatch wrapper for
# resfit/rl_finetuning/shell/train_residual_td3_delta.sh.
# Follows the account/queueing conventions in ~/Documents/isambard_slurm_jobs.md
# and ~/Documents/humanoid_challenge_gpu_isambard.md: normal workq queue, 1 GH200
# GPU, uv-managed venv already synced on the login node (no per-job reinstall
# needed).
#
# Submit from the repo root:
#     sbatch resfit/rl_finetuning/shell/isambard/train_residual_td3_delta.sbatch
# Fast-start fallback if still PENDING after ~10 min (see the doc, section 1):
#     sbatch --partition=interactive --reservation=interactive --time=08:00:00 \
#       resfit/rl_finetuning/shell/isambard/train_residual_td3_delta.sbatch
#
#SBATCH --job-name=residual_td3_delta
#SBATCH --account=brics.u6vu
#SBATCH --qos=normal
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=resfit/rl_finetuning/shell/isambard/logs/%x-%j.out
#SBATCH --error=resfit/rl_finetuning/shell/isambard/logs/%x-%j.out

set -euo pipefail

# Non-interactive sbatch shells don't source ~/.bashrc automatically -
# WANDB_API_KEY lives there (see isambard_slurm_jobs.md section 2.5).
source ~/.bashrc

# Slurm copies the submitted script to a per-job spool file, so
# dirname "${BASH_SOURCE[0]}" does NOT point back at the repo - use the
# directory sbatch was invoked from instead.
cd "$SLURM_SUBMIT_DIR"

# sbatch propagates the submitting shell's environment by default. If that
# shell has TMPDIR pointing at a login-node-local scratch path (e.g. an agent
# session's scratchpad under /local/user/<uid>/...), it doesn't exist on the
# compute node this job lands on - torch.distributed's lazy init crashes with
# FileNotFoundError trying to mkdtemp() there. Unset it so Python falls back
# to this node's own /tmp.
unset TMPDIR

# Only one GPU is allocated to this job; it's always device 0 from the job's
# own point of view regardless of the physical GPU Slurm picked. The script
# itself defaults to GPU=1 (meant for running alongside sibling scripts on a
# shared multi-GPU dev box) but honours an inherited GPU env var, so override
# it here.
GPU=0 bash resfit/rl_finetuning/shell/train_residual_td3_delta.sh
