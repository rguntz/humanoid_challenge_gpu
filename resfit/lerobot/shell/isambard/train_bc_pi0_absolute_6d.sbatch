#!/usr/bin/env bash
# Isambard-AI sbatch wrapper for resfit/lerobot/shell/train_bc_pi0_absolute_6d.sh.
# Follows the account/queueing conventions in ~/Documents/isambard_slurm_jobs.md
# and ~/Documents/humanoid_challenge_gpu_isambard.md: normal workq queue, 1 GH200
# GPU, uv-managed venv already synced on the login node (no per-job reinstall
# needed).
#
# Submit from the repo root:
#     sbatch resfit/lerobot/shell/isambard/train_bc_pi0_absolute_6d.sbatch
# Fast-start fallback if still PENDING after ~10 min (see the doc, section 1):
#     sbatch --partition=interactive --reservation=interactive --time=08:00:00 \
#       resfit/lerobot/shell/isambard/train_bc_pi0_absolute_6d.sbatch
#
#SBATCH --job-name=pi0_6d
#SBATCH --account=brics.u6vu
#SBATCH --qos=normal
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --output=resfit/lerobot/shell/isambard/logs/%x-%j.out
#SBATCH --error=resfit/lerobot/shell/isambard/logs/%x-%j.out

set -euo pipefail

# --mem=250G (above the --gpus=1 default share, ~112GB): confirmed needed via a
# real smoke test - PI0's PaliGemma-3B backbone + Gemma action-expert are built
# in fp32 first (no from_pretrained-style low-memory loading path), then
# converted to bf16 and loaded from a separately-read checkpoint file, which
# transiently spikes well past the default share during construction alone.

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

# Avoids a harmless but noisy warning from the dataloader workers forking
# after the PaliGemma tokenizer has already used its own internal threading.
export TOKENIZERS_PARALLELISM=false

# Only one GPU is allocated to this job; it's always device 0 from the job's
# own point of view regardless of the physical GPU Slurm picked.
GPU=0 bash resfit/lerobot/shell/train_bc_pi0_absolute_6d.sh
