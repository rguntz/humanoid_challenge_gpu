#!/usr/bin/env bash
#SBATCH --job-name=mip_tanh_vision_dropterm_simplifiedinit
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
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
GPU=0 bash policy/shell/train_mip_tanh_vision_drop_simplified.sh
