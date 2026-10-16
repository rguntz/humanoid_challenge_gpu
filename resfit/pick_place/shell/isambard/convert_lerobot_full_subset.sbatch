#!/usr/bin/env bash
# Isambard-AI sbatch wrapper to rebuild just the two LeRobot datasets currently needed
# (pick_place_bottle_full_absolute_6d for PI0, pick_place_bottle_full_joint for SmolVLA),
# skipping resfit/pick_place/make_dataset.py's hdf5-build/smoothing stage entirely: that
# stage already completed successfully in a previous run (pick_place_bottle_base.hdf5 on
# disk is current and complete), so this calls the LeRobot conversion step directly on it,
# restricted to --modes, instead of rebuilding everything from scratch. Much faster than
# the full make_dataset.sbatch job (skips the 6 other action-mode/version combinations).
#
# Runs on the interactive reservation directly (both --partition and --reservation are set
# below as #SBATCH directives, not just CLI flags, since Slurm requires --reservation to
# actually be passed for the interactive partition - a script-only --partition=interactive
# with no --reservation anywhere is refused with "Access/permission denied"). Caps walltime
# at 8h and has a fixed daily end time (see ~/Documents/isambard_slurm_jobs.md section 1) -
# fine here, this job is short.
#
# Submit from the repo root:
#     sbatch resfit/pick_place/shell/isambard/convert_lerobot_full_subset.sbatch
#
#SBATCH --job-name=lerobot_full_subset
#SBATCH --account=brics.u6vu
#SBATCH --qos=normal
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --output=resfit/pick_place/shell/isambard/logs/%x-%j.out
#SBATCH --error=resfit/pick_place/shell/isambard/logs/%x-%j.out

set -euo pipefail

source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
unset TMPDIR

uv run resfit/lerobot/dataset/convert_pick_place_to_lerobot.py \
    --full-episodes --name pick_place_bottle_full --modes absolute_6d joint
