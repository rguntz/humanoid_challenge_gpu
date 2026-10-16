#!/usr/bin/env bash
# Isambard-AI sbatch wrapper for resfit/pick_place/make_dataset.py: rebuilds
# pick_place_bottle_base.hdf5 and every LeRobot dataset (pick_place_bottle_*,
# pick_place_bottle_full_*) from the robomimic dataset, picking up the dataset's
# current --task default. Short, one-off job (not training) - runs on the
# interactive reservation directly (both --partition and --reservation are set
# below as #SBATCH directives, not just CLI flags, since Slurm requires
# --reservation to actually be passed for the interactive partition - a
# script-only --partition=interactive with no --reservation anywhere is refused
# with "Access/permission denied"). Caps walltime at 8h and has a fixed daily end
# time (see ~/Documents/isambard_slurm_jobs.md section 1) - fine here, well under
# the 2h requested below.
#
# Submit from the repo root:
#     sbatch resfit/pick_place/shell/isambard/make_dataset.sbatch
#
#SBATCH --job-name=make_dataset
#SBATCH --account=brics.u6vu
#SBATCH --qos=normal
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=resfit/pick_place/shell/isambard/logs/%x-%j.out
#SBATCH --error=resfit/pick_place/shell/isambard/logs/%x-%j.out

set -euo pipefail

# Non-interactive sbatch shells don't source ~/.bashrc automatically.
source ~/.bashrc

# Slurm copies the submitted script to a per-job spool file, so
# dirname "${BASH_SOURCE[0]}" does NOT point back at the repo - use the
# directory sbatch was invoked from instead.
cd "$SLURM_SUBMIT_DIR"

# Same TMPDIR gotcha as the other sbatch wrappers in this repo: an agent
# session's scratchpad path doesn't exist on the compute node this job lands on.
unset TMPDIR

uv run resfit/pick_place/make_dataset.py
