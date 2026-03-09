#!/bin/bash
# =============================================================================
# Minimal smoke-test for grid-isambard.py — runs a tiny grid quickly
# to verify the environment, paths, and data pipeline before the real job.
#
# Submit with:  sbatch grid-isambard-test.sh
# =============================================================================

#SBATCH --job-name=tabicl-grid-test
#SBATCH --output=slurm-test-%A_%a.out
#SBATCH --error=slurm-test-%A_%a.err
#SBATCH --partition=milan
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=00:30:00

# -----------------------------------------------------------------------
# Edit these to match your setup (same as grid-isambard.sh):
# -----------------------------------------------------------------------
CONDA_ENV="tabdistance"
WORK_DIR="$PROJECTDIR/TabDistances"

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
source ~/miniforge3/bin/activate "$CONDA_ENV"
cd "$WORK_DIR"

export TAB_DATA_ROOT="."
export TAB_OUT_DIR="${WORK_DIR}/grid-results-test-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/shard_${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export WANDB_MODE=offline  # nodes have internet; comment out to log online

# -----------------------------------------------------------------------
# Smoke-test overrides
#   GRANULARITY=2 → 1,536 combos (vs 150k at 5)
#   N_EVAL=200    → synthetic tables per evaluation (production default)
#   N_REAL=200    → real tables cached
# -----------------------------------------------------------------------
export TAB_GRANULARITY=2
export TAB_N_EVAL=200
export TAB_N_REAL=200

echo "======================================================================"
echo "TEST job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}  GRANULARITY=2  N_EVAL=200  N_REAL=200"
echo "======================================================================"

python grid-isambard.py

echo "======================================================================"
echo "TEST job ${SLURM_JOB_ID} finished at $(date)"
echo "======================================================================"
