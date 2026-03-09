#!/bin/bash
# =============================================================================
# Minimal smoke-test for grid-isambard.py — runs a tiny grid quickly
# to verify the environment, paths, and data pipeline before the real job.
#
# Submit with:  sbatch grid-isambard-test.sh
# =============================================================================

#SBATCH --job-name=tabicl-grid-test
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
export TAB_OUT_DIR="${WORK_DIR}/grid-results-test-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline

# -----------------------------------------------------------------------
# Low-resolution overrides — keeps the run short
#   GRANULARITY=2 → 2^3 * 3 * 4 * 2 * 2 = 384 combos (vs ~11k at 3)
#   N_EVAL=10     → 10 synthetic tables per evaluation (vs 200)
#   N_REAL=50     → 50 real tables cached (vs 5000)
# -----------------------------------------------------------------------
export TAB_GRANULARITY=2
export TAB_N_EVAL=10
export TAB_N_REAL=50

echo "======================================================================"
echo "TEST job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}  GRANULARITY=2  N_EVAL=10  N_REAL=50"
echo "======================================================================"

python grid-isambard.py

echo "======================================================================"
echo "TEST job ${SLURM_JOB_ID} finished at $(date)"
echo "======================================================================"
