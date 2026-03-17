#!/bin/bash
# =============================================================================
# Slurm script for grid-combine.py — runs automatically after grid-isambard.sh
# via a dependency set by grid-submit.sh.  Do not submit this directly.
# =============================================================================

#SBATCH --job-name=tabicl-combine
#SBATCH --output=logs/slurm-combine-%j.out
#SBATCH --error=logs/slurm-combine-%j.err
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

CONDA_ENV="tabdistance"
WORK_DIR="$PROJECTDIR/TabDistances"

source ~/miniforge3/bin/activate "$CONDA_ENV"
cd "$WORK_DIR"

# ARRAY_JOB_ID and RESULTS_PREFIX are passed via --export from grid-submit.sh
RESULTS_DIR="${WORK_DIR}/${RESULTS_PREFIX:-grid-results}-${ARRAY_JOB_ID}"

echo "======================================================================"
echo "Combine job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "ARRAY_JOB_ID : ${ARRAY_JOB_ID}"
echo "RESULTS_DIR  : ${RESULTS_DIR}"
echo "======================================================================"

python grid-combine.py "$RESULTS_DIR"

echo "======================================================================"
echo "Combine job ${SLURM_JOB_ID} finished at $(date)"
echo "======================================================================"
