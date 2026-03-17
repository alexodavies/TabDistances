#!/bin/bash
# =============================================================================
# Isambard HPC batch script for grid-isambard.py
#
# Single-node:
#   sbatch grid-isambard.sh
#
# Multi-node array — recommended for G=5 (150k combos, 7 idle Milan nodes):
#   sbatch --array=0-6 grid-isambard.sh   →  7 nodes, ~1.5 hrs wall-clock
#   sbatch --array=0-3 grid-isambard.sh   →  4 nodes, ~2.5 hrs wall-clock
#
# Adjust CONDA_ENV and WORK_DIR before submitting.
# =============================================================================

#SBATCH --job-name=tabicl-grid
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --partition=milan
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

# -----------------------------------------------------------------------
# Edit these two variables to match your Isambard setup:
# -----------------------------------------------------------------------
CONDA_ENV="tabdistance"
WORK_DIR="$PROJECTDIR/TabDistances"

# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------
source ~/miniforge3/bin/activate "$CONDA_ENV"
cd "$WORK_DIR"

# Root containing tables-t4/parquet and tables-tabfm/parquet.
# Use an absolute path on /work/<project>/<user>/ for best I/O performance.
export TAB_DATA_ROOT="."

# Output directory — one subfolder per job to avoid collisions.
export TAB_OUT_DIR="${WORK_DIR}/grid-results-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}/shard_${SLURM_ARRAY_TASK_ID:-0}"

# Prevent NumPy / PyTorch / OpenBLAS / MKL from spawning their own thread
# pools inside ProcessPoolExecutor workers and loky subprocesses.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Wandb offline mode — remove this line if compute nodes have internet access.
# After the job, sync with: wandb sync grid-results-<JOB_ID>/wandb/offline-run-*
# export WANDB_MODE=offline  # uncomment if nodes lose internet access

# -----------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------
echo "======================================================================"
echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "CPUs requested : ${SLURM_CPUS_PER_TASK}"
echo "Working dir    : $(pwd)"
echo "Conda env      : ${CONDA_ENV}"
echo "TAB_DATA_ROOT  : ${TAB_DATA_ROOT}"
echo "TAB_OUT_DIR    : ${TAB_OUT_DIR}"
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Array task     : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
fi
echo "======================================================================"

# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
python grid/grid-isambard.py

echo "======================================================================"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "======================================================================"
