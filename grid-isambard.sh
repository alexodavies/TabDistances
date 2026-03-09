#!/bin/bash
# =============================================================================
# Isambard HPC batch script for grid-isambard.py
#
# Single-node (default):
#   sbatch grid-isambard.sh
#
# Multi-node array (splits grid across 4 nodes):
#   sbatch --array=0-3 grid-isambard.sh
#
# Adjust CONDA_ENV and WORK_DIR before submitting.
# =============================================================================

#SBATCH --job-name=tabicl-grid
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00

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
export TAB_OUT_DIR="${WORK_DIR}/grid-results-${SLURM_JOB_ID}"

# Prevent NumPy / PyTorch / OpenBLAS / MKL from spawning their own thread
# pools inside ProcessPoolExecutor workers and loky subprocesses.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Wandb offline mode — remove this line if compute nodes have internet access.
# After the job, sync with: wandb sync grid-results-<JOB_ID>/wandb/offline-run-*
export WANDB_MODE=offline

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
python grid-isambard.py

echo "======================================================================"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
echo "======================================================================"
