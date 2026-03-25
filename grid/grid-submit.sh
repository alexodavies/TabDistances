#!/bin/bash
# =============================================================================
# Submit the grid search array job and queue the combine step automatically.
#
# Usage:
#   bash grid-submit.sh           # uses grid-isambard.sh defaults
#   bash grid-submit.sh --test    # smoke-test version (grid-isambard-test.sh)
# =============================================================================

if [[ "${1}" == "--test" ]]; then
    GRID_SCRIPT="grid/grid-isambard-test.sh"
    RESULTS_PREFIX="grid-results-test"
else
    GRID_SCRIPT="grid/grid-isambard.sh"
    RESULTS_PREFIX="grid-results"
fi

# Slurm writes logs before the job body runs, so the directory must exist first
mkdir -p logs

# Submit the array job and capture the job ID
ARRAY_JOB_ID=$(sbatch --parsable "$GRID_SCRIPT")
if [[ -z "$ARRAY_JOB_ID" ]]; then
    echo "ERROR: sbatch failed for $GRID_SCRIPT" >&2
    exit 1
fi
echo "Grid array job submitted : ${ARRAY_JOB_ID}  (${GRID_SCRIPT})"

# Queue the combine job to run only after ALL array tasks succeed
COMBINE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export=ALL,ARRAY_JOB_ID=${ARRAY_JOB_ID},RESULTS_PREFIX=${RESULTS_PREFIX} \
    grid/grid-combine.sh)
if [[ -z "$COMBINE_JOB_ID" ]]; then
    echo "ERROR: sbatch failed for grid-combine.sh" >&2
    exit 1
fi
echo "Combine job queued       : ${COMBINE_JOB_ID}  (runs after ${ARRAY_JOB_ID})"
echo
echo "Monitor with:  squeue --me"
echo "Grid logs   :  tail -f logs/slurm-${ARRAY_JOB_ID}_0.out"
echo "Combine log :  tail -f logs/slurm-combine-${COMBINE_JOB_ID}.out"
