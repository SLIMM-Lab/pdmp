#!/bin/bash
# Array job: runs the 10 separate geometry inference cases in parallel.
# Each task handles one geometry (01–10).
#
# Submit from the pdmp project root:
#   sbatch examples/inverse_problem/itz/itz_noise_low/slurm/submit_separate.sh
#
# Or from this directory:
#   sbatch submit_separate.sh

#SBATCH --job-name=itz-sep
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=6:00:00
#SBATCH --array=1-10
#SBATCH --output=separate_%A_%a.out
#SBATCH --error=separate_%A_%a.err
# SBATCH --account=research-CEG-3MD      # <-- fill in your account
#SBATCH --account=innovation      # <-- fill in your account

set -euo pipefail

# SLURM_SUBMIT_DIR is the directory sbatch was called from.
# Always submit from the pdmp project root (see header comment).
PDMP_DIR="${SLURM_SUBMIT_DIR}"
SIF="${PDMP_DIR}/apptainer/pdmp.sif"

IDX="$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
CONFIG="/pdmp/examples/inverse_problem/itz/itz_noise_low/separate/${IDX}/rwm/config.yaml"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: geometry ${IDX} ==="
echo "Node:   $(hostname)"
echo "Config: ${CONFIG}"
echo "SIF:    ${SIF}"
date

apptainer exec \
    --bind "${PDMP_DIR}:/pdmp" \
    --env "JAX_PLATFORM_NAME=cpu" \
    --env "OMP_NUM_THREADS=8" \
    "${SIF}" \
    python /pdmp/run_inference.py --config "${CONFIG}"

echo "=== Done ==="
date
