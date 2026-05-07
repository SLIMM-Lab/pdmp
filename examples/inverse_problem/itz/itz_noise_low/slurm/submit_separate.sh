#!/bin/bash
# Array job: runs the 10 separate geometry inference cases in parallel.
# Each task handles one geometry (01–10).
#
# Requires ~/.pdmp_env to set PDMP_DIR, e.g.:
#   export PDMP_DIR="/scratch/user/pdmp"

#SBATCH --job-name=noise-low-sep
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3968M
#SBATCH --time=3:59:00
#SBATCH --array=1-10
#SBATCH --output=separate_%A_%a.out
#SBATCH --error=separate_%A_%a.err
# SBATCH --account=research-CEG-3MD      # <-- fill in your account
#SBATCH --account=innovation      # <-- fill in your account

set -euo pipefail

source "${HOME}/.pdmp_env"
SIF="${PDMP_DIR}/apptainer/pdmp.sif"

CASE_DIR="/pdmp/$(realpath --relative-to="${PDMP_DIR}" "${SLURM_SUBMIT_DIR}/..")"
IDX="$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
CONFIG="${CASE_DIR}/separate/${IDX}/rwm/config.yaml"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: geometry ${IDX} ==="
echo "Node:   $(hostname)"
echo "Config: ${CONFIG}"
echo "SIF:    ${SIF}"
date

apptainer exec \
    --bind "${PDMP_DIR}:/pdmp" \
    --env "JAX_PLATFORM_NAME=cpu" \
    --env "OMP_NUM_THREADS=2" \
    "${SIF}" \
    python /pdmp/run_inference.py --config "${CONFIG}"

echo "=== Done ==="
date
