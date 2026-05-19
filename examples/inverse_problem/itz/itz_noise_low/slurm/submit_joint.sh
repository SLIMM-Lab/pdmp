#!/bin/bash
# Single job: joint inference over all 10 geometries simultaneously.
# Uses more memory than a separate case (larger observation vector).
#
# Requires ~/.pdmp_env to set PDMP_DIR, e.g.:
#   export PDMP_DIR="/scratch/user/pdmp"

#SBATCH --job-name=noise-low-joint
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3968M
#SBATCH --time=3:59:00
#SBATCH --output=joint_%j.out
#SBATCH --error=joint_%j.err
#SBATCH --account=research-CEG-3MD      # <-- fill in your account
# SBATCH --account=innovation      # <-- fill in your account

set -euo pipefail

source "${HOME}/.pdmp_env"
SIF="${PDMP_DIR}/apptainer/pdmp.sif"

CASE_DIR="/pdmp/$(realpath --relative-to="${PDMP_DIR}" "${SLURM_SUBMIT_DIR}/..")"
CONFIG="${CASE_DIR}/joint/rwm/config.yaml"

echo "=== Joint inference ==="
echo "Node:   $(hostname)"
echo "Config: ${CONFIG}"
echo "SIF:    ${SIF}"
date

apptainer exec \
    --bind "${PDMP_DIR}:/pdmp" \
    --env "JAX_PLATFORM_NAME=cpu" \
    --env "OMP_NUM_THREADS=4" \
    "${SIF}" \
    python /pdmp/run_inference.py --config "${CONFIG}"

echo "=== Done ==="
date
