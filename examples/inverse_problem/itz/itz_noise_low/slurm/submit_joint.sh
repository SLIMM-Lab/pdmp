#!/bin/bash
# Single job: joint inference over all 10 geometries simultaneously.
# Uses more memory than a separate case (larger observation vector).
#
# Submit from the pdmp project root:
#   sbatch examples/inverse_problem/itz/itz_noise_low/slurm/submit_joint.sh

#SBATCH --job-name=itz-joint
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2:00:00
#SBATCH --output=joint_%j.out
#SBATCH --error=joint_%j.err
#SBATCH --account=research-CEG-3MD      # <-- fill in your account

set -euo pipefail

# SLURM_SUBMIT_DIR is the directory sbatch was called from.
# Always submit from the pdmp project root (see header comment).
PDMP_DIR="${SLURM_SUBMIT_DIR}"
SIF="${PDMP_DIR}/apptainer/pdmp.sif"

CONFIG="/pdmp/examples/inverse_problem/itz/itz_noise_low/joint/rwm/config.yaml"

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
