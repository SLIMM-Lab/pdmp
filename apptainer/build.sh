#!/usr/bin/env bash
# Build the pdmp Apptainer container image.
# Requires root or user-namespace support (apptainer build --fakeroot).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/pdmp.sif"
DEF="${SCRIPT_DIR}/pdmp.def"

echo "=== Building pdmp Apptainer container ==="
echo "    definition:  ${DEF}"
echo "    output:      ${SIF}"
echo ""

# Use --fakeroot if not running as root
if [[ $(id -u) -eq 0 ]]; then
    apptainer build "${SIF}" "${DEF}"
else
    apptainer build --fakeroot "${SIF}" "${DEF}"
fi

echo ""
echo "=== Build complete: ${SIF} ==="
