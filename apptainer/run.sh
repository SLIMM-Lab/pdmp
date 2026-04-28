#!/usr/bin/env bash
# Run a command inside the pdmp container.
# The pdmp source tree is bind-mounted at /pdmp.
# The container working directory mirrors the host CWD when it is inside the
# pdmp tree, so you can cd to any subdirectory and call scripts naturally.
#
# Usage (from project root):
#   ./apptainer/run.sh python run_inference.py --config examples/.../config.yaml
#   ./apptainer/run.sh pytest tests/
#   ./apptainer/run.sh bash
#
# Usage (from a config directory):
#   cd examples/inverse_problem/itz/itz_noise_low/separate/01/rwm
#   ../../../../../../../apptainer/run.sh python /pdmp/run_inference.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDMP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIF="${SCRIPT_DIR}/pdmp.sif"

if [[ ! -f "${SIF}" ]]; then
    echo "Container image not found: ${SIF}" >&2
    echo "Build it first with:  bash apptainer/build.sh" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $(basename "$0") <command> [args...]" >&2
    echo "Examples:" >&2
    echo "  $(basename "$0") python apptainer/test_jaxfem.py" >&2
    echo "  $(basename "$0") pytest tests/" >&2
    echo "  $(basename "$0") bash" >&2
    exit 1
fi

# Mirror host CWD into the container: if we are inside the pdmp tree, map to
# the corresponding /pdmp/... path; otherwise fall back to /pdmp.
HOST_CWD="$(pwd -P)"
PDMP_DIR_REAL="$(cd "${PDMP_DIR}" && pwd -P)"
if [[ "${HOST_CWD}" == "${PDMP_DIR_REAL}"* ]]; then
    CONTAINER_WORKDIR="/pdmp${HOST_CWD#${PDMP_DIR_REAL}}"
else
    CONTAINER_WORKDIR="/pdmp"
fi

exec apptainer exec \
    --bind "${PDMP_DIR_REAL}:/pdmp" \
    --workdir "${CONTAINER_WORKDIR}" \
    "${SIF}" \
    "$@"
