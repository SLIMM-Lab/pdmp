#!/usr/bin/env bash
# Run the K&O 3D beam test case with the RWM sampler.
# Execute from this directory or from the repo root.
#
# Options:
#   --no-ko     Skip the standard (no-discrepancy) RWM run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$REPO_ROOT/run_inference.py"

RUN_NO_KO=true
for arg in "$@"; do
    if [ "$arg" = "--no-ko" ]; then
        RUN_NO_KO=false
    fi
done

cd "$SCRIPT_DIR"

echo "=== Generating synthetic observations and RWM configs ==="
python generate_data.py

echo ""
echo "=== Running RWM sampler (K&O discrepancy) ==="
cd "$SCRIPT_DIR/rwm"
python "$RUNNER" --config config.yaml
cd "$SCRIPT_DIR"

if [ "$RUN_NO_KO" = true ]; then
    echo ""
    echo "=== Running RWM sampler (no K&O discrepancy) ==="
    cd "$SCRIPT_DIR/rwm_no_ko"
    python "$RUNNER" --config config.yaml
    cd "$SCRIPT_DIR"
fi

echo ""
echo "=== Analyzing results ==="
ANALYZE_ARGS=""
[ "$RUN_NO_KO" = false ] && ANALYZE_ARGS="$ANALYZE_ARGS --no-ko"
python analyze_results.py $ANALYZE_ARGS

echo ""
echo "Done. Plots written to $(pwd)/"
