#!/usr/bin/env bash
# Run the K&O piecewise-constant test case with both RWM and BPS samplers.
# Execute from this directory or from the repo root.
#
# Options:
#   --no-bps    Skip the BPS sampler and BPS analysis
#   --no-ko     Skip the standard (no-discrepancy) RWM run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$REPO_ROOT/run_inference.py"

RUN_BPS=true
RUN_NO_KO=true
for arg in "$@"; do
    if [ "$arg" = "--no-bps" ]; then
        RUN_BPS=false
    fi
    if [ "$arg" = "--no-ko" ]; then
        RUN_NO_KO=false
    fi
done

cd "$SCRIPT_DIR"

echo "=== Generating synthetic observations and BPS config ==="
python generate_data.py

echo ""
echo "=== Running RWM sampler ==="
cd "$SCRIPT_DIR/rwm"
python "$RUNNER" --config config.yaml
cd "$SCRIPT_DIR"

if [ "$RUN_BPS" = true ]; then
    echo ""
    echo "=== Running BPS sampler (affine-whitened) ==="
    cd "$SCRIPT_DIR/bps"
    python "$RUNNER" --config config.yaml
    cd "$SCRIPT_DIR"
fi

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
[ "$RUN_BPS" = false ] && ANALYZE_ARGS="$ANALYZE_ARGS --no-bps"
[ "$RUN_NO_KO" = false ] && ANALYZE_ARGS="$ANALYZE_ARGS --no-ko"
python analyze_results.py $ANALYZE_ARGS

echo ""
echo "Done. Plots written to $(pwd)/"
