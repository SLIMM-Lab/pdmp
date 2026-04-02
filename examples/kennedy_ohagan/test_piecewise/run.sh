#!/usr/bin/env bash
# Run the K&O piecewise-constant test case with both RWM and BPS samplers.
# Execute from this directory or from the repo root.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$REPO_ROOT/run_inference.py"

cd "$SCRIPT_DIR"

echo "=== Generating synthetic observations and BPS config ==="
python generate_data.py

echo ""
echo "=== Running RWM sampler ==="
cd "$SCRIPT_DIR/rwm"
python "$RUNNER" --config config.yaml
cd "$SCRIPT_DIR"

echo ""
echo "=== Running BPS sampler (affine-whitened) ==="
cd "$SCRIPT_DIR/bps"
python "$RUNNER" --config config.yaml
cd "$SCRIPT_DIR"

echo ""
echo "=== Analyzing results ==="
python analyze_results.py

echo ""
echo "Done. Plots written to $(pwd)/"
