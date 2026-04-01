#!/usr/bin/env bash
# Run the K&O piecewise-constant test case.
# Execute from the repo root or from this directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$SCRIPT_DIR"

echo "=== Generating synthetic observations ==="
python generate_data.py

echo ""
echo "=== Running K&O calibration (RWM sampler) ==="
python "$REPO_ROOT/run_inference.py" --config config.yaml

echo ""
echo "=== Done. Results written to ./results/ ==="
