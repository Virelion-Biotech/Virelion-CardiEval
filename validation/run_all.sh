#!/usr/bin/env bash
# One-shot local validation helper. Run from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Installing package (editable + dev extras)..."
pip install -e ".[dev]" -q

echo "==> Generating synthetic benchmark..."
python validation/generate_synthetic_benchmark.py --n 500 --seed 42

echo "==> Running CardiEval validation..."
python validation/run_local_validation.py --pytest

echo "==> Done. See validation/outputs/"
