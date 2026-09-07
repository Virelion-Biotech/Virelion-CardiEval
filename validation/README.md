# Local validation kit for CardiEval

This folder gives you everything needed to validate CardiEval **on your own machine** without relying on external cardiac datasets or fabricated model predictions.

## What you get

| File | Purpose |
|------|---------|
| `generate_synthetic_benchmark.py` | Builds a reproducible synthetic binary-classification benchmark (default n=500) with two models + sex subgroups |
| `run_local_validation.py` | Runs the full evaluate → report → bundle → (optional) leaderboard path |
| `data/` | Created by the generator (manifest, task, submissions) |
| `outputs/` | Created by the runner (reports, bundles, summary) |

The synthetic data is **not clinical**. It only exercises contracts, metrics, CIs, subgroups, and publication plumbing.

## Quick start (from repo root)

```bash
# 1. Install the package
pip install -e ".[dev]"

# 2. Generate synthetic benchmark
python validation/generate_synthetic_benchmark.py
# optional: python validation/generate_synthetic_benchmark.py --n 800 --seed 7

# 3. Run evaluation for both models
python validation/run_local_validation.py

# 4. (Optional) also run unit tests
python validation/run_local_validation.py --pytest
```

Expected outcome:

- `validation/outputs/report_baseline.json` and `report_strong.json`
- matching bundles under `validation/outputs/bundles/`
- `validation/outputs/validation_summary.json`
- strong model AUROC higher than baseline (by construction)

## Use your own predictions instead

Replace the generated files with your own, keeping the same layout:

```text
validation/data/
  manifest.json          # BenchmarkManifest
  task.json              # BenchmarkTask (IDs/version must match manifest)
  submissions/
    your-model.jsonl     # one PredictionRecord per line
```

Then:

```bash
python validation/run_local_validation.py --data-dir validation/data --out-dir validation/outputs
```

Each JSONL line must look like:

```json
{"sample_id":"sample-0001","y_true":0,"y_pred":0,"score":0.12,"subgroup":"male"}
```

- `sample_id` set must match the manifest **exactly**
- `score` is required for AUROC / AUPRC / Brier / ECE
- `subgroup` is optional but enables robustness reporting

## Requirements

- Python ≥ 3.10
- `numpy`, `pydantic`, `scipy`, `scikit-learn` (installed via `pip install -e .`)

No internet access is required after install.

## Notes

- Generator uses only NumPy so it can run even before the full package is installed.
- Runner imports `cardieval` from `src/` automatically when you run from a source checkout.
- For real public datasets (PTB-XL, PhysioNet Challenges, etc.) you still need a trained model that emits predictions in the JSONL contract above; this kit does not train models.
