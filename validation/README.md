# Local validation kit for CardiEval

This folder provides a reproducible software-level validation of CardiEval without relying on external cardiac datasets or fabricated model predictions.

## What it exercises

| File | Purpose |
|------|---------|
| generate_synthetic_benchmark.py | Builds a reproducible synthetic binary-classification benchmark with evaluator-controlled labels/subgroups, two model submissions, and a complete BenchmarkPackage |
| run_local_validation.py | Runs the package-verified evaluation pipeline, report/bundle/run-manifest generation, integrity checks, and publication path |
| data/ | Generated manifest, task, package, and protected submissions |
| outputs/ | Generated reports, bundles, run manifests, leaderboard snapshot, and validation summary |

The synthetic data is not clinical. It only exercises contracts, metrics, confidence intervals, subgroup handling, integrity checks, and publication plumbing.

## Quick start

From the repository root:

    pip install -e ".[dev]"
    python validation/generate_synthetic_benchmark.py
    python validation/run_local_validation.py --pytest

Or run the one-shot helper:

    bash validation/run_all.sh

Expected outputs include:

- validation/data/package.json
- validation/outputs/report_baseline.json and report_strong.json
- validation/outputs/bundles/
- validation/outputs/run_baseline.json and run_strong.json
- validation/outputs/leaderboard.json
- validation/outputs/validation_summary.json

The synthetic benchmark is intentionally generated with evaluator-controlled authoritative_labels. The model submissions contain no y_true, so the validation path exercises the protected-label contract.

## Submission contract

A protected submission is JSONL with one prediction per benchmark sample:

    {"sample_id":"sample-0001","y_pred":0,"score":0.12}

The sample_id set must match the benchmark manifest exactly. For a task that requires evaluator-controlled labels, y_true may be omitted from the submission.

For self-contained/demo evaluation, submissions may include y_true when the task does not require authoritative benchmark labels.

score is required for score-based metrics such as AUROC, AUPRC, Brier score, and ECE. subgroup is optional for unprotected workflows; protected benchmarks can keep subgroup assignments in the evaluator-controlled manifest.

## Requirements

- Python >= 3.10
- NumPy, Pydantic, SciPy, scikit-learn
- pytest, Ruff, build, pip-audit, and jsonschema for the developer validation suite

No internet access is required after dependencies are installed.

## Real-dataset boundary

PTB-XL, PhysioNet Challenge data, MIMIC-IV-ECG, and similar datasets can be wrapped by CardiBench, but this repository does not fabricate or bundle trained-model predictions for them. Real benchmark validation therefore requires an actual frozen benchmark package and independently generated submissions.

A successful synthetic/software validation run does not establish clinical validity, safety, effectiveness, or regulatory acceptance.
