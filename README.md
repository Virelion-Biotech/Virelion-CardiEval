# Virelion-CardiEval

CardiEval is an independent evaluation library for cardiac machine-learning models. It evaluates submitted predictions against versioned benchmark packages without requiring access to model internals.

## What it contains

- Benchmark and task contract validation.
- Exact sample-set validation with optional evaluator-controlled labels/subgroups.
- Classification, diagnostic, calibration, regression, and ranking metrics.
- Bootstrap confidence intervals and paired model comparisons.
- Permutation/Wilcoxon testing and multiple-testing correction.
- Subgroup and robustness analysis.
- Deterministic leaderboard snapshots and historical comparisons.
- Release-artifact verification and SHA-256 integrity records.
- End-to-end evaluation run manifests.
- Versioned machine-readable evaluation contracts.
- A strict synthetic validation kit and CI release checks.

## Installation

```bash
pip install 'cardieval[test]'
pytest
```

For development:

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

## Usage

```bash
cardieval --manifest examples/benchmark_manifest.json \
  --submission examples/submission.jsonl \
  --model-id demo-model \
  --task-file examples/demo_task.json \
  --output cardiEval-report.json
```

For a protected end-to-end run:

```bash
cardieval run --package benchmark-package.json \
  --root ./benchmark-release \
  --submission submission.jsonl \
  --model-id my-model \
  --task-id binary-challenge-detection \
  --report-output outputs/report.json \
  --bundle-output outputs/bundle.json \
  --run-output outputs/run.json
```

## Independent evaluation

The legacy JSONL contract includes y_true for self-contained demonstrations. A protected benchmark should keep reference labels outside the submission and declare them in evaluator-controlled benchmark metadata, then set:

```json
{
  "requires_authoritative_labels": true
}
```

In that mode CardiEval ignores submission-supplied y_true values and uses the benchmark's authoritative labels. Authoritative subgroup assignments can be handled the same way. Keep those benchmark artifacts private until the evaluation window closes.

## Validation

Run the complete local validation kit:

```bash
bash validation/run_all.sh
```

CI additionally runs the full Python 3.10–3.12 / NumPy 1.x–2.x test matrix, linting, dependency checks, package builds, the synthetic end-to-end publication path, and pip-audit.

A successful software test or evaluation run is not evidence of clinical validity, safety, effectiveness, or regulatory acceptance.

## Inputs and outputs

**Inputs:** a versioned benchmark package, task definition, prediction submission, model identifier, and optional evaluation configuration.

**Outputs:** validated evaluation reports, metric/statistical results, confidence intervals, subgroup/robustness summaries, submission bundles, run manifests, leaderboard snapshots, publication comparisons, and integrity records.

## Limitations

CardiEval evaluates serialized model outputs; it is not a raw ECG, imaging, or signal preprocessing pipeline. Real cardiac validation requires a frozen benchmark package with independently controlled labels, explicit split policy, subject-level separation where applicable, and an independently produced model submission. Synthetic validation only establishes software/contract behavior.

Statistical significance does not establish clinical significance or causal validity. Hashes detect accidental substitution or tampering but do not provide authenticity; digital signing, key management, access control, and transport security remain deployment concerns.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
