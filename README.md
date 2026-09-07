# Virelion-CardiEval

CardiEval is an independent evaluation library for cardiac machine-learning models. It evaluates submitted predictions against versioned benchmark packages without requiring access to model internals.

## What it contains

- Benchmark and task contract validation.
- Exact sample-set validation.
- Classification, diagnostic, calibration, regression, and ranking metrics.
- Bootstrap confidence intervals and paired model comparisons.
- Permutation/Wilcoxon testing and multiple-testing correction.
- Subgroup and robustness analysis.
- Deterministic leaderboard snapshots and historical comparisons.
- Release-artifact verification and SHA-256 integrity records.
- End-to-end evaluation run manifests.
- Versioned machine-readable evaluation contracts.

## Installation

```bash
pip install -e '.[test]'
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

## Inputs and outputs

**Inputs:** a versioned benchmark package, task definition, prediction submission, model identifier, and optional evaluation configuration.

**Outputs:** validated evaluation reports, metric/statistical results, confidence intervals, subgroup/robustness summaries, submission bundles, run manifests, leaderboard snapshots, and integrity records.

Evaluation rejects incompatible benchmark/task identities, duplicate or unknown samples, and invalid split assignments.

## Validation

Software tests cover contract validation, evaluation behavior, statistical functions, artifact integrity, and reproducibility. Evaluation also verifies benchmark identity, sample membership, and split compatibility before scoring.

A successful software test or evaluation run is not evidence of clinical validity, safety, effectiveness, or regulatory acceptance.

## Limitations

Results depend on benchmark construction, data quality, subgroup size, split policy, and the chosen metrics. Statistical significance does not establish clinical significance or causal validity. External key management and cryptographic signing are outside the evaluator.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
