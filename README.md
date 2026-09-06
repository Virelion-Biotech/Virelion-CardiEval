# Virelion-CardiEval

CardiEval is an independent evaluation library for cardiac machine-learning models. It evaluates submitted predictions against versioned benchmark packages without requiring access to model internals.

## Scope

CardiEval provides:

- benchmark and task contract validation;
- exact sample-set validation;
- classification, diagnostic, calibration, regression, and ranking metrics;
- bootstrap confidence intervals and paired model comparisons;
- permutation/Wilcoxon testing and multiple-testing correction;
- subgroup and robustness analysis;
- deterministic leaderboard snapshots and historical comparisons;
- release artifact verification and SHA-256 integrity records;
- end-to-end evaluation run manifests;
- CardiBench and CardiBridge integration contracts.

A CardiEval result is an evaluation result, not evidence of clinical safety, effectiveness, or regulatory approval.

## Evaluation workflow

```text
CardiBench package
      ↓
package verification
      ↓
submission validation
      ↓
metric/statistical evaluation
      ↓
EvaluationReport
      ↓
SubmissionBundle / run manifest
      ↓
publication snapshot
```

## Installation

```bash
pip install -e '.[test]'
pytest
```

## CLI

```bash
cardieval --manifest examples/benchmark_manifest.json \
  --submission examples/submission.jsonl \
  --model-id demo-model \
  --task-file examples/demo_task.json \
  --output cardiEval-report.json

cardieval verify-benchmark --package benchmark-package.json --root ./benchmark-release
cardieval bridge-validate --package benchmark-package.json --envelope submission-envelope.json --source-role agent
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

## Contracts

`BenchmarkPackage`, `BenchmarkTask`, `PredictionRecord`, `SubmissionBundle`, `EvaluationRunManifest`, `LeaderboardSnapshot`, and `BridgeEnvelope` are versioned machine-readable contracts. Evaluation rejects incompatible benchmark/task identities, duplicate or unknown samples, and invalid split assignments.

## Statistical policy

CardiEval separates descriptive score differences from superiority claims. Decision rules use declared metric direction, confidence intervals, margins, alpha values, and optional multiplicity correction. See `docs/DECISION_POLICY.md`.

## Reproducibility

The evaluation chain is:

`BenchmarkPackage → BenchmarkTask → Submission → EvaluationReport → SubmissionBundle → EvaluationRunManifest → LeaderboardSnapshot`

Artifacts carry stable identities and/or hashes. Cryptographic signing and external key management are outside the evaluator.

## Integration

- **CardiBench:** supplies benchmark packages.
- **CardiBridge:** transports validated Agent/Vex/model submissions.
- **CardiTrace:** can record evaluation provenance.
- **HeartTwin:** uses CardiEval as the independent evaluation boundary.

## Limitations

Passing an evaluation gate does not establish clinical validity. Small subgroups, dataset shift, benchmark construction errors, and unmeasured confounding can still limit interpretation.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the CardiEval release and the exact benchmark package and evaluation protocol used.
