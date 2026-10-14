# Virelion-CardiEval

CardiEval is an independent evaluation library for cardiac machine-learning models. It evaluates submitted predictions against versioned benchmark packages without requiring access to model internals.


<!-- VALIDATION_PRODUCT_STATUS:BEGIN -->
## Validation and product completion status

**Reviewed 2026-10-14; implementation base `294628f87044`.** **Status: Synthetic/contract/statistical verification; real benchmark adoption incomplete.** This is an evidence-backed scope assessment, not a percentage-complete estimate or a declaration that the product is finished.

| Area | Current evidence / remaining gate |
|---|---|
| Software / operational checks | Automated test sources and CI definitions are present; this documentation audit does not report a new algorithm/data-validation run. |
| Scientific or domain evidence | The strict synthetic kit checks evaluator-controlled labels, metrics, publication and integrity. Binary probability reporting is opt-in; software checks do not establish cardiac model performance or clinical utility. |
| Remaining for the declared product scope | Adopt frozen real benchmark packages and independently produced submissions; enforce appropriate subject/cluster units and subgroup coverage; qualify calibration/clinical-utility reporting and interval behavior for intended tasks; complete deployment signing/key/access-control requirements where needed. |

Evidence and limits: [repository report](validation/README.md). The [organization status index](https://github.com/Virelion-Biotech/Virelion-HeartTwin/blob/main/docs/STACK_VALIDATION_STATUS.md) separates software, numerical, data-readiness and empirical qualification. Code/schema execution or a completed download is not a substitute for endpoint-specific validation.
<!-- VALIDATION_PRODUCT_STATUS:END -->

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

Scientific repairs in 1.5.0 add patient/query-aware inference, exact IID accuracy
intervals, reference-bound publication and explicit unavailable-metric reasons.
See [scientific repairs](docs/SCIENTIFIC_REPAIRS.md) and
[statistical methods](docs/STATISTICAL_METHODS.md) for contracts and migration.

CardiEval evaluates serialized model outputs; it is not a raw ECG, imaging, or signal preprocessing pipeline. Real cardiac validation requires a frozen benchmark package with independently controlled labels, explicit split policy, subject-level separation where applicable, and an independently produced model submission. Synthetic validation only establishes software/contract behavior.

Statistical significance does not establish clinical significance or causal validity. Hashes detect accidental substitution or tampering but do not provide authenticity; digital signing, key management, access control, and transport security remain deployment concerns.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
