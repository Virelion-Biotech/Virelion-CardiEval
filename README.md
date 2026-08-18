# Virelion-CardiEval

**Independent evaluation infrastructure for cardiac challenge models.**

CardiEval is the evaluation layer of the Virelion cardiac AI stack. Its job is to judge model outputs independently from the code that produced them, using immutable benchmark manifests, strict submission contracts, reproducible metrics, uncertainty intervals, robustness checks, statistical comparison utilities, auditable leaderboard rules, self-contained evaluation bundles, publication snapshots, historical comparison, release integrity verification, explicit statistical decision policies, CardiBench-compatible benchmark packages, an end-to-end evaluation runner, and a versioned CardiBridge exchange contract.

## Architecture

```text
CardiBench package
       |
       v
package verification --> BenchmarkManifest + BenchmarkTask
                                  |
                      CardiBridge envelope
                    agent/vex -------------> eval
                                  |
                                  v
Submission JSONL ------------> validation
                                  |
                                  v
                              CardiEval
                 ┌────────────────┼─────────────────┐
                 │                │                 │
              metrics       statistics       robustness
              calibration   CI/tests          subgroup/shift
              diagnostics  multiplicity       stress
                 └────────────────┼─────────────────┘
                                  |
                                  v
                         EvaluationReport
                                  |
                                  +--> provenance --> SubmissionBundle
                                  |
                                  +--> leaderboard --> publication snapshot
                                                        |
                         ┌──────────────────────────────┼───────────────┐
                         v                              v               v
                  historical comparison        cross-benchmark     release manifest
                                                  scorecard         + verification
```

## 1.4 core

- Strict `PredictionRecord` and `BenchmarkManifest` schemas with Pydantic.
- Exact sample-set validation to catch missing, duplicated, or out-of-benchmark predictions.
- Versioned `BenchmarkTask` contracts defining task type, allowed metrics, primary metric/direction, and permitted splits.
- **CardiBench-compatible `BenchmarkPackage`** contract bundling the manifest, task definitions, metadata, and optional artifact hashes.
- Package-level contract validation requiring package and manifest identity to match and all tasks to validate against the manifest.
- Benchmark artifact fingerprinting and verification before model evaluation.
- Classification metrics: accuracy, balanced accuracy, macro-F1, AUROC, AUPRC.
- Binary diagnostic metrics: sensitivity, specificity, positive/negative predictive value, MCC, Cohen's kappa, and explicit confusion-matrix counts.
- Calibration metrics: Brier score and expected calibration error (ECE), plus reliability bins.
- Regression metrics: MAE and RMSE.
- Ranking metrics: reciprocal rank (MRR), hit rate@10, and NDCG@10.
- Seeded percentile bootstrap confidence intervals.
- Paired metric-difference confidence intervals for model-vs-model comparisons.
- Explicit superiority/non-inferiority/inferiority/inconclusive decision rules with metric-direction handling and optional multiplicity-adjusted p-values.
- Release quality gates for evaluation integrity, artifact verification, primary-metric presence, and subgroup warnings.
- Declared subgroup evaluation with minimum-size warnings.
- Robustness and stress/shift analysis with direction-aware degradation.
- Paired permutation testing and optional Wilcoxon testing when an explicit sample-wise score/loss is supplied.
- Bonferroni and Benjamini-Hochberg multiple-testing correction.
- Deterministic leaderboard aggregation and task-controlled primary scoring.
- `SubmissionBundle` ingestion with benchmark/task/hash/split/model consistency checks.
- Immutable `LeaderboardSnapshot` publication artifacts.
- Deterministic snapshot integrity hashes and historical rank/score delta reports.
- Cross-benchmark `Scorecard` aggregation with per-benchmark normalization and mean-rank reporting.
- `ReleaseManifest` artifact records with SHA-256 and size verification.
- **End-to-end `run` pipeline** producing an evaluation report, submission bundle, and deterministic run manifest from a validated benchmark package plus model submission.
- **CardiBridge protocol** with strict versioned envelopes, Agent/Vex submission payloads, payload integrity checks, benchmark/task identity checks, and capability negotiation.
- CLI commands for evaluation, publication, historical comparison, release verification, benchmark-package verification, end-to-end runs, and bridge envelope validation.
- Machine-readable JSON schemas and explicit evaluation/decision/package/run/bridge protocols.

## CardiBridge integration

CardiBridge is the formal exchange boundary between CardiAgent/CardiVex and CardiEval. A producer sends a strict `BridgeEnvelope` whose payload is a `PredictionSubmission` using the same `PredictionRecord` contract as native CardiEval submissions.

The bridge is deliberately fail-closed. Before accepting an envelope, CardiEval verifies:

1. protocol/schema validity;
2. expected source role and `eval` target role;
3. benchmark ID/version;
4. supported payload type;
5. canonical payload SHA-256;
6. payload task ID;
7. task existence and compatibility with the benchmark manifest.

Capabilities are negotiated through `BridgeCapabilities`, allowing producers and CardiEval to intersect supported payload types and task types before sending a submission.

Validate an incoming Agent/Vex envelope without running a full evaluation:

```bash
cardieval bridge-validate \
  --package benchmark-package.json \
  --envelope submission-envelope.json \
  --source-role agent
```

The bridge uses the same `PredictionRecord` objects as the local evaluator, so bridge-originated data does not bypass sample-set or benchmark/task validation.

See `docs/CARDIBRIDGE_PROTOCOL.md` and `schemas/bridge-envelope.schema.json` for the exchange contract.

## CardiBench integration

A `BenchmarkPackage` is the formal handoff from CardiBench into CardiEval. It contains a benchmark release identity, an exact `BenchmarkManifest`, one or more `BenchmarkTask` definitions, and optional artifact integrity records.

The package must satisfy these invariants:

1. Package benchmark ID equals manifest benchmark ID.
2. Package release version equals manifest version.
3. Every task references the same benchmark ID/version.
4. Every task's type and permitted split are compatible with the manifest.
5. Task IDs and artifact paths are unique.
6. Model submissions contain exactly the package manifest's sample IDs for the selected task.
7. Duplicate or unknown sample IDs are rejected.

Verify a package before evaluation:

```bash
cardieval verify-benchmark \
  --package benchmark-package.json \
  --root ./benchmark-release
```

### One-command end-to-end evaluation

The `run` command performs the full protected path:

```text
verify package → validate submission → evaluate → write report → write bundle → write run manifest
```

Example:

```bash
cardieval run \
  --package benchmark-package.json \
  --root ./benchmark-release \
  --submission submission.jsonl \
  --model-id my-model \
  --task-id binary-challenge-detection \
  --report-output outputs/report.json \
  --bundle-output outputs/bundle.json \
  --run-output outputs/run.json
```

`EvaluationRunManifest` records the benchmark package hash, submission hash, evaluation fingerprint, bundle ID, model/task identity, and output paths. This is the canonical machine-readable record for one evaluation event.

See `docs/CARDIBENCH_INTEGRATION.md` for the full package contract and `schemas/evaluation-run-manifest.schema.json` for the run artifact schema.

## Task contract

A registered task definition is authoritative for scoring. For example:

```json
{
  "benchmark_id": "demo-cardiac-benchmark",
  "version": "1.0.0",
  "task_id": "binary-challenge-detection",
  "task_type": "binary_classification",
  "allowed_metrics": ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "brier", "ece", "sensitivity", "specificity"],
  "primary_metric": "auroc",
  "primary_direction": "higher_is_better",
  "splits": ["validation", "test"]
}
```

The evaluator verifies benchmark identity, version, task type, permitted split, allowed metric contract, and the declared primary metric before producing a contract-aware report.

## Statistical decision policy

CardiEval separates descriptive score differences from claims of superiority. Differences are oriented so positive values always favor model A. A declared confidence interval, decision margin, alpha level, and optional multiplicity-adjusted p-value can be passed to the decision layer to classify comparisons as superior, non-inferior, inconclusive, or inferior.

These are evaluation rules, not clinical approval criteria. Passing a CardiEval gate does not establish clinical safety, effectiveness, or readiness for patient care.

See `docs/DECISION_POLICY.md` for the formal policy.

## Publication and history

Validated `SubmissionBundle` files can be published into a leaderboard snapshot. CardiEval rejects incompatible benchmark/task identities, inconsistent benchmark hashes, duplicate bundle IDs, duplicate models, missing primary metrics, and mixed splits.

Snapshots are deterministically hashed. Historical comparison reports track rank changes, score changes, new models, and removed models. Multiple snapshots can also be combined into a cross-benchmark `Scorecard`.

## CLI

Evaluate a model and optionally emit an interoperable bundle:

```bash
cardieval \
  --manifest examples/benchmark_manifest.json \
  --submission examples/submission.jsonl \
  --model-id demo-model \
  --task-file examples/demo_task.json \
  --output cardiEval-report.json \
  --bundle-output cardiEval-bundle.json
```

Publish bundles from a directory:

```bash
cardieval publish \
  --task-file examples/demo_task.json \
  --bundles-dir ./bundles \
  --output leaderboard.json
```

Compare two publication snapshots:

```bash
cardieval compare \
  --previous previous-leaderboard.json \
  --current leaderboard.json \
  --output leaderboard-delta.json
```

Verify a release manifest against local artifacts:

```bash
cardieval verify \
  --manifest release-manifest.json \
  --root .
```

Run the test suite with:

```bash
pytest
```

## Reproducibility boundary

The intended traceability chain is:

`BenchmarkPackage → BenchmarkManifest → BenchmarkTask → CardiBridge Envelope / Submission JSONL → EvaluationReport → SubmissionBundle → EvaluationRunManifest → LeaderboardSnapshot → ReleaseManifest`

Every stage carries stable identity fields and/or cryptographic hashes. CardiEval provides integrity verification of published artifacts; cryptographic signing and external key management are intentionally separate concerns.

## Design principles

1. **Independent:** evaluation consumes model outputs, not model internals.
2. **Reproducible:** seeds, benchmark versions, evaluator versions, artifact hashes, and evaluation fingerprints are explicit.
3. **Leakage-resistant:** exact benchmark membership is checked before scoring.
4. **Statistically honest:** uncertainty, multiplicity correction, and explicit decision rules are preferred to unsupported winner claims.
5. **Robustness-aware:** subgroup performance and small-cell warnings are reported instead of hiding heterogeneity.
6. **Leaderboard-safe:** scoring contracts are versioned and publication sets reject incompatible or duplicate results.
7. **Composable:** the report, bundle, publication, scorecard, release, decision, benchmark-package, run, and bridge schemas are designed as contracts for CardiBench/CardiBridge and audit inputs for CardiTrace.

## Documentation

See `docs/EVALUATION_PROTOCOL.md`, `docs/DECISION_POLICY.md`, `docs/CARDIBENCH_INTEGRATION.md`, `docs/CARDIBRIDGE_PROTOCOL.md`, and `schemas/` for the formal evaluation, decision, integration, and machine-readable contracts.
