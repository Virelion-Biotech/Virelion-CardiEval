# Virelion-CardiEval

**Independent evaluation infrastructure for cardiac challenge models.**

CardiEval is the evaluation layer of the Virelion cardiac AI stack. Its job is to judge model outputs independently from the code that produced them, using immutable benchmark manifests, strict submission contracts, reproducible metrics, uncertainty intervals, robustness checks, statistical comparison utilities, auditable leaderboard rules, self-contained evaluation bundles, publication snapshots, historical comparison, and release integrity verification.

## Architecture

```text
Benchmark manifest + task contract
              |
              v
Submission JSONL --> validation --> metric engine --> confidence intervals
                                      |                       |
                                      +--> calibration        +--> subgroup robustness
                                      |                       |
                                      +--> model comparison  +--> corrected significance
                                      |                       |
                                      +--> ranking metrics   +--> stress/shift analysis
                                      v                       |
                              EvaluationReport               |
                                      |                       |
                                      +--> provenance --> SubmissionBundle
                                      |                       |
                                      +--> task registry --> leaderboard --> publication snapshot
                                                                      |
                                                                      +--> historical comparison
                                                                      |
                                                                      +--> cross-benchmark scorecard
                                                                      |
                                                                      +--> release manifest --> verification
```

## 1.0 core

- Strict `PredictionRecord` and `BenchmarkManifest` schemas with Pydantic.
- Exact sample-set validation to catch missing, duplicated, or out-of-benchmark predictions.
- Versioned `BenchmarkTask` contracts defining task type, allowed metrics, primary metric/direction, and permitted splits.
- Task-contract enforcement inside `evaluate_submission`.
- Classification metrics: accuracy, balanced accuracy, macro-F1, AUROC, AUPRC.
- Calibration metrics: Brier score and expected calibration error (ECE), plus reliability bins.
- Regression metrics: MAE and RMSE.
- Ranking metrics: reciprocal rank (MRR), hit rate@10, and NDCG@10.
- Seeded percentile bootstrap confidence intervals.
- Paired metric-difference confidence intervals for model-vs-model comparisons.
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
- CLI commands for evaluation, publication, historical comparison, and integrity verification.
- Machine-readable JSON schemas and an explicit evaluation protocol.

## Task contract

A registered task definition is authoritative for scoring. For example:

```json
{
  "benchmark_id": "demo-cardiac-benchmark",
  "version": "1.0.0",
  "task_id": "binary-challenge-detection",
  "task_type": "binary_classification",
  "allowed_metrics": ["accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "brier", "ece"],
  "primary_metric": "auroc",
  "primary_direction": "higher_is_better",
  "splits": ["validation", "test"]
}
```

The evaluator verifies benchmark identity, version, task type, permitted split, allowed metric contract, and the declared primary metric before producing a contract-aware report.

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

Run tests with:

```bash
pytest
```

## Reproducibility boundary

The intended traceability chain is:

`BenchmarkManifest → BenchmarkTask → Submission JSONL → EvaluationReport → SubmissionBundle → LeaderboardSnapshot → ReleaseManifest`

Every stage carries stable identity fields and/or cryptographic hashes. CardiEval provides integrity verification of published artifacts; cryptographic signing and external key management are intentionally separate concerns.

## Design principles

1. **Independent:** evaluation consumes model outputs, not model internals.
2. **Reproducible:** seeds, benchmark versions, evaluator versions, artifact hashes, and evaluation fingerprints are explicit.
3. **Leakage-resistant:** exact benchmark membership is checked before scoring.
4. **Statistically honest:** uncertainty and paired tests are explicit rather than relying on a single point estimate.
5. **Robustness-aware:** subgroup performance and small-cell warnings are reported instead of hiding heterogeneity.
6. **Leaderboard-safe:** scoring contracts are versioned and publication sets reject incompatible or duplicate results.
7. **Composable:** the report, bundle, publication, scorecard, and release schemas are designed as contracts for CardiBench/CardiBridge and audit inputs for CardiTrace.

## Documentation

See `docs/EVALUATION_PROTOCOL.md` for the evaluation contract and `schemas/` for machine-readable publication/release schemas.
