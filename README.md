# Virelion-CardiEval

**Independent evaluation infrastructure for cardiac challenge models.**

CardiEval is the evaluation layer of the Virelion cardiac AI stack. Its job is to judge model outputs independently from the code that produced them, using immutable benchmark manifests, strict submission contracts, reproducible metrics, uncertainty intervals, robustness checks, statistical comparison utilities, auditable leaderboard rules, self-contained evaluation bundles, and publication snapshots.

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
                                      +--> task registry --> leaderboard publication
```

## Current core

- Strict `PredictionRecord` and `BenchmarkManifest` schemas with Pydantic.
- Exact sample-set validation to catch missing, duplicated, or out-of-benchmark predictions.
- Versioned `BenchmarkTask` contracts defining task type, allowed metrics, primary metric/direction, and permitted splits.
- Task-contract enforcement inside `evaluate_submission`, not just in the registry or CLI.
- Classification metrics: accuracy, balanced accuracy, macro-F1, AUROC, AUPRC.
- Calibration metrics: Brier score and expected calibration error (ECE).
- Calibration-curve/reliability-bin output for probabilistic binary models.
- Regression metrics: MAE and RMSE.
- Ranking metrics: reciprocal rank (MRR), hit rate@10, and NDCG@10.
- Seeded percentile bootstrap confidence intervals.
- Paired metric-difference confidence intervals for model-vs-model comparisons.
- Declared subgroup evaluation with minimum-size warnings.
- Robustness summaries for subgroup spread and relative degradation under perturbation.
- Stress/shift comparison utilities with direction-aware degradation and aggregate stress scores.
- Paired permutation testing for model-vs-model comparisons.
- Optional paired Wilcoxon testing only when an explicit sample-wise score/loss is supplied; aggregate metrics such as AUROC and macro-F1 are never treated as per-sample quantities.
- Bonferroni and Benjamini-Hochberg multiple-testing correction.
- Deterministic leaderboard aggregation with mean repeated-report scores, stable ranking, and tie handling.
- `SubmissionBundle` contract linking benchmark identity, submission hash, evaluation fingerprint, task ID, model ID, and full report for CardiBench/CardiBridge interoperability.
- `LeaderboardSnapshot` publication contract that ingests validated bundles, enforces primary-metric eligibility, rejects duplicate models, and produces a deterministic publication-ready ranking.
- SHA-256 benchmark/artifact fingerprinting, canonical JSON hashing, and deterministic evaluation fingerprints.
- Python 3.10-3.12 CI with linting and tests.

## Submission format

Predictions are JSON Lines. Every line must contain a stable `sample_id`, `y_true`, and `y_pred`; probabilistic classifiers may additionally provide `score` and `subgroup`. Ranking tasks require `score` for every record, with `y_true` representing non-negative relevance.

```json
{"sample_id":"sample-001","y_true":0,"y_pred":0,"score":0.08,"subgroup":"example-group"}
```

The evaluator rejects duplicate or out-of-benchmark IDs and requires an exact sample-set match. This prevents silent evaluation on a convenient subset or accidental contamination by unknown samples.

## Task contract

A registered task definition is JSON and is authoritative for scoring. For example:

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

The evaluator verifies benchmark identity, version, task type, permitted split, allowed metrics, and the existence of the declared primary metric before producing a contract-aware report.

## Leaderboard publication

Validated `SubmissionBundle` files can be ingested as a publication set. CardiEval checks benchmark/version identity, benchmark hash consistency, task identity, split consistency, primary-metric availability, unique models, and unique bundle IDs before ranking.

```python
from cardieval import load_bundle, publish_leaderboard

bundles = [load_bundle("model-a.json"), load_bundle("model-b.json")]
snapshot = publish_leaderboard(bundles, task)
```

The resulting `LeaderboardSnapshot` is machine-readable and contains the benchmark/task identity, primary scoring contract, bundle IDs, model count, and deterministic leaderboard entries.

## Run locally

```bash
pip install -e '.[dev]'
cardieval \
  --manifest examples/benchmark_manifest.json \
  --submission examples/submission.jsonl \
  --model-id demo-model \
  --task-file examples/demo_task.json \
  --output cardiEval-report.json \
  --bundle-output cardiEval-bundle.json
```

Or run the test suite:

```bash
pytest
```

## Design principles

1. **Independent:** evaluation consumes model outputs, not model internals.
2. **Reproducible:** seeds, benchmark versions, evaluator versions, artifact hashes, and evaluation fingerprints are explicit.
3. **Leakage-resistant:** exact benchmark membership is checked before scoring.
4. **Statistically honest:** uncertainty and paired tests are explicit rather than relying on a single point estimate.
5. **Robustness-aware:** subgroup performance and small-cell warnings are reported instead of hiding heterogeneity.
6. **Leaderboard-safe:** scoring contracts are versioned and publication sets are restricted to compatible bundles.
7. **Composable:** the report, bundle, and publication schemas are designed as contracts for CardiBench and CardiBridge and as audit inputs for CardiTrace.

## Roadmap

Next: richer stress/shift suites, correction-aware decision rules, signed artifact verification, automated publication CLI, and full CardiBench/CardiAgent/CardiVex integration.
