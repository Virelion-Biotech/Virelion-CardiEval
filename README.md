# Virelion-CardiEval

**Independent evaluation infrastructure for cardiac challenge models.**

CardiEval is the evaluation layer of the Virelion cardiac AI stack. Its job is to judge model outputs independently from the code that produced them, using immutable benchmark manifests, strict submission contracts, reproducible metrics, uncertainty intervals, robustness checks, statistical comparison utilities, and auditable leaderboard rules.

## Architecture

```text
Benchmark manifest
       |
       v
Submission JSONL --> validation --> metric engine --> confidence intervals
                                      |                       |
                                      +--> calibration        +--> subgroup robustness
                                      |                       |
                                      +--> model comparison  +--> corrected significance
                                      v                       |
                              EvaluationReport               |
                                      |                       |
                                      +--> task registry --> leaderboard
```

## Current core

- Strict `PredictionRecord` and `BenchmarkManifest` schemas with Pydantic.
- Exact sample-set validation to catch missing, duplicated, or out-of-benchmark predictions.
- Classification metrics: accuracy, balanced accuracy, macro-F1, AUROC, AUPRC.
- Calibration metrics: Brier score and expected calibration error (ECE).
- Regression metrics: MAE and RMSE.
- Seeded percentile bootstrap confidence intervals.
- Declared subgroup evaluation with minimum-size warnings.
- Paired permutation testing for model-vs-model comparisons.
- Optional paired Wilcoxon testing only when an explicit sample-wise score/loss is supplied; aggregate metrics such as AUROC and macro-F1 are never treated as per-sample quantities.
- Bonferroni and Benjamini-Hochberg multiple-testing correction.
- Deterministic leaderboard aggregation with mean repeated-report scores, stable ranking, and tie handling.
- Versioned benchmark/task registry defining allowed metrics, primary metric/direction, task type, and valid splits.
- SHA-256 benchmark/artifact fingerprinting and provenance-aware reports.
- JSON reports designed for later CardiBridge and CardiTrace integration.
- Python 3.10-3.12 CI with linting and tests.

## Submission format

Predictions are JSON Lines. Every line must contain a stable `sample_id`, `y_true`, and `y_pred`; probabilistic classifiers may additionally provide `score` and `subgroup`.

```json
{"sample_id":"sample-001","y_true":0,"y_pred":0,"score":0.08,"subgroup":"example-group"}
```

The evaluator rejects duplicate or out-of-benchmark IDs and requires an exact sample-set match. This prevents silent evaluation on a convenient subset or accidental contamination by unknown samples.

## Run locally

```bash
pip install -e '.[dev]'
cardieval \
  --manifest examples/benchmark_manifest.json \
  --submission examples/submission.jsonl \
  --model-id demo-model \
  --output cardiEval-report.json
```

Or run the test suite:

```bash
pytest
```

## Design principles

1. **Independent:** evaluation consumes model outputs, not model internals.
2. **Reproducible:** seeds, benchmark versions, and evaluator versions are explicit.
3. **Leakage-resistant:** exact benchmark membership is checked before scoring.
4. **Statistically honest:** uncertainty and paired tests are explicit rather than relying on a single point estimate.
5. **Robustness-aware:** subgroup performance and small-cell warnings are reported instead of hiding heterogeneity.
6. **Leaderboard-safe:** scoring contracts are versioned and rankings are derived from compatible reports only.
7. **Composable:** report schemas are designed to become the contract for CardiBridge and the audit layer for CardiTrace.

## Roadmap

Next: ranking-task metrics, richer robustness suites, calibration curves, confidence-aware model comparison reports, signed/provenance-aware artifacts, and integration with CardiBench, CardiAgent, and CardiVex.
