# Virelion-CardiEval

**Independent evaluation infrastructure for cardiac challenge models.**

CardiEval is the evaluation layer of the Virelion cardiac AI stack. Its job is to judge model outputs independently from the code that produced them, using immutable benchmark manifests, strict submission contracts, reproducible metrics, uncertainty intervals, and statistical comparison utilities.

## Current architecture

```text
Benchmark manifest
       |
       v
Submission JSONL --> validation --> metric engine --> confidence intervals
                                      |
                                      +--> statistical comparison utilities
                                      |
                                      v
                              EvaluationReport
```

### v0.1 core

- Strict `PredictionRecord` and `BenchmarkManifest` schemas with Pydantic.
- Exact sample-set validation to catch missing, duplicated, or out-of-benchmark predictions.
- Classification metrics: accuracy, balanced accuracy, macro-F1, AUROC, AUPRC.
- Regression metrics: MAE and RMSE.
- Seeded percentile bootstrap confidence intervals.
- Paired permutation and Wilcoxon testing utilities for future model-vs-model evaluation.
- SHA-256 file fingerprinting for provenance-sensitive benchmark artifacts.
- JSON report output suitable for later CardiBridge/CardiTrace integration.
- Python 3.10-3.12 CI with linting and tests.

## Submission format

Predictions are JSON Lines. Every line must contain a stable `sample_id`, `y_true`, and `y_pred`; probabilistic classifiers may additionally provide `score`.

```json
{"sample_id":"sample-001","y_true":0,"y_pred":0,"score":0.08}
```

The evaluator rejects duplicates and requires the submission sample set to exactly match the benchmark manifest. This prevents silent evaluation on a convenient subset or accidental contamination by unknown samples.

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
4. **Statistically honest:** uncertainty is reported rather than treating one point estimate as ground truth.
5. **Composable:** report schemas are designed to become the contract for CardiBridge and the audit layer for CardiTrace.

## Roadmap

The next major layers are hierarchical benchmark/task definitions, subgroup and robustness evaluation, calibration, model-vs-model significance testing, leaderboard aggregation, artifact signing/provenance, and integration with CardiBench, CardiAgent, and CardiVex.
