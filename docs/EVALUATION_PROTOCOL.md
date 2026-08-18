# CardiEval Evaluation Protocol

CardiEval is an independent evaluation layer. A model submission is evaluated from serialized outputs and a versioned benchmark task contract; model internals are not required.

## 1. Inputs

A valid evaluation consists of:

- a `BenchmarkManifest` identifying the benchmark, version, split, sample IDs, and dataset hash;
- a `BenchmarkTask` identifying the task type, permitted splits, allowed metrics, primary metric, and direction;
- a JSONL submission containing one prediction per benchmark sample.

The sample set must match exactly. Duplicate, missing, or unknown sample IDs are evaluation errors.

## 2. Scoring

CardiEval computes task-appropriate metrics and uncertainty intervals. The task contract is authoritative for the primary score used for publication.

Classification supports accuracy, balanced accuracy, macro-F1, AUROC, AUPRC, Brier score, ECE, and calibration curves. Regression supports MAE and RMSE. Ranking supports MRR, hit-rate@10, and NDCG@10.

## 3. Robustness and statistics

Declared subgroups are evaluated independently with minimum-size warnings. Model comparisons may use paired permutation testing and explicit paired confidence intervals. Multiple-testing corrections are available for families of hypotheses.

## 4. Provenance

Reports retain benchmark hashes and evaluator versions. `SubmissionBundle` adds a submission hash and deterministic evaluation fingerprint. Publication snapshots are deterministic over their serialized contents.

Release manifests record artifact paths, SHA-256 hashes, sizes, and publication identity. `cardieval verify` checks the local artifacts against the manifest.

## 5. Publication

Only bundles matching the same benchmark/version/task/split and containing the declared primary metric may enter a publication set. Duplicate models or duplicate bundles are rejected.

`cardieval publish` produces a `LeaderboardSnapshot`. `cardieval compare` produces a historical publication comparison. `cardieval verify` validates release artifacts.

## 6. Cross-benchmark reporting

`Scorecard` aggregates compatible leaderboard snapshots into per-model benchmark records, normalized scores, and mean rank. Normalization is performed independently within each benchmark and respects the benchmark's metric direction.

## 7. Reproducibility boundary

A published result should be traceable through:

`BenchmarkManifest → BenchmarkTask → Submission JSONL → EvaluationReport → SubmissionBundle → LeaderboardSnapshot → ReleaseManifest`

Every stage has stable identity fields and/or cryptographic hashes sufficient to detect accidental substitution or tampering.
