# CardiEval Evaluation Protocol

CardiEval is an independent evaluation layer. A model submission is evaluated from serialized outputs and a versioned benchmark task contract; model internals are not required.

## 1. Inputs

A valid evaluation consists of:

- a BenchmarkPackage identifying the benchmark release, immutable manifest, and task contracts;
- a benchmark manifest defining the exact sample set and dataset identity;
- a BenchmarkTask defining task type, permitted splits, allowed metrics, primary metric, direction, and whether evaluator-controlled ground truth is required;
- a JSONL submission containing one prediction per benchmark sample.

The sample set must match exactly. Duplicate, missing, or unknown sample IDs are evaluation errors.

For independent evaluation, BenchmarkTask.requires_authoritative_labels should be true and the manifest should provide evaluator-controlled authoritative_labels. In that mode CardiEval ignores any submission-supplied y_true; protected submissions can omit y_true entirely. Evaluator-controlled subgroup assignments can likewise be supplied through authoritative_subgroups.

## 2. Scoring

CardiEval computes task-appropriate metrics and uncertainty intervals. The task contract is authoritative for the primary score used for publication.

Binary classification supports accuracy, balanced accuracy, macro-F1, AUROC, AUPRC, Brier score, negative log score, ECE, and threshold-dependent diagnostic metrics. Regression supports MAE and RMSE. Ranking supports MRR, hit-rate@10, and NDCG@10.

The evaluator rejects metrics that are incompatible with the declared task type and rejects primary-metric direction mismatches.

For a binary probabilistic task, prespecify `requires_probability_reporting: true`
and include `brier`, `log_score` and `ece` in `allowed_metrics`. Every prediction
must provide a probability score. The task enforces these proper-score/diagnostic
outputs at evaluation and again before leaderboard publication. ECE must include
equal-width reliability bins with counts and rates; Brier details must include
finite logistic calibration intercept/slope diagnostics or a nonempty unavailable
reason; log score must declare its endpoint clipping epsilon. Missing metrics,
empty diagnostics and inconsistent bin counts cannot be published under this policy.
An AUROC/AUPRC primary metric can still be used alongside those required outputs.

Unavailable coefficients, such as separated or constant-score fits, remain explicitly
unavailable and produce warnings. Reporting completeness does not establish that
predictions are calibrated. Coefficients are held-out diagnostics, not a recalibration
instruction; their clustered uncertainty and clinically relevant decision utility
still require a prespecified analysis. Independent ground truth and biological-unit
metadata are separate requirements. Existing nonprobabilistic/legacy tasks default
to `requires_probability_reporting: false`; no global scientific qualification is
introduced. The strict policy is included in the canonical reference hash, so strict
and optional runs cannot be pooled into one publication set.

## 3. Robustness and statistics

Declared subgroups are evaluated independently with minimum-size warnings. Task metric restrictions apply to subgroup results as well as the overall report.

Model comparisons may use paired permutation testing, Wilcoxon signed-rank testing when a valid samplewise score is supplied, and explicit paired confidence intervals. Multiple-testing corrections are available for families of hypotheses, and non-finite p-values are rejected.

## 4. Provenance and integrity

Reports retain benchmark identity, benchmark hashes, evaluator version, and ground-truth source. SubmissionBundle binds the submission hash, benchmark hash, evaluator fingerprint, and report into a self-verifying artifact.

Publication snapshots validate bundle/model counts, leaderboard identity, and bundle ID integrity. Release manifests verify declared artifact hashes and their own deterministic self-hash.

Path handling is fail-closed against absolute paths, parent-directory traversal, and symlink escapes.

## 5. Publication

Only bundles matching the same benchmark/version/task/split and validated against the task contract may enter a publication set. Duplicate models, duplicate bundles, invalid primary metrics, and tampered bundles are rejected.

cardieval publish produces a LeaderboardSnapshot. cardieval compare produces a historical publication comparison. cardieval verify validates release artifacts.

## 6. Cross-benchmark reporting

Scorecard aggregates compatible leaderboard snapshots into per-model benchmark records, normalized scores, and mean rank. Normalization is performed independently within each benchmark and respects the benchmark's metric direction. A model scorecard entry is tied to one distinct benchmark/task/split snapshot.

## 7. Reproducibility boundary

A published result should be traceable through:

BenchmarkPackage -> BenchmarkTask -> Submission JSONL -> EvaluationReport -> SubmissionBundle -> LeaderboardSnapshot -> ReleaseManifest

Every stage carries stable identity fields and/or cryptographic hashes sufficient to detect accidental substitution or tampering.

SHA-256 provides integrity detection, not authenticity. Digital signing, key management, authorization, access control, and secure transport belong to the deployment layer.

## 8. Validation boundary

The repository's synthetic validation kit is a software/contract test. It is not clinical validation and does not establish safety, effectiveness, or regulatory acceptance.

Real cardiac validation additionally requires a frozen CardiBench package, evaluator-controlled labels, explicit subject-level split policy where applicable, and independently produced model predictions for the frozen sample set.
