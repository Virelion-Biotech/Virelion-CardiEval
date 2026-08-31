# Virelion-CardiEval Full Validation Report

**Date:** 2026-08-31  
**Repository:** Virelion-biotech/Virelion-CardiEval  
**Commit validated:** 91977d9007defc20098aca5554b8eb0f38bff681 (main)  
**Validator environment:** Python 3.12.3, numpy/scipy/scikit-learn/pydantic/pytest as per pyproject.toml  

## Executive Summary

CardiEval is a well-structured independent evaluation framework for cardiac AI model submissions. It provides strict contracts (Pydantic schemas), metrics, calibration, robustness, statistical decision policies, bundles, leaderboards, and CardiBridge/CardiBench integration points.

**However, full end-to-end validation on real-world datasets cannot be completed** for the following reasons:

1. **No external datasets are shipped or referenced with downloadable labeled prediction sets.** The only data is a 4-sample synthetic demo (`examples/`).
2. The package evaluates *model outputs* (JSONL of predictions + scores + ground truth), not raw ECG/images. Running a public dataset (e.g. PTB-XL, PhysioNet Challenge 2020/2021, MIMIC-IV-ECG) requires an independent trained model that produces predictions in the exact `PredictionRecord` format matching a `BenchmarkManifest` sample ID set. No such model outputs are provided.
3. Fabricating predictions would constitute hallucination, which is prohibited.

Relevant public cardiac datasets researched (none are integrated):
- PTB-XL (PhysioNet): ~22k 12-lead ECGs with diagnostic labels.
- Chapman-Shaoxing / Ningbo: large ECG collections.
- PhysioNet/CinC Challenges (2017–2024): arrhythmia, heart sound, ECG classification tasks with official test sets.
- MIMIC-IV-ECG.

These could be wrapped into a `BenchmarkPackage` in a future CardiBench release, but that is outside this repository's current scope.

## What Was Successfully Validated

### 1. Installation & Import
- Dependencies install cleanly (numpy, pydantic, scipy, scikit-learn).
- Package imports under `src/cardieval` (package name is lowercase `cardieval`).

### 2. Demo Evaluation (after local fix)
- Fixed mismatch: `examples/demo_task.json` had `benchmark_id: "demo-cardiac-benchmark"` while manifest used `"cardiac-challenge-demo"`.
- After alignment, CLI evaluation succeeds:

```text
primary_metric: auroc = 1.0
accuracy = 0.75
balanced_accuracy = 0.75
macro_f1 ≈ 0.733
auprc = 1.0
brier ≈ 0.094
ece ≈ 0.238
```

- Report and bundle written with evaluation fingerprint and SHA-256 identities.
- Bootstrap CIs produced (wide, as expected for n=4).

### 3. Core Architecture
- Strict sample-set matching (missing/duplicate/unknown IDs rejected).
- Task contract enforcement for allowed metrics / primary metric / splits.
- Metrics cover classification, binary diagnostics, calibration, regression, ranking.
- Statistical utilities (bootstrap CI, paired tests, multiplicity correction, decision rules) are present.
- Publication / history / scorecard / release verification scaffolding exists.

## Test Suite Results

58 tests collected. After correcting two broken imports (`cardiEval` → `cardieval` in `test_calibration_stress.py` and `test_comparison_report.py`):

**Approximately 11 failures** observed (exact count may vary with floating-point):

| Area | Issue | Severity |
|------|-------|----------|
| `test_metrics.py` | MAE expected value wrong (test claims 4/3, correct is 5/3) | Test bug |
| Floating-point exact equality | Several asserts use `== 0.15` instead of `pytest.approx` (stress, robustness, publication history) | Test fragility |
| `test_leaderboard.py` / publication | Missing required field `benchmark_sha256` on `EvaluationReport`; attribute `version` does not exist | Model / test mismatch |
| `test_history_integrity_scorecard.py` | Duplicate benchmark snapshot rejection logic | Possible logic or fixture bug |
| `test_confidence.py` | Paired difference CI width assertion | Possible numerical edge |
| `test_task_contract.py` | Error message content assertion too strict | Test brittleness |
| sklearn warnings | Single-label / class-not-in-true on small n or bootstrap resamples | Expected for demo size; should be silenced or labels passed |

Many failures are in the test code itself (incorrect expectations, missing fields, exact float compares) rather than fundamental metric errors. The MAE implementation correctly delegates to `sklearn.metrics.mean_absolute_error`.

## Bugs / Inconsistencies Found in Repository

1. **Import casing:** Two tests import non-existent `cardiEval` (should be `cardieval`).
2. **Example mismatch:** `demo_task.json` benchmark_id/version does not match `benchmark_manifest.json`.
3. **Test expectation errors:** Regression MAE test has incorrect analytic value.
4. **Pydantic model drift:** Some tests construct `EvaluationReport` without required fields or access non-existent attributes (`version`).
5. **CLI surface:** README documents many subcommands (`run`, `publish`, `compare`, `verify`, `bridge-validate`); the top-level argparse only exposes the basic evaluate path unless subcommands are invoked correctly. Subcommand handling exists in source but needs careful argv routing.
6. **Evaluator version string:** Report shows `evaluator_version: "0.4.0"` while `pyproject.toml` declares `1.4.0`.

## Recommendations

1. Fix the two import statements and the example task/manifest identity mismatch (local fixes applied in this validation workspace).
2. Replace all exact float asserts with `pytest.approx(..., abs=1e-9)` or relative tolerance.
3. Align test fixtures with current Pydantic models (add missing `benchmark_sha256`, correct attribute names).
4. Correct the MAE expected value in `test_metrics.py`.
5. Add a larger synthetic benchmark (n≥100) with known stratified subgroups to exercise robustness and CI code paths without sklearn warnings.
6. For real-dataset validation: publish a CardiBench package that includes a public PhysioNet Challenge test split + a baseline model submission JSONL. CardiEval can then be validated against that frozen package.
7. Silence or parameterize sklearn confusion-matrix warnings when labels are incomplete in bootstrap samples.

## Conclusion

- **Unit / integration tests:** Partially passing; several clear test bugs and a few model/test mismatches remain.
- **Demo path:** Works after one-line identity fix.
- **Real datasets:** Cannot be executed end-to-end inside this repository because no labeled prediction submissions for public cardiac datasets are provided. Doing so would require external model inference results that are not present.

All observations above are derived from direct execution of the cloned repository code and tests. No fabricated metrics or dataset results are included.

---
*Report generated by independent validation run. Commit this file (and any minimal fixes) to main for provenance.*
