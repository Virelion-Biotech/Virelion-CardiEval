# Scientific audit changes — 2026-10-09

## Behavior

Add binary negative log score and held-out logistic calibration intercept/slope diagnostics. Reject constant scores and complete/quasi separation.

## Scope and remaining evidence

Diagnostic estimates are not a recalibration on the held-out cohort. Coefficient confidence intervals, clinical decision curves and mandatory metric policies are not implemented. Existing biological-unit resampling requires the manifest to declare authoritative clusters.

## Implementation

- `src/cardieval/calibration.py`
- `src/cardieval/evaluator.py`
- `src/cardieval/metrics.py`
- `tests/test_calibration.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
