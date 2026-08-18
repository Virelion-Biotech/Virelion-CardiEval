# CardiEval Decision Policy

## Purpose

CardiEval separates descriptive scoring from statistical claims. A higher leaderboard score does not by itself establish that one model is meaningfully better than another.

## Oriented differences

For a `higher_is_better` metric, the oriented difference is:

`model_a - model_b`

For a `lower_is_better` metric, the evaluator reverses the sign so positive values still favor model A.

## Superiority

A comparison is classified as **superior** when the confidence interval clears the declared margin in the favorable direction and, when a p-value is supplied, the adjusted p-value is below alpha.

## Non-inferiority

A comparison is classified as **non-inferior** when the confidence interval does not cross the adverse margin and the supplied adjusted p-value supports the decision rule.

## Inconclusive

A result is **inconclusive** when the interval overlaps the decision boundary or the supplied multiplicity-adjusted evidence is insufficient.

## Inferiority

A result is **inferior** when the confidence interval lies beyond the adverse margin under the declared metric direction.

## Release gates

A release should fail on hard errors such as evaluation errors, artifact-integrity failures, or missing primary metrics. Small subgroup warnings can remain non-blocking when explicitly permitted by the release policy.

## Multiplicity

When multiple hypotheses are tested, CardiEval expects callers to provide corrected p-values using the existing Bonferroni or Benjamini-Hochberg utilities. The decision layer does not silently reinterpret uncorrected p-values as corrected evidence.

## Scope

These rules are statistical evaluation rules, not clinical approval criteria. A model passing CardiEval's gates is not thereby validated for clinical use.
