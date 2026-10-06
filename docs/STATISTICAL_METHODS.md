# Statistical assumptions and repaired inference interfaces

`bootstrap_ci`, `paired_difference_ci`, `paired_permutation_pvalue`, and
`compare_predictions` accept `clusters`: one authoritative independent-unit
identifier per record. Bootstrap draws whole clusters with replacement; paired
randomization swaps both models for all records in a cluster together. At least
two independent clusters are required. Unequal cluster sizes retain the
record-weighted metric; cluster weighting is a different estimand and must be
chosen explicitly upstream. Cluster IDs do not prove subject separation across
training and evaluation; the benchmark must establish that independently.

Randomization enumerates all assignments for at most 12 independent units and
uses Monte Carlo plus-one p-values otherwise. Every assignment must define the
metric; undefined/nonfinite assignments cause an error rather than conditioning
the null on valid draws. The test requires paired assignment exchangeability
under its null. Per-record Wilcoxon is rejected for repeated clusters unless a
declared `cluster_aggregate` reduces per-record losses to one scalar per cluster.
Signed-rank assumptions remain necessary after aggregation.

`binomial_accuracy_ci` offers Wilson score and Clopper-Pearson exact intervals
for independent correctness observations. It avoids degenerate percentile
intervals at all-correct/all-wrong boundaries. It must not treat correlated
records as independent binomial trials. Percentile bootstrap remains a method
with parameter-dependent coverage and can be degenerate at boundaries, especially
for perfect discrimination or very small samples. More resamples cannot repair
such degeneracy. Rare-class invalid-bootstrap rejection yields conditional
empirical intervals; validation must report that limitation.

`decide_comparison` records `ci_confidence` (default .95) and `ci_sidedness`
(`two_sided`, `lower`, or `upper`). Its confidence must be at least `1-alpha`;
the two-sided rule is conservative for a single directional test. Caller
metadata must match the actual interval calculation. One-sidedness refers to
raw A-minus-B input; returned bounds/sidedness are oriented so positive favors A.
Only the available one-sided bound can support a directional decision.

An optional adjusted p-value must declare `pvalue_hypothesis`. Equality tests
remain compatible with zero-margin superiority/inferiority. A nonzero margin
requires its corresponding superiority, non-inferiority, or inferiority null
and a matching `pvalue_margin`; a test of equality cannot serve as a margin test.
Strict boundary clearance is required. Multiple-testing family definition,
correction, and test-set reuse control remain the caller's responsibilities.

These software repairs do not establish clinical safety, clinical efficacy,
transportability, or regulatory acceptance. Scientific validation still needs
frozen independent data, labels, predictions, split policy, and a prespecified
analysis plan.
