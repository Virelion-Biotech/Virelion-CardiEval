# Scientific repairs in 1.5.0

Direct evaluation rejects duplicate sample IDs and binary labels outside 0/1.
Reference labels and subgroup assignments are evaluator-controlled independently
of whether both maps are supplied. Reports bind the actual sample/label/subgroup
panel, cluster/query maps and metric configuration through `reference_sha256`.
Publication and leaderboards require that reference hash and a shared dataset
hash/task/split. `expected_dataset_sha256` optionally pins a task to its admitted
dataset. Old reports and bundles remain readable; unbound old reports need
re-evaluation before publication. Legacy bundle fingerprints retain their
original 0.4 report representation. Historical comparisons check data/reference
identity. Explicit repeated-model aggregation is available only through
`allow_repeated_models=True` on the low-level leaderboard, and needs a
prespecified repeat protocol. Strict publication continues to reject duplicates.

Supply `BenchmarkManifest.authoritative_clusters` as a complete sample-ID to
independent-subject-ID map. Repeated rows are resampled together; metadata reports
both row and independent-unit counts. The independent-record assumption is
warned when that map is absent. IID accuracy uses a conservative exact
Clopper-Pearson interval, avoiding the all-correct [1,1] bootstrap failure.
Wilson intervals remain an explicit helper option. Other metrics retain
percentile bootstrap, report rejected draws, and withhold degenerate/unavailable
intervals with reasons. Coverage of those methods must still be assessed for the
intended population and endpoint; this release is not a universal coverage proof.

For ranking, supply `authoritative_queries` as a complete sample-ID to query-ID
map. MRR, hit rate and linear-gain NDCG average equally across queries; uncertainty
resamples queries, or their subjects when a cluster map is also supplied. A query
must belong to one subject and one subgroup. With no query map the benchmark is
explicitly one fixed list, with a warning and no population CI. Equal-score ties
use expected metric values under uniform tie order; manifest ordering cannot
create artificial discrimination. Gains remain linear and are recorded.

Calibration metrics and applicable diagnostic measures remain available in
single-class subgroups. Undefined metrics and missing confidence intervals carry
explicit reasons. ECE reports its bin count, edges, counts and observed/predicted
means; `evaluate_submission(..., ece_bins=20)` changes that declared binning.
Empty calibration bins use null values. ECE is a bin-dependent summary, not proof
of calibration. Loss-metric subgroup ranges are nonnegative.

Comparison decisions bind CI confidence/sidedness to alpha and reject equality
p-values used for nonzero-margin hypotheses. Supplied metadata must match the
actual calculation. See `STATISTICAL_METHODS.md` for interfaces and assumptions.
The caller still prespecifies multiplicity families and controls test-set reuse.

These fixes support auditable statistical evaluation. They do not establish
clinical efficacy, physiological validity, subject disjointness across cohorts,
authenticity of uploaded data, or regulatory approval.
