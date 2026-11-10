# Clinical regenerative-therapy evidence policy

CardiEval records clinical trial evidence without flattening endpoint hierarchy or
analysis populations. The new `cardieval.clinical_evidence` module is an evidence
contract, not a meta-analysis and not a regulatory decision engine.

## Required distinctions

- Primary endpoint versus key secondary, secondary, exploratory and safety endpoints.
- Intention-to-treat, modified intention-to-treat, per-protocol, safety and subgroup populations.
- Prespecified versus post-hoc subgroup analysis.
- Multiplicity-adjusted versus unadjusted inference.
- Effect estimate, confidence interval, p-value, denominator and follow-up.
- Registry, peer-reviewed publication, regulatory document and company-reported source.

## Interpretation rules

1. A positive subgroup does not convert a failed primary endpoint into trial success.
2. Statistical significance within one subgroup does not establish treatment-effect heterogeneity; record interaction testing separately in the evidence notes.
3. Missing or undisclosed values are `not_reported`, not zero and not a negative result.
4. A company announcement is labelled as such until corroborated against a registry, publication or regulatory source.
5. Regulatory consultation, scientific advice or ATMP classification is not marketing authorization.
6. Clinical efficacy and product potency are separate evidence domains.

## Initial casebook

The CardiBench therapeutic-product casebook defines curation questions for CardiAMP,
ProtheraCytes and SGT-501. CardiEval should ingest adjudicated source records only;
it must not treat casebook prompts or unverified weekly intelligence as trial data.

## Scope limitation

The module validates the structure and basic consistency of manually adjudicated
trial evidence. It does not calculate a treatment effect, assess risk of bias, perform
multiplicity correction, or establish clinical benefit. Those require a dedicated,
prespecified statistical analysis and source-level review.
