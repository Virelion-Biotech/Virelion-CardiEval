# Regeneration versus repair assessment

CardiEval's regeneration assessment keeps eight evidence domains independent and emits a categorical interpretation. It does not calculate a weighted or composite regeneration score.

## Domains

1. Cardiomyocyte cell-cycle entry
2. Completed mitosis/cytokinesis
3. Increased cardiomyocyte number and lineage evidence
4. Functional maturation of newly generated cardiomyocytes
5. Ventricular function/remodelling
6. Fibrosis/scar
7. Arrhythmia/electrophysiological safety
8. Off-target proliferation, tumorigenicity, organ toxicity and surveillance duration

Each domain records status, assay, model/species, comparator, timepoint, sample size, denominator, effect estimate, uncertainty, p-value, prespecification, subgroup and multiplicity context, follow-up and source IDs. Unknown values stay null or not_assessed; they are never inferred.

## Interpretation rules

- **regeneration_demonstrated_in_model** requires direct support in all four regenerative-core domains: cell-cycle entry, completed mitosis/cytokinesis, cardiomyocyte number/lineage, and functional maturation.
- **repair_or_preservation_only** applies when function/remodelling or scar evidence is present but the regenerative core is not established.
- **mixed_evidence** applies when multiple regenerative-core domains are supported but the full chain is incomplete, or when adverse safety evidence conflicts with positive findings.
- **insufficient_evidence** applies when the evidence does not reach those thresholds.

Ki67, EdU/BrdU, pHH3, LVEF, scar area or marker expression alone cannot establish regeneration. Even positive lineage/division evidence does not establish mature electrical/mechanical integration.

## Source-linked initial case

`examples/regeneration/myc_cyclint1_modrna_mi.v1.json` encodes the 2026 Boikova et al. mouse MI study (DOI: 10.1038/s44161-026-00910-7). It separates the localized transient M-C modRNA exposure from the distinct global/sustained MycER(T2)+AAV-Ccnt1 genetic exposure that produced severe toxicity. It records the reported cell-cycle, mitosis, cardiomyocyte-number, function, scar and safety findings while marking daughter-cell functional maturation and formal electrophysiological safety as not assessed. The current strict interpretation is therefore mixed evidence, not an unqualified regeneration claim.

## CLI

```bash
cardieval assess-regeneration \
  --input examples/regeneration/myc_cyclint1_modrna_mi.v1.json \
  --output build/myc-cyclint1-regeneration-report.v1.json
```

The JSON report is deterministic and includes caveats for missing denominators, small samples, subgroup-only analyses, prespecification, multiplicity and safety follow-up. This is a research evidence assessment, not clinical decision support.
