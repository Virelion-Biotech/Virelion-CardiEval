# Initial source-adjudicated cardiac therapy evidence register

As-of date: 2026-11-10. This is a curated evidence snapshot, not a meta-analysis,
risk-of-bias assessment, or regulatory determination. Source class and analysis
population must remain visible in downstream evaluation.

## CardiAMP Cell Therapy System — NCT02438306

- Product: autologous bone-marrow mononuclear cells delivered intramyocardially
  using the CardiAMP system.
- Trial design source: Raval et al., *American Heart Journal* (2018), PMID 29803986.
  The pivotal study was designed as randomized, double-blind and sham-controlled;
  the primary endpoint was change in 6-minute walk distance adjusted for major
  adverse cardiovascular events at 12 months.
- Outcome source: BioCardia corporate presentation (August 2025) states that the
  primary three-tier efficacy composite was not met. It separately reports a
  favourable observation in participants with baseline NT-proBNP >500 pg/mL.
- Adjudication: record the overall primary efficacy endpoint as **not met** based
  on sponsor-reported results. Do not convert the biomarker subgroup into overall
  trial success. The subgroup finding remains company-reported and requires
  interaction/multiplicity details before any effect-modification claim.
- Sources:
  - https://pubmed.ncbi.nlm.nih.gov/29803986/
  - https://www.biocardia.com/media/Investors/presentations_and_events/Corporate_Presentation_08_2025.pdf

## ProtheraCytes — EXCELLENT, NCT02669810

- Product: autologous expanded peripheral-blood CD34-positive cells, administered
  by transendocardial intramyocardial injection after acute myocardial infarction.
- Manufacturing/potency source: Aries et al., *Scientific Reports* (2023),
  PMID 37952030, describes clinical-batch characterization and VEGF secretion as
  the most practical, reliable and consistent potency assay among tested readouts.
- Trial design source: EXCELLENT protocol paper, PMCID PMC11911568, describes the
  primary objective as adjudicated major adverse cardiac events through six months,
  with secondary efficacy objectives.
- Clinical report: Roncalli et al. (2026), PMID 41774093, reports a randomized,
  open-label Phase I/IIb study enrolling 77 participants; 49 were in the
  per-protocol population and complete SF-36 data were available for 31 treated
  and 12 control participants. Several health-related quality-of-life domains
  improved in the treated complete-case subset.
- Adjudication: this HRQoL report is not interchangeable with the primary MACE
  endpoint. Preserve the open-label design, small complete-case subset, and
  analysis-population limitations. The cited sources do not establish definitive
  clinical efficacy or marketing authorization.
- Sources:
  - https://pubmed.ncbi.nlm.nih.gov/37952030/
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC11911568/
  - https://pubmed.ncbi.nlm.nih.gov/41774093/

## SGT-501 — ARTEMIS, NCT07148089

- Product: investigational rAAV8-DES-CASQ2 gene therapy candidate for CPVT.
- Registry source: ClinicalTrials.gov NCT07148089 describes a Phase 1b,
  multicentre, open-label, dose-finding study of a single intravenous dose.
- Sponsor filing: Solid Biosciences' 2026 Form 10-Q states that the study is
  designed to assess safety and tolerability, with first participant dosing
  anticipated by the end of 2026 and initial safety data anticipated in the
  first half of 2027.
- Adjudication: do not record human efficacy or participant dosing as established
  from these sources. Registry recruitment status and sponsor-anticipated dosing
  are distinct facts. Trial authorization/designations are not marketing approval.
- Sources:
  - https://clinicaltrials.gov/study/NCT07148089
  - https://www.sec.gov/Archives/edgar/data/1707502/000119312526447178/sldb-20260930.htm

## CardiEval encoding rules

1. A sponsor-reported outcome can be source-verified without being independently
   adjudicated; keep source class explicit.
2. A trial's primary endpoint status must not be overwritten by a favourable
   secondary or subgroup result.
3. For subgroup analyses, preserve prespecification, interaction testing and
   multiplicity status as separate fields.
4. Use `not_reported` when the source does not disclose a value. Never impute zero.
5. Product potency, clinical efficacy, trial status and regulatory authorization
   are separate evidence domains.
