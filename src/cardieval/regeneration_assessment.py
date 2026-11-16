"""Regeneration-versus-repair evidence assessment.

The evaluator preserves independent biological domains and emits a categorical
interpretation. It intentionally does not calculate a composite regeneration score.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

DomainName = Literal["cell_cycle_entry", "mitosis_cytokinesis", "cardiomyocyte_number_lineage", "functional_maturation", "ventricular_function_remodelling", "fibrosis_scar", "arrhythmia_safety", "off_target_toxicity"]
DomainStatus = Literal["direct_support", "surrogate_support", "no_support", "adverse_signal", "mixed", "not_reported", "not_assessed"]
Interpretation = Literal["regeneration_demonstrated_in_model", "repair_or_preservation_only", "mixed_evidence", "insufficient_evidence"]
DOMAIN_ORDER = ("cell_cycle_entry", "mitosis_cytokinesis", "cardiomyocyte_number_lineage", "functional_maturation", "ventricular_function_remodelling", "fibrosis_scar", "arrhythmia_safety", "off_target_toxicity")
REGENERATIVE_CORE = ("cell_cycle_entry", "mitosis_cytokinesis", "cardiomyocyte_number_lineage", "functional_maturation")

@dataclass(frozen=True, slots=True)
class RegenerationDomainEvidence:
    domain: DomainName
    status: DomainStatus
    assay: str | None = None
    model: str | None = None
    species: str | None = None
    comparator: str | None = None
    timepoint: str | None = None
    sample_size: int | None = None
    denominator: int | None = None
    effect_estimate: str | None = None
    uncertainty: str | None = None
    p_value: float | None = None
    prespecified: bool | None = None
    subgroup: bool = False
    multiplicity_adjusted: bool | None = None
    follow_up: str | None = None
    source_ids: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be positive")
        if self.denominator is not None and self.denominator < 1:
            raise ValueError("denominator must be positive")
        if self.p_value is not None and not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be between 0 and 1")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        if self.status not in ("not_assessed", "not_reported") and not self.source_ids:
            raise ValueError("observed domain evidence requires source_ids")

@dataclass(frozen=True, slots=True)
class RegenerationAssessment:
    assessment_id: str
    title: str
    model: str
    species: str
    source_ids: tuple[str, ...]
    domains: tuple[RegenerationDomainEvidence, ...]
    source_reported_interpretation: str = ""
    small_n_threshold: int = 10
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not all(x.strip() for x in (self.assessment_id, self.title, self.model, self.species)):
            raise ValueError("assessment_id, title, model and species are required")
        if self.schema_version != "1.0":
            raise ValueError("unsupported regeneration assessment schema_version")
        if self.small_n_threshold < 2:
            raise ValueError("small_n_threshold must be at least 2")
        if len(set(self.source_ids)) != len(self.source_ids) or not self.source_ids:
            raise ValueError("assessment requires unique, non-empty source_ids")
        by_name = {item.domain: item for item in self.domains}
        if len(by_name) != len(self.domains):
            raise ValueError("each assessment domain must appear exactly once")
        missing, extra = set(DOMAIN_ORDER) - set(by_name), set(by_name) - set(DOMAIN_ORDER)
        if missing or extra:
            raise ValueError(f"assessment domains mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        if not set().union(*(set(item.source_ids) for item in self.domains)) <= set(self.source_ids):
            raise ValueError("domain source_ids must be declared in assessment source_ids")

    def ordered_domains(self):
        by_name = {item.domain: item for item in self.domains}
        return tuple(by_name[name] for name in DOMAIN_ORDER)

    @property
    def interpretation(self) -> Interpretation:
        by_name = {item.domain: item for item in self.domains}
        core = [by_name[name].status for name in REGENERATIVE_CORE]
        direct_core = sum(status == "direct_support" for status in core)
        adverse = any(by_name[name].status == "adverse_signal" for name in ("arrhythmia_safety", "off_target_toxicity"))
        unresolved_core = any(status in ("not_assessed", "not_reported", "mixed", "surrogate_support") for status in core)
        if direct_core == len(REGENERATIVE_CORE) and not adverse and not unresolved_core:
            return "regeneration_demonstrated_in_model"
        repair_signal = any(by_name[name].status in ("direct_support", "surrogate_support") for name in ("ventricular_function_remodelling", "fibrosis_scar"))
        if direct_core >= 2 or (direct_core >= 1 and adverse):
            return "mixed_evidence"
        if repair_signal:
            return "repair_or_preservation_only"
        return "insufficient_evidence"

    def caveats(self) -> list[str]:
        warnings: list[str] = []
        for item in self.ordered_domains():
            prefix = item.domain
            if item.status in ("not_assessed", "not_reported"):
                warnings.append(f"{prefix}: domain is {item.status.replace('_', ' ')}")
            if item.denominator is None and item.status not in ("not_assessed", "not_reported"):
                warnings.append(f"{prefix}: denominator not reported")
            if item.sample_size is not None and item.sample_size < self.small_n_threshold:
                warnings.append(f"{prefix}: small sample (n={item.sample_size}; threshold={self.small_n_threshold})")
            if item.subgroup:
                warnings.append(f"{prefix}: subgroup-only evidence")
                if item.prespecified is False:
                    warnings.append(f"{prefix}: subgroup was not prespecified")
                elif item.prespecified is None:
                    warnings.append(f"{prefix}: subgroup prespecification unknown")
                if item.multiplicity_adjusted is False:
                    warnings.append(f"{prefix}: multiplicity was not adjusted")
                elif item.multiplicity_adjusted is None:
                    warnings.append(f"{prefix}: multiplicity handling unknown")
            if item.p_value is not None and item.p_value < 0.05 and item.multiplicity_adjusted is not True:
                warnings.append(f"{prefix}: nominal p-value below 0.05 without confirmed multiplicity adjustment")
            if item.domain in ("arrhythmia_safety", "off_target_toxicity") and item.follow_up is None:
                warnings.append(f"{prefix}: safety follow-up duration not reported")
        if self.interpretation != "regeneration_demonstrated_in_model":
            warnings.append("classification does not meet the complete four-domain regenerative evidence threshold")
        warnings.append("No composite score is calculated; domains must be interpreted independently.")
        warnings.append("This is a preclinical evidence assessment, not a patient-level or treatment recommendation.")
        return warnings

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "assessment_id": self.assessment_id,
            "title": self.title, "model": self.model, "species": self.species,
            "source_ids": list(self.source_ids),
            "source_reported_interpretation": self.source_reported_interpretation,
            "small_n_threshold": self.small_n_threshold, "interpretation": self.interpretation,
            "domains": [asdict(item) for item in self.ordered_domains()], "caveats": self.caveats(),
        }

def load_regeneration_assessment(path: str | Path) -> RegenerationAssessment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported regeneration assessment schema_version")
    domains = tuple(RegenerationDomainEvidence(**{**item, "source_ids": tuple(item.get("source_ids", ()))}) for item in payload["domains"])
    return RegenerationAssessment(
        assessment_id=payload["assessment_id"], title=payload["title"], model=payload["model"],
        species=payload["species"], source_ids=tuple(payload["source_ids"]), domains=domains,
        source_reported_interpretation=payload.get("source_reported_interpretation", ""),
        small_n_threshold=payload.get("small_n_threshold", 10), schema_version=payload["schema_version"],
    )

def write_regeneration_report(assessment: RegenerationAssessment, path: str | Path) -> dict:
    payload = assessment.to_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
