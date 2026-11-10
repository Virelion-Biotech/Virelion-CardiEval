"""Clinical evidence structures that preserve endpoint hierarchy and analysis population.

This module stores adjudicated trial facts; it does not infer efficacy from raw p-values
or convert company announcements into independent clinical evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

EndpointRole = Literal["primary", "key_secondary", "secondary", "exploratory", "safety"]
AnalysisPopulation = Literal["intention_to_treat", "modified_intention_to_treat", "per_protocol", "safety", "subgroup"]
EndpointResultStatus = Literal["met", "not_met", "inconclusive", "not_reported"]


@dataclass(frozen=True, slots=True)
class ClinicalEndpointResult:
    endpoint_name: str
    role: EndpointRole
    population: AnalysisPopulation
    status: EndpointResultStatus
    effect_estimate: float | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    denominator: int | None = None
    timepoint: str | None = None
    prespecified: bool | None = None
    multiplicity_adjusted: bool | None = None
    treatment_by_subgroup_interaction_p_value: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.endpoint_name.strip():
            raise ValueError("endpoint_name is required")
        if self.denominator is not None and self.denominator < 1:
            raise ValueError("denominator must be positive")
        if self.p_value is not None and not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be between 0 and 1")
        if self.treatment_by_subgroup_interaction_p_value is not None and not 0 <= self.treatment_by_subgroup_interaction_p_value <= 1:
            raise ValueError("interaction p-value must be between 0 and 1")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if low > high:
                raise ValueError("confidence interval lower bound exceeds upper bound")


@dataclass(frozen=True, slots=True)
class ClinicalTrialEvidence:
    trial_id: str
    product_id: str
    phase: str
    design: str
    comparator: str | None
    randomised: bool
    blinded: bool | None
    endpoints: tuple[ClinicalEndpointResult, ...]
    source_ids: tuple[str, ...]
    source_type: Literal["registry", "peer_reviewed_publication", "regulatory_document", "company_report", "other"]
    source_verified: bool = False
    follow_up: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(x.strip() for x in (self.trial_id, self.product_id, self.phase, self.design)):
            raise ValueError("trial identity, phase and design are required")
        if not self.endpoints:
            raise ValueError("at least one endpoint record is required")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        if not self.source_ids:
            raise ValueError("at least one source_id is required for provenance")
    
    @property
    def primary_endpoint_status(self) -> str:
        primary = [x for x in self.endpoints if x.role == "primary"]
        if not primary or all(x.status == "not_reported" for x in primary):
            return "not_reported"
        if any(x.status == "not_met" for x in primary):
            return "primary_endpoint_not_met"
        if all(x.status == "met" for x in primary):
            return "primary_endpoint_met"
        return "primary_endpoint_inconclusive"

    @property
    def subgroup_caveats(self) -> tuple[str, ...]:
        caveats = []
        for endpoint in self.endpoints:
            if endpoint.population == "subgroup":
                if endpoint.prespecified is False:
                    caveats.append(f"{endpoint.endpoint_name}: subgroup was not prespecified")
                if endpoint.multiplicity_adjusted is False:
                    caveats.append(f"{endpoint.endpoint_name}: multiplicity was not adjusted")
                if endpoint.prespecified is None:
                    caveats.append(f"{endpoint.endpoint_name}: prespecification status unknown")
                if endpoint.multiplicity_adjusted is None:
                    caveats.append(f"{endpoint.endpoint_name}: multiplicity status unknown")
                if endpoint.treatment_by_subgroup_interaction_p_value is None:
                    caveats.append(f"{endpoint.endpoint_name}: interaction test not reported")
        return tuple(caveats)

    def to_dict(self) -> dict:
        return asdict(self)
