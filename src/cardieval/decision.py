"""Explicit statistical decision rules for model comparisons and release gates."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Decision = Literal["superior", "non_inferior", "inconclusive", "inferior"]


class ComparisonDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    metric: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better"]
    alpha: float = Field(gt=0, lt=1)
    ci_confidence: float = Field(default=0.95, gt=0, lt=1)
    ci_sidedness: Literal["two_sided", "lower", "upper"] = "two_sided"
    pvalue_hypothesis: Literal["equality", "superiority_margin", "non_inferiority_margin", "inferiority_margin"] | None = None
    pvalue_margin: float | None = Field(default=None, ge=0)
    margin: float = Field(ge=0)
    observed_difference: float
    ci_low: float
    ci_high: float
    adjusted_pvalue: float | None = Field(default=None, ge=0, le=1)
    decision: Decision
    rationale: str


class QualityGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    passed: bool
    severity: Literal["info", "warning", "error"]
    message: str


class ReleaseGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    gates: list[QualityGate]

    @property
    def errors(self) -> list[QualityGate]:
        return [gate for gate in self.gates if gate.severity == "error" and not gate.passed]


def decide_comparison(
    *,
    metric: str,
    direction: Literal["higher_is_better", "lower_is_better"],
    observed_difference: float,
    ci_low: float,
    ci_high: float,
    alpha: float = 0.05,
    margin: float = 0.0,
    adjusted_pvalue: float | None = None,
    ci_confidence: float = 0.95,
    ci_sidedness: Literal["two_sided", "lower", "upper"] = "two_sided",
    pvalue_hypothesis: Literal["equality", "superiority_margin", "non_inferiority_margin", "inferiority_margin"] = "equality",
    pvalue_margin: float | None = None,
) -> ComparisonDecision:
    """Apply a confidence-bound decision with explicit error-rate metadata.

    The difference is interpreted as model A minus model B after orienting the
    metric so positive values favor model A. The CI must clear the superiority
    or non-inferiority margin. Supplied p-values must target the candidate's
    null hypothesis; equality p-values cannot gate nonzero-margin inference.
    ``lower``/``upper`` identify one-sided bounds in the raw A-minus-B metric.
    Caller-supplied metadata must describe how bounds were actually computed.
    """
    if not math.isfinite(observed_difference):
        raise ValueError("observed_difference must be finite")
    if not math.isfinite(ci_low) or not math.isfinite(ci_high):
        raise ValueError("confidence interval bounds must be finite")
    if adjusted_pvalue is not None and not math.isfinite(adjusted_pvalue):
        raise ValueError("adjusted_pvalue must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be non-negative")
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("unknown metric direction")
    if not 0 < ci_confidence < 1:
        raise ValueError("ci_confidence must be in (0, 1)")
    if ci_sidedness not in {"two_sided", "lower", "upper"}:
        raise ValueError("unknown confidence interval sidedness")
    if ci_confidence < 1-alpha-1e-12:
        raise ValueError("CI confidence is insufficient for the declared alpha")
    if pvalue_hypothesis not in {"equality", "superiority_margin", "non_inferiority_margin", "inferiority_margin"}:
        raise ValueError("unknown p-value hypothesis")
    if pvalue_margin is not None and (not math.isfinite(pvalue_margin) or pvalue_margin < 0):
        raise ValueError("pvalue_margin must be finite and non-negative")
    if ci_low > ci_high:
        raise ValueError("ci_low must be <= ci_high")
    if adjusted_pvalue is not None and not 0 <= adjusted_pvalue <= 1:
        raise ValueError("adjusted_pvalue must be in [0, 1]")

    if adjusted_pvalue is not None and margin > 0 and pvalue_hypothesis == "equality":
        raise ValueError("equality p-value cannot gate inference about a nonzero margin")
    if adjusted_pvalue is not None and pvalue_hypothesis != "equality":
        if pvalue_margin is None or pvalue_margin != margin:
            raise ValueError("margin-specific p-value must declare the matching pvalue_margin")
    adjusted_ok = adjusted_pvalue is None or adjusted_pvalue < alpha
    if direction == "lower_is_better":
        oriented_low, oriented_high = -ci_high, -ci_low
        oriented_difference = -observed_difference
        oriented_sidedness = {"two_sided": "two_sided", "lower": "upper", "upper": "lower"}[ci_sidedness]
    else:
        oriented_low, oriented_high = ci_low, ci_high
        oriented_difference = observed_difference
        oriented_sidedness = ci_sidedness

    lower_available = oriented_sidedness in {"two_sided", "lower"}
    upper_available = oriented_sidedness in {"two_sided", "upper"}
    if lower_available and oriented_low > margin:
        decision: Decision = "superior"
        rationale = "The confidence interval clears the superiority margin in the favorable direction."
    elif lower_available and oriented_low > -margin:
        decision = "non_inferior"
        rationale = "The confidence interval stays above the non-inferiority boundary in the favorable orientation."
    elif upper_available and oriented_high < -margin:
        decision = "inferior"
        rationale = "The confidence interval lies beyond the adverse margin."
    else:
        decision = "inconclusive"
        rationale = "The interval and/or corrected significance evidence does not support a directional claim."

    if adjusted_pvalue is not None and decision != "inconclusive":
        required = {"superior": "superiority_margin", "non_inferior": "non_inferiority_margin", "inferior": "inferiority_margin"}[decision]
        equality_allowed = margin == 0 and decision in {"superior", "inferior"}
        if pvalue_hypothesis != required and not (pvalue_hypothesis == "equality" and equality_allowed):
            raise ValueError(f"p-value hypothesis must target {required} for this decision")
        if not adjusted_ok:
            decision = "inconclusive"
            rationale = "The confidence bound clears the margin, but its adjusted hypothesis-specific p-value fails alpha."

    if adjusted_pvalue is not None and adjusted_pvalue >= alpha:
        rationale += " The adjusted p-value does not meet alpha, so the claim is not statistically supported."

    return ComparisonDecision(
        metric=metric,
        direction=direction,
        alpha=alpha,
        ci_confidence=ci_confidence,
        ci_sidedness=oriented_sidedness,
        pvalue_hypothesis=pvalue_hypothesis if adjusted_pvalue is not None else None,
        pvalue_margin=pvalue_margin if adjusted_pvalue is not None else None,
        margin=margin,
        observed_difference=oriented_difference,
        ci_low=oriented_low,
        ci_high=oriented_high,
        adjusted_pvalue=adjusted_pvalue,
        decision=decision,
        rationale=rationale,
    )


def evaluate_release_gates(
    *,
    report_ok: bool,
    verification_errors: int = 0,
    subgroup_warnings: int = 0,
    required_primary_metric: bool = True,
    allow_warnings: bool = True,
) -> ReleaseGateReport:
    """Create a deterministic release-readiness report from evaluation facts."""
    if verification_errors < 0 or subgroup_warnings < 0:
        raise ValueError("gate error/warning counts must be non-negative")
    gates = [
        QualityGate(
            name="evaluation_report",
            passed=report_ok,
            severity="error",
            message="Evaluation report contains no hard errors." if report_ok else "Evaluation report contains errors.",
        ),
        QualityGate(
            name="artifact_integrity",
            passed=verification_errors == 0,
            severity="error",
            message="All release artifacts verified." if verification_errors == 0 else f"{verification_errors} artifact verification error(s).",
        ),
        QualityGate(
            name="primary_metric",
            passed=required_primary_metric,
            severity="error",
            message="Declared primary metric is present." if required_primary_metric else "Declared primary metric is missing.",
        ),
        QualityGate(
            name="subgroup_warnings",
            passed=subgroup_warnings == 0 or allow_warnings,
            severity="warning" if allow_warnings else "error",
            message="No subgroup warnings." if subgroup_warnings == 0 else f"{subgroup_warnings} subgroup warning(s).",
        ),
    ]
    return ReleaseGateReport(passed=all(gate.passed for gate in gates if gate.severity == "error"), gates=gates)
