"""Explicit statistical decision rules for model comparisons and release gates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Decision = Literal["superior", "non_inferior", "inconclusive", "inferior"]


class ComparisonDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    direction: Literal["higher_is_better", "lower_is_better"]
    alpha: float = Field(gt=0, lt=1)
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
) -> ComparisonDecision:
    """Apply a two-sided CI + optional adjusted-p-value decision rule.

    The difference is interpreted as model A minus model B after orienting the
    metric so positive values favor model A. The CI must clear the superiority
    or non-inferiority margin and the adjusted p-value must be below alpha when
    supplied.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if ci_low > ci_high:
        raise ValueError("ci_low must be <= ci_high")
    if adjusted_pvalue is not None and not 0 <= adjusted_pvalue <= 1:
        raise ValueError("adjusted_pvalue must be in [0, 1]")

    adjusted_ok = adjusted_pvalue is None or adjusted_pvalue < alpha
    if direction == "lower_is_better":
        oriented_low, oriented_high = -ci_high, -ci_low
        oriented_difference = -observed_difference
    else:
        oriented_low, oriented_high = ci_low, ci_high
        oriented_difference = observed_difference

    if oriented_low > margin and adjusted_ok:
        decision: Decision = "superior"
        rationale = "The confidence interval clears the superiority margin in the favorable direction."
    elif oriented_low >= -margin and adjusted_ok:
        decision = "non_inferior"
        rationale = "The confidence interval stays above the non-inferiority boundary in the favorable orientation."
    elif oriented_high < -margin and adjusted_ok:
        decision = "inferior"
        rationale = "The confidence interval lies beyond the adverse margin."
    else:
        decision = "inconclusive"
        rationale = "The interval and/or corrected significance evidence does not support a directional claim."

    if adjusted_pvalue is not None and adjusted_pvalue >= alpha:
        rationale += " The adjusted p-value does not meet alpha, so the claim is not statistically supported."

    return ComparisonDecision(
        metric=metric,
        direction=direction,
        alpha=alpha,
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
