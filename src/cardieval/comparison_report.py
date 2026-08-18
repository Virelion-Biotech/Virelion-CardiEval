"""Serializable, audit-friendly model comparison reports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .comparison import compare_predictions
from .provenance import evaluation_fingerprint


class ComparisonReport(BaseModel):
    """Machine-readable comparison artifact for two models on one benchmark."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    model_a: str = Field(min_length=1)
    model_b: str = Field(min_length=1)
    comparison: dict
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_comparison_report(
    y_true,
    pred_a,
    pred_b,
    metric,
    *,
    metric_name: str,
    direction: str,
    benchmark_id: str,
    benchmark_version: str,
    dataset_sha256: str,
    evaluator_version: str,
    model_a: str,
    model_b: str,
    n_resamples: int = 5000,
    seed: int = 0,
) -> ComparisonReport:
    result = compare_predictions(
        y_true,
        pred_a,
        pred_b,
        metric,
        metric_name=metric_name,
        n_resamples=n_resamples,
        seed=seed,
    )
    fingerprint = evaluation_fingerprint(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        dataset_sha256=dataset_sha256,
        evaluator_version=evaluator_version,
        model_id=f"{model_a}|{model_b}",
        config={"metric": metric_name, "direction": direction, "n_resamples": n_resamples, "seed": seed},
    )
    return ComparisonReport(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        metric=metric_name,
        direction=direction,
        model_a=model_a,
        model_b=model_b,
        comparison=result.model_dump(),
        fingerprint=fingerprint,
    )
