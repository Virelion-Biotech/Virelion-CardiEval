"""Deterministic leaderboard aggregation for CardiEval reports."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from .models import EvaluationReport


class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rank: int = Field(ge=1)
    model_id: str = Field(min_length=1)
    score: float
    n_reports: int = Field(ge=1)
    benchmarks: list[str] = Field(default_factory=list)


class Leaderboard(BaseModel):
    """A ranked model table built only from compatible evaluation reports."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    benchmark_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    task_id: str | None = None
    split: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    direction: str
    entries: list[LeaderboardEntry]

    @property
    def n_models(self) -> int:
        return len(self.entries)


def _metric(report: EvaluationReport, metric_name: str) -> float:
    for metric in report.metrics:
        if metric.name == metric_name:
            if metric.direction == "informational":
                raise ValueError(f"metric {metric_name!r} cannot be used as a leaderboard score")
            return metric.value
    raise ValueError(f"Report for {report.model_id!r} has no metric {metric_name!r}")


def build_leaderboard(
    reports: Sequence[EvaluationReport],
    *,
    metric: str,
    direction: str,
    allow_repeated_models: bool = False,
) -> Leaderboard:
    """Rank models on a single benchmark/version/split and metric."""
    if not reports:
        raise ValueError("At least one evaluation report is required")
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("direction must be higher_is_better or lower_is_better")
    benchmark_id = reports[0].benchmark_id
    version = reports[0].benchmark_version
    split = reports[0].split
    dataset_hash = reports[0].benchmark_sha256
    reference_hash = reports[0].reference_sha256
    if reference_hash is None:
        raise ValueError("Leaderboard requires a reference hash; re-evaluate legacy reports")
    task_identity = (reports[0].task, reports[0].task_id)
    grouped: dict[str, list[float]] = {}
    benchmark_names: dict[str, set[str]] = {}
    for report in reports:
        if not report.ok:
            raise ValueError("Cannot rank a report containing evaluation errors")
        if report.benchmark_sha256 != dataset_hash:
            raise ValueError("All reports must use the same benchmark dataset hash")
        if report.reference_sha256 != reference_hash:
            raise ValueError("All reports must use the same reference samples, labels, units and metric configuration")
        if (report.task, report.task_id) != task_identity:
            raise ValueError("All reports must use the same task identity")
        if (report.benchmark_id, report.benchmark_version, report.split) != (
            benchmark_id,
            version,
            split,
        ):
            raise ValueError("All reports must use the same benchmark, version, and split")
        value = _metric(report, metric)
        report_metric = next(m for m in report.metrics if m.name == metric)
        if report_metric.direction != direction:
            raise ValueError(f"metric direction mismatch for {metric!r}")
        if not isfinite(value):
            raise ValueError(f"Non-finite metric value for model {report.model_id!r}")
        if report.model_id in grouped and not allow_repeated_models:
            raise ValueError(f"duplicate model_id in leaderboard: {report.model_id}")
        grouped.setdefault(report.model_id, []).append(value)
        benchmark_names.setdefault(report.model_id, set()).add(
            f"{report.benchmark_id}@{report.benchmark_version}"
        )

    scored = [
        (model_id, sum(values) / len(values), len(values))
        for model_id, values in grouped.items()
    ]
    scored.sort(key=lambda row: row[1], reverse=direction == "higher_is_better")

    entries: list[LeaderboardEntry] = []
    previous_score: float | None = None
    previous_rank = 0
    for index, (model_id, score, n_reports) in enumerate(scored, 1):
        if previous_score is not None and score == previous_score:
            rank = previous_rank
        else:
            rank = index
        entries.append(
            LeaderboardEntry(
                rank=rank,
                model_id=model_id,
                score=score,
                n_reports=n_reports,
                benchmarks=sorted(benchmark_names[model_id]),
            )
        )
        previous_score = score
        previous_rank = rank

    return Leaderboard(
        benchmark_id=benchmark_id,
        benchmark_version=version,
        benchmark_sha256=dataset_hash,
        reference_sha256=reference_hash,
        task_id=task_identity[1],
        split=split,
        metric=metric,
        direction=direction,
        entries=entries,
    )
