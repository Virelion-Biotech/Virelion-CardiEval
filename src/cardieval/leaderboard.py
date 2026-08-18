"""Deterministic leaderboard aggregation for CardiEval reports."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field

from .models import EvaluationReport


class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    model_id: str = Field(min_length=1)
    score: float
    n_reports: int = Field(ge=1)
    benchmarks: list[str] = Field(default_factory=list)


class Leaderboard(BaseModel):
    """A ranked model table built only from compatible evaluation reports."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    entries: list[LeaderboardEntry]


def _metric(report: EvaluationReport, metric_name: str) -> float:
    for metric in report.metrics:
        if metric.name == metric_name:
            return metric.value
    raise ValueError(f"Report for {report.model_id!r} has no metric {metric_name!r}")


def build_leaderboard(
    reports: Sequence[EvaluationReport],
    *,
    metric: str,
    direction: str,
) -> Leaderboard:
    """Rank models on a single benchmark/version/split and metric.

    Reports must all belong to the same benchmark/version/split. A model may have
    multiple reports; its score is the arithmetic mean across those reports.
    """
    if not reports:
        raise ValueError("At least one evaluation report is required")
    benchmark_id = reports[0].benchmark_id
    version = reports[0].benchmark_version
    split = reports[0].split
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise ValueError("direction must be higher_is_better or lower_is_better")

    grouped: dict[str, list[float]] = {}
    benchmark_names: dict[str, set[str]] = {}
    for report in reports:
        if (report.benchmark_id, report.benchmark_version, report.split) != (
            benchmark_id,
            version,
            split,
        ):
            raise ValueError("All reports must use the same benchmark, version, and split")
        value = _metric(report, metric)
        if not isfinite(value):
            raise ValueError(f"Non-finite metric value for model {report.model_id!r}")
        grouped.setdefault(report.model_id, []).append(value)
        benchmark_names.setdefault(report.model_id, set()).add(
            f"{report.benchmark_id}@{report.benchmark_version}"
        )

    scored = [(model_id, sum(values) / len(values), len(values)) for model_id, values in grouped.items()]
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
        split=split,
        metric=metric,
        direction=direction,
        entries=entries,
    )
