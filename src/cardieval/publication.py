"""Leaderboard publication and SubmissionBundle ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .bundle import SubmissionBundle
from .leaderboard import Leaderboard, build_leaderboard
from .registry import BenchmarkTask


class LeaderboardSnapshot(BaseModel):
    """Immutable publication snapshot derived from compatible submission bundles."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    primary_direction: str
    n_bundles: int = Field(ge=1)
    n_models: int = Field(ge=1)
    bundles: list[str]
    leaderboard: Leaderboard


def ingest_bundles(
    bundles: Iterable[SubmissionBundle],
    task: BenchmarkTask,
) -> list[SubmissionBundle]:
    """Validate bundle identity and primary-metric eligibility before publication."""
    accepted: list[SubmissionBundle] = []
    seen_models: set[str] = set()
    seen_bundles: set[str] = set()
    seen_splits: set[str] = set()
    for bundle in bundles:
        if bundle.benchmark_id != task.benchmark_id or bundle.benchmark_version != task.version:
            raise ValueError("bundle benchmark identity does not match task")
        if bundle.benchmark_sha256 != bundle.report.benchmark_sha256:
            raise ValueError("bundle benchmark_sha256 does not match report")
        task.validate_manifest(bundle.report.model_copy())
        if bundle.task_id != task.task_id:
            raise ValueError(f"bundle task_id {bundle.task_id!r} does not match {task.task_id!r}")
        metric_names = {metric.name for metric in bundle.report.metrics}
        if task.primary_metric not in metric_names:
            raise ValueError(
                f"bundle for {bundle.model_id!r} is missing primary metric {task.primary_metric!r}"
            )
        if bundle.bundle_id in seen_bundles:
            raise ValueError(f"duplicate bundle_id: {bundle.bundle_id}")
        if bundle.model_id in seen_models:
            raise ValueError(f"duplicate model_id in publication set: {bundle.model_id}")
        seen_bundles.add(bundle.bundle_id)
        seen_models.add(bundle.model_id)
        seen_splits.add(bundle.report.split)
        accepted.append(bundle)
    if not accepted:
        raise ValueError("At least one bundle is required")
    if len(seen_splits) != 1:
        raise ValueError("all bundles in a publication set must use the same split")
    return accepted


def publish_leaderboard(
    bundles: Iterable[SubmissionBundle],
    task: BenchmarkTask,
) -> LeaderboardSnapshot:
    """Create a publication-ready leaderboard from validated bundles."""
    accepted = ingest_bundles(bundles, task)
    reports = [bundle.report for bundle in accepted]
    leaderboard = build_leaderboard(
        reports,
        metric=task.primary_metric,
        direction=task.primary_direction,
    )
    return LeaderboardSnapshot(
        benchmark_id=task.benchmark_id,
        benchmark_version=task.version,
        task_id=task.task_id,
        split=accepted[0].report.split,
        primary_metric=task.primary_metric,
        primary_direction=task.primary_direction,
        n_bundles=len(accepted),
        n_models=len({bundle.model_id for bundle in accepted}),
        bundles=sorted(bundle.bundle_id for bundle in accepted),
        leaderboard=leaderboard,
    )


def load_bundle(path: str | Path) -> SubmissionBundle:
    """Load and validate a serialized SubmissionBundle."""
    return SubmissionBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_snapshot(snapshot: LeaderboardSnapshot, path: str | Path) -> None:
    """Write a publication snapshot as JSON."""
    Path(path).write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
