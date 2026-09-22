"""Leaderboard publication and SubmissionBundle ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bundle import SubmissionBundle
from .leaderboard import Leaderboard, build_leaderboard
from .registry import BenchmarkTask


class LeaderboardSnapshot(BaseModel):
    """Immutable publication snapshot derived from compatible submission bundles."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.1"
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    primary_direction: str
    n_bundles: int = Field(ge=1)
    n_models: int = Field(ge=1)
    bundles: list[str] = Field(min_length=1)
    leaderboard: Leaderboard

    @model_validator(mode="after")
    def validate_consistency(self) -> "LeaderboardSnapshot":
        if self.primary_direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("primary_direction must be higher_is_better or lower_is_better")
        if self.n_bundles != len(self.bundles) or len(set(self.bundles)) != len(self.bundles):
            raise ValueError("snapshot bundle count must match unique bundle IDs")
        entries = self.leaderboard.entries
        if self.n_models != len(entries):
            raise ValueError("snapshot model count must match leaderboard entries")
        if len({entry.model_id for entry in entries}) != len(entries):
            raise ValueError("leaderboard model IDs must be unique")
        if (
            self.leaderboard.benchmark_id != self.benchmark_id
            or self.leaderboard.benchmark_version != self.benchmark_version
            or self.leaderboard.split != self.split
            or self.leaderboard.metric != self.primary_metric
            or self.leaderboard.direction != self.primary_direction
        ):
            raise ValueError("snapshot and leaderboard identities do not match")
        return self


def ingest_bundles(
    bundles: Iterable[SubmissionBundle],
    task: BenchmarkTask,
) -> list[SubmissionBundle]:
    """Validate bundle identity, integrity, and primary-metric eligibility before publication."""
    accepted: list[SubmissionBundle] = []
    seen_models: set[str] = set()
    seen_bundles: set[str] = set()
    seen_splits: set[str] = set()
    for bundle in bundles:
        bundle.verify_integrity()
        if bundle.benchmark_id != task.benchmark_id or bundle.benchmark_version != task.version:
            raise ValueError("bundle benchmark identity does not match task")
        if bundle.task_id != task.task_id:
            raise ValueError(f"bundle task_id {bundle.task_id!r} does not match {task.task_id!r}")
        task.validate_report_contract(bundle.report)
        if bundle.model_id != bundle.report.model_id:
            raise ValueError("bundle model_id does not match report model_id")
        if bundle.benchmark_sha256 != bundle.report.benchmark_sha256:
            raise ValueError("bundle benchmark_sha256 does not match report")
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
    """Load and validate a serialized SubmissionBundle, including integrity."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"bundle must be a regular file: {path}")
    bundle = SubmissionBundle.model_validate_json(path.read_text(encoding="utf-8"))
    bundle.verify_integrity()
    return bundle


def save_snapshot(snapshot: LeaderboardSnapshot, path: str | Path) -> None:
    """Write a validated publication snapshot as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
