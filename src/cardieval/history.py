"""Historical leaderboard comparison utilities."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .publication import LeaderboardSnapshot


class ModelMovement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    previous_rank: int | None = None
    current_rank: int | None = None
    rank_change: int | None = None
    previous_score: float | None = None
    current_score: float | None = None
    score_change: float | None = None
    status: str


class PublicationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_id: str
    benchmark_version: str
    task_id: str
    split: str
    primary_metric: str
    previous_snapshot_id: str
    current_snapshot_id: str
    movements: list[ModelMovement]


def snapshot_id(snapshot: LeaderboardSnapshot) -> str:
    from .provenance import canonical_json_hash

    return canonical_json_hash(snapshot.model_dump(mode="json"))


def compare_snapshots(previous: LeaderboardSnapshot, current: LeaderboardSnapshot) -> PublicationComparison:
    identities = (
        (previous.benchmark_id, previous.benchmark_version, previous.task_id, previous.split, previous.primary_metric),
        (current.benchmark_id, current.benchmark_version, current.task_id, current.split, current.primary_metric),
    )
    if identities[0] != identities[1]:
        raise ValueError("Snapshots must share benchmark, task, split, and primary metric")

    before = {entry.model_id: entry for entry in previous.leaderboard.entries}
    after = {entry.model_id: entry for entry in current.leaderboard.entries}
    movements: list[ModelMovement] = []
    for model_id in sorted(set(before) | set(after)):
        old = before.get(model_id)
        new = after.get(model_id)
        if old and new:
            movements.append(
                ModelMovement(
                    model_id=model_id,
                    previous_rank=old.rank,
                    current_rank=new.rank,
                    rank_change=old.rank - new.rank,
                    previous_score=old.score,
                    current_score=new.score,
                    score_change=new.score - old.score,
                    status="unchanged" if old.rank == new.rank else "moved",
                )
            )
        elif new:
            movements.append(ModelMovement(model_id=model_id, current_rank=new.rank, current_score=new.score, status="new"))
        else:
            movements.append(ModelMovement(model_id=model_id, previous_rank=old.rank, previous_score=old.score, status="removed"))

    return PublicationComparison(
        benchmark_id=current.benchmark_id,
        benchmark_version=current.benchmark_version,
        task_id=current.task_id,
        split=current.split,
        primary_metric=current.primary_metric,
        previous_snapshot_id=snapshot_id(previous),
        current_snapshot_id=snapshot_id(current),
        movements=movements,
    )


def load_snapshot(path: str | Path) -> LeaderboardSnapshot:
    return LeaderboardSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_comparison(comparison: PublicationComparison, path: str | Path) -> None:
    Path(path).write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
