"""Historical leaderboard comparison and integrity-aware publication utilities."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .provenance import canonical_json_hash
from .publication import LeaderboardSnapshot


class LeaderboardDelta(BaseModel):
    """Change in a model's published rank/score between two snapshots."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    previous_rank: int | None = None
    current_rank: int | None = None
    rank_change: int | None = None
    previous_score: float | None = None
    current_score: float | None = None
    score_change: float | None = None


class PublicationComparison(BaseModel):
    """Machine-readable comparison between two compatible leaderboard snapshots."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_id: str
    benchmark_version: str
    task_id: str
    split: str
    primary_metric: str
    primary_direction: str
    previous_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    deltas: list[LeaderboardDelta]

    @property
    def changed_models(self) -> int:
        return sum(delta.rank_change not in (None, 0) or delta.score_change not in (None, 0.0) for delta in self.deltas)


def snapshot_hash(snapshot: LeaderboardSnapshot) -> str:
    """Return a deterministic integrity hash for a publication snapshot."""
    return canonical_json_hash(snapshot.model_dump(mode="json"))


def compare_snapshots(
    previous: LeaderboardSnapshot,
    current: LeaderboardSnapshot,
) -> PublicationComparison:
    """Compare two snapshots only when their publication contract is compatible."""
    identity = (
        "benchmark_id",
        "benchmark_version",
        "task_id",
        "split",
        "primary_metric",
        "primary_direction",
    )
    for field in identity:
        if getattr(previous, field) != getattr(current, field):
            raise ValueError(f"snapshot {field} does not match")

    old = {entry.model_id: entry for entry in previous.leaderboard.entries}
    new = {entry.model_id: entry for entry in current.leaderboard.entries}
    deltas: list[LeaderboardDelta] = []
    for model_id in sorted(set(old) | set(new)):
        old_entry = old.get(model_id)
        new_entry = new.get(model_id)
        old_rank = old_entry.rank if old_entry else None
        new_rank = new_entry.rank if new_entry else None
        old_score = old_entry.score if old_entry else None
        new_score = new_entry.score if new_entry else None
        deltas.append(
            LeaderboardDelta(
                model_id=model_id,
                previous_rank=old_rank,
                current_rank=new_rank,
                rank_change=(old_rank - new_rank) if old_rank is not None and new_rank is not None else None,
                previous_score=old_score,
                current_score=new_score,
                score_change=(new_score - old_score) if old_score is not None and new_score is not None else None,
            )
        )

    return PublicationComparison(
        benchmark_id=current.benchmark_id,
        benchmark_version=current.benchmark_version,
        task_id=current.task_id,
        split=current.split,
        primary_metric=current.primary_metric,
        primary_direction=current.primary_direction,
        previous_snapshot_hash=snapshot_hash(previous),
        current_snapshot_hash=snapshot_hash(current),
        deltas=deltas,
    )


def load_snapshot(path: str | Path) -> LeaderboardSnapshot:
    """Load and validate a serialized leaderboard snapshot."""
    return LeaderboardSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_comparison(comparison: PublicationComparison, path: str | Path) -> None:
    """Write a publication comparison as JSON."""
    Path(path).write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
