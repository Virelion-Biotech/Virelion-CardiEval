"""Cross-benchmark model scorecards."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .publication import LeaderboardSnapshot


class BenchmarkScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    benchmark_version: str
    task_id: str
    split: str
    metric: str
    direction: str
    score: float
    rank: int


class ModelScorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    n_benchmarks: int = Field(ge=1)
    mean_normalized_score: float
    mean_rank: float
    benchmarks: list[BenchmarkScore]


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    task_family: str = "cross_benchmark"
    n_benchmarks: int = Field(ge=1)
    n_models: int = Field(ge=1)
    models: list[ModelScorecard]


def _normalize(score: float, entries: Sequence, direction: str) -> float:
    values = [entry.score for entry in entries]
    lo, hi = min(values), max(values)
    if hi == lo:
        return 1.0
    x = (score - lo) / (hi - lo)
    return x if direction == "higher_is_better" else 1.0 - x


def build_scorecard(snapshots: Sequence[LeaderboardSnapshot]) -> Scorecard:
    if not snapshots:
        raise ValueError("At least one leaderboard snapshot is required")
    seen_keys = set()
    per_model: dict[str, list[BenchmarkScore]] = defaultdict(list)
    for snapshot in snapshots:
        key = (snapshot.benchmark_id, snapshot.benchmark_version, snapshot.task_id, snapshot.split)
        if key in seen_keys:
            raise ValueError(f"duplicate benchmark snapshot: {key}")
        seen_keys.add(key)
        entries = snapshot.leaderboard.entries
        for entry in entries:
            per_model[entry.model_id].append(
                BenchmarkScore(
                    benchmark_id=snapshot.benchmark_id,
                    benchmark_version=snapshot.benchmark_version,
                    task_id=snapshot.task_id,
                    split=snapshot.split,
                    metric=snapshot.primary_metric,
                    direction=snapshot.primary_direction,
                    score=entry.score,
                    rank=entry.rank,
                )
            )

    scorecards: list[ModelScorecard] = []
    for model_id, benchmarks in sorted(per_model.items()):
        groups: dict[tuple[str, str, str, str], list[BenchmarkScore]] = defaultdict(list)
        for item in benchmarks:
            groups[(item.benchmark_id, item.benchmark_version, item.task_id, item.split)].append(item)
        normalized = []
        for snapshot in snapshots:
            key = (snapshot.benchmark_id, snapshot.benchmark_version, snapshot.task_id, snapshot.split)
            group = groups.get(key, [])
            if group:
                normalized.append(_normalize(group[0].score, snapshot.leaderboard.entries, snapshot.primary_direction))
        scorecards.append(
            ModelScorecard(
                model_id=model_id,
                n_benchmarks=len(normalized),
                mean_normalized_score=sum(normalized) / len(normalized),
                mean_rank=sum(item.rank for item in benchmarks) / len(benchmarks),
                benchmarks=benchmarks,
            )
        )

    scorecards.sort(key=lambda item: (-item.mean_normalized_score, item.mean_rank, item.model_id))
    return Scorecard(n_benchmarks=len(snapshots), n_models=len(scorecards), models=scorecards)
