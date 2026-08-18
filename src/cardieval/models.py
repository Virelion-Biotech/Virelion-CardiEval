"""Typed contracts used by CardiEval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["classification", "binary_classification", "regression", "ranking"]
SplitName = Literal["train", "validation", "test", "external"]


class PredictionRecord(BaseModel):
    """One model prediction tied to a stable benchmark sample ID."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    y_true: float | int | str
    y_pred: float | int | str
    score: float | None = None
    subgroup: str | None = None


class BenchmarkManifest(BaseModel):
    """Immutable description of the benchmark data presented to an evaluator."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task: TaskType
    split: SplitName
    sample_ids: list[str] = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_schema: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)

    def sample_set(self) -> set[str]:
        return set(self.sample_ids)


class MetricResult(BaseModel):
    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int
    direction: Literal["higher_is_better", "lower_is_better", "informational"]


class SubgroupResult(BaseModel):
    """Metric result scoped to a declared evaluation subgroup."""

    subgroup: str
    n: int
    metrics: list[MetricResult]
    warning: str | None = None


class ModelComparison(BaseModel):
    """Paired comparison of two models evaluated on the same samples."""

    metric: str
    model_a_score: float
    model_b_score: float
    difference: float
    permutation_pvalue: float
    wilcoxon_pvalue: float | None = None
    winner: Literal["model_a", "model_b", "tie", "undetermined"]
    n: int


class EvaluationReport(BaseModel):
    """Serializable, provenance-aware result produced by the evaluator."""

    schema_version: str = "0.3"
    evaluator_version: str
    benchmark_id: str
    benchmark_version: str
    benchmark_sha256: str
    task: TaskType
    split: SplitName
    model_id: str
    task_id: str | None = None
    primary_metric: str | None = None
    primary_value: float | None = None
    primary_direction: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: list[MetricResult]
    subgroups: list[SubgroupResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors
