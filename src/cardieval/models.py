"""Typed contracts used by CardiEval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["classification", "binary_classification", "regression", "ranking"]
SplitName = Literal["train", "validation", "test", "external"]


class PredictionRecord(BaseModel):
    """One model prediction tied to a stable benchmark sample ID.

    y_true is optional so protected benchmark submissions can omit reference
    labels entirely. When a task does not supply evaluator-controlled labels,
    evaluation requires y_true to be present.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sample_id: str = Field(min_length=1)
    y_true: float | int | str | None = None
    y_pred: float | int | str
    score: float | None = None
    subgroup: str | None = None


class BenchmarkManifest(BaseModel):
    """Immutable description of the benchmark data presented to an evaluator.

    authoritative_labels and authoritative_subgroups are optional so existing
    demo/self-contained workflows remain valid. For independent evaluation,
    keep evaluator-controlled labels protected and set
    BenchmarkTask.requires_authoritative_labels=True.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task: TaskType
    split: SplitName
    sample_ids: list[str] = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_schema: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    authoritative_labels: dict[str, float | int | str] | None = None
    authoritative_subgroups: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_integrity(self) -> "BenchmarkManifest":
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Benchmark manifest contains duplicate sample IDs")
        if any(not sample_id.strip() for sample_id in self.sample_ids):
            raise ValueError("Benchmark sample IDs must not be blank")
        expected = set(self.sample_ids)
        if self.authoritative_labels is not None and set(self.authoritative_labels) != expected:
            raise ValueError("authoritative_labels must contain exactly the benchmark sample IDs")
        if self.authoritative_subgroups is not None and set(self.authoritative_subgroups) != expected:
            raise ValueError("authoritative_subgroups must contain exactly the benchmark sample IDs")
        return self

    def sample_set(self) -> set[str]:
        return set(self.sample_ids)


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = Field(ge=1)
    direction: Literal["higher_is_better", "lower_is_better", "informational"]


class SubgroupResult(BaseModel):
    """Metric result scoped to a declared evaluation subgroup."""

    model_config = ConfigDict(extra="forbid")

    subgroup: str = Field(min_length=1)
    n: int = Field(ge=1)
    metrics: list[MetricResult]
    warning: str | None = None


class ModelComparison(BaseModel):
    """Paired comparison of two models evaluated on the same samples."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    metric: str = Field(min_length=1)
    model_a_score: float
    model_b_score: float
    difference: float
    permutation_pvalue: float = Field(ge=0, le=1)
    wilcoxon_pvalue: float | None = Field(default=None, ge=0, le=1)
    winner: Literal["model_a", "model_b", "tie", "undetermined"]
    n: int = Field(ge=2)


class EvaluationReport(BaseModel):
    """Serializable, provenance-aware result produced by the evaluator."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "0.4"
    evaluator_version: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: TaskType
    split: SplitName
    model_id: str = Field(min_length=1)
    task_id: str | None = None
    primary_metric: str | None = None
    primary_value: float | None = None
    primary_direction: str | None = None
    ground_truth_source: Literal["benchmark_manifest", "submission"] = "submission"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: list[MetricResult]
    subgroups: list[SubgroupResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors
