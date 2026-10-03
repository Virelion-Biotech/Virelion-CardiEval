"""Versioned benchmark/task registry for CardiEval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .metrics import METRIC_DIRECTIONS
from .models import BenchmarkManifest, EvaluationReport, SplitName, TaskType


_SUPPORTED_METRICS = {
    "classification": {"accuracy", "balanced_accuracy", "macro_f1"},
    "binary_classification": {
        "accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "brier", "ece",
        "sensitivity", "specificity", "positive_predictive_value",
        "negative_predictive_value", "matthews_correlation", "cohen_kappa",
    },
    "regression": {"mae", "rmse"},
    "ranking": {"mrr", "hit_rate@10", "ndcg@10"},
}


class BenchmarkTask(BaseModel):
    """Public definition of a benchmark task and its scoring contract."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_type: TaskType
    allowed_metrics: list[str] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    primary_direction: str
    splits: list[SplitName] = Field(min_length=1)
    description: str = ""
    requires_authoritative_labels: bool = False

    @model_validator(mode="after")
    def validate_contract(self) -> "BenchmarkTask":
        if self.primary_metric not in self.allowed_metrics:
            raise ValueError("primary_metric must be listed in allowed_metrics")
        if self.primary_direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("primary_direction must be higher_is_better or lower_is_better")
        if len(set(self.allowed_metrics)) != len(self.allowed_metrics):
            raise ValueError("allowed_metrics must not contain duplicates")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must not contain duplicates")
        unsupported = sorted(set(self.allowed_metrics) - _SUPPORTED_METRICS[self.task_type])
        if unsupported:
            raise ValueError(
                f"metrics {unsupported} are not supported for task_type {self.task_type!r}"
            )
        expected_direction = METRIC_DIRECTIONS.get(self.primary_metric)
        if expected_direction and expected_direction != self.primary_direction:
            raise ValueError(
                f"primary_direction {self.primary_direction!r} does not match "
                f"metric {self.primary_metric!r} direction {expected_direction!r}"
            )
        return self

    def validate_manifest(self, manifest: BenchmarkManifest) -> None:
        """Ensure a benchmark manifest is exactly compatible with this task."""
        if manifest.benchmark_id != self.benchmark_id:
            raise ValueError("task benchmark_id does not match manifest")
        if manifest.version != self.version:
            raise ValueError("task version does not match manifest")
        if manifest.task != self.task_type:
            raise ValueError("task_type does not match manifest task")
        if manifest.split not in self.splits:
            raise ValueError(f"split {manifest.split!r} is not permitted by task {self.task_id!r}")
        if self.requires_authoritative_labels and manifest.authoritative_labels is None:
            raise ValueError(
                "task requires authoritative benchmark labels, but the manifest provides none"
            )

    def validate_report_contract(self, report: EvaluationReport) -> None:
        """Validate a completed evaluation report against this task contract."""
        if report.benchmark_id != self.benchmark_id:
            raise ValueError("report benchmark_id does not match task")
        if report.benchmark_version != self.version:
            raise ValueError("report benchmark_version does not match task")
        if report.task != self.task_type:
            raise ValueError("report task_type does not match task")
        if report.split not in self.splits:
            raise ValueError(f"report split {report.split!r} is not permitted by task")
        if report.task_id != self.task_id:
            raise ValueError("report task_id does not match task")
        if report.primary_metric != self.primary_metric:
            raise ValueError("report primary_metric does not match task")
        if report.primary_direction != self.primary_direction:
            raise ValueError("report primary_direction does not match task")
        names = {metric.name for metric in report.metrics}
        disallowed = sorted(names - set(self.allowed_metrics))
        if disallowed:
            raise ValueError(f"report contains metrics not allowed by task: {disallowed}")
        if self.primary_metric not in names or report.primary_value is None:
            raise ValueError("report is missing the declared primary metric")
        primary = next(metric for metric in report.metrics if metric.name == self.primary_metric)
        if report.primary_value != primary.value:
            raise ValueError("report primary_value does not match the primary metric value")
        if self.requires_authoritative_labels and report.ground_truth_source != "benchmark_manifest":
            raise ValueError("independent task requires evaluator-controlled ground truth")
        if not report.ok:
            raise ValueError("report contains evaluation errors")


class TaskRegistry:
    """In-memory registry with exact benchmark/version/task identity."""

    def __init__(self, tasks: list[BenchmarkTask] | None = None) -> None:
        self._tasks: dict[tuple[str, str, str], BenchmarkTask] = {}
        for task in tasks or []:
            self.register(task)

    def register(self, task: BenchmarkTask) -> None:
        key = (task.benchmark_id, task.version, task.task_id)
        if key in self._tasks:
            raise ValueError(f"Task already registered: {key}")
        self._tasks[key] = task

    def get(self, benchmark_id: str, version: str, task_id: str) -> BenchmarkTask:
        key = (benchmark_id, version, task_id)
        try:
            return self._tasks[key]
        except KeyError as exc:
            raise KeyError(f"Unknown benchmark task: {key}") from exc

    def list(self) -> list[BenchmarkTask]:
        return [self._tasks[key] for key in sorted(self._tasks)]
