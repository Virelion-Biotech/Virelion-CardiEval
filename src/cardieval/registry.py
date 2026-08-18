"""Versioned benchmark/task registry for CardiEval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import BenchmarkManifest, SplitName, TaskType


class BenchmarkTask(BaseModel):
    """Public definition of a benchmark task and its scoring contract."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_type: TaskType
    allowed_metrics: list[str] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    primary_direction: str
    splits: list[SplitName] = Field(min_length=1)
    description: str = ""

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
