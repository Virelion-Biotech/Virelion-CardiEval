import pytest

from cardieval.registry import BenchmarkTask, TaskRegistry


def test_registry_requires_primary_metric_contract():
    task = BenchmarkTask(
        benchmark_id="cardi-bench",
        version="1.0",
        task_id="challenge-classification",
        task_type="classification",
        allowed_metrics=["macro_f1", "auroc"],
        primary_metric="macro_f1",
        primary_direction="higher_is_better",
        splits=["validation", "test"],
    )
    registry = TaskRegistry([task])
    assert registry.get("cardi-bench", "1.0", "challenge-classification") == task
    assert registry.list() == [task]


def test_registry_rejects_duplicate_identity():
    task = BenchmarkTask(
        benchmark_id="b",
        version="1",
        task_id="t",
        task_type="regression",
        allowed_metrics=["mae"],
        primary_metric="mae",
        primary_direction="lower_is_better",
        splits=["test"],
    )
    registry = TaskRegistry([task])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(task)
