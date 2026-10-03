from cardieval.bundle import build_bundle
from cardieval.models import BenchmarkManifest, EvaluationReport, MetricResult


def make_inputs():
    manifest = BenchmarkManifest(
        benchmark_id="bench",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="0" * 64,
    )
    report = EvaluationReport(
        evaluator_version="0.6.0",
        benchmark_id="bench",
        benchmark_version="1",
        benchmark_sha256="0" * 64,
        task="binary_classification",
        split="test",
        model_id="model-a",
        task_id="task-1",
        primary_metric="accuracy",
        primary_value=1.0,
        primary_direction="higher_is_better",
        metrics=[MetricResult(name="accuracy", value=1.0, n=2, direction="higher_is_better")],
    )
    return manifest, report


def test_bundle_is_deterministic():
    manifest, report = make_inputs()
    first = build_bundle(manifest, report, task_id="task-1", submission_sha256="1" * 64)
    second = build_bundle(manifest, report, task_id="task-1", submission_sha256="1" * 64)
    assert first.bundle_id == second.bundle_id
    assert first.evaluation_fingerprint == second.evaluation_fingerprint


def test_bundle_rejects_mismatched_report():
    manifest, report = make_inputs()
    report = report.model_copy(update={"benchmark_version": "2"})
    try:
        build_bundle(manifest, report, task_id="task-1", submission_sha256="1" * 64)
    except ValueError as exc:
        assert "benchmark_version" in str(exc)
    else:
        raise AssertionError("expected mismatch to fail")


def test_bundle_rejects_mismatched_task():
    manifest, report = make_inputs()
    try:
        build_bundle(manifest, report, task_id="task-2", submission_sha256="1" * 64)
    except ValueError as exc:
        assert "task_id" in str(exc)
    else:
        raise AssertionError("expected task mismatch to fail")
