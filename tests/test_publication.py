from cardieval.bundle import SubmissionBundle
from cardieval.models import EvaluationReport, MetricResult
from cardieval.publication import ingest_bundles, publish_leaderboard
from cardieval.registry import BenchmarkTask


def task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark_id="bench",
        version="1",
        task_id="primary",
        task_type="binary_classification",
        allowed_metrics=["accuracy", "macro_f1"],
        primary_metric="macro_f1",
        primary_direction="higher_is_better",
        splits=["test"],
    )


def bundle(model_id: str, score: float) -> SubmissionBundle:
    report = EvaluationReport(
        evaluator_version="0.7.0",
        benchmark_id="bench",
        benchmark_version="1",
        benchmark_sha256="0" * 64,
        task="binary_classification",
        split="test",
        model_id=model_id,
        metrics=[MetricResult(name="macro_f1", value=score, n=20, direction="higher_is_better")],
    )
    return SubmissionBundle(
        bundle_id=f"bundle-{model_id}",
        benchmark_id="bench",
        benchmark_version="1",
        task_id="primary",
        model_id=model_id,
        submission_sha256="1" * 64,
        benchmark_sha256="0" * 64,
        evaluation_fingerprint="2" * 64,
        report=report,
    )


def test_publication_ranks_models_by_primary_metric():
    snapshot = publish_leaderboard([bundle("a", 0.8), bundle("b", 0.9)], task())
    assert snapshot.primary_metric == "macro_f1"
    assert [x.model_id for x in snapshot.leaderboard.entries] == ["b", "a"]
    assert snapshot.n_bundles == 2


def test_duplicate_models_are_rejected():
    try:
        ingest_bundles([bundle("a", 0.8), bundle("a", 0.9)], task())
    except ValueError as exc:
        assert "duplicate model_id" in str(exc)
    else:
        raise AssertionError("expected duplicate model rejection")
