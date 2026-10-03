import pytest

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
        evaluator_version="1.4.1",
        benchmark_id="bench",
        benchmark_version="1",
        benchmark_sha256="0" * 64,
        task="binary_classification",
        split="test",
        model_id=model_id,
        task_id="primary",
        primary_metric="macro_f1",
        primary_value=score,
        primary_direction="higher_is_better",
        metrics=[MetricResult(name="macro_f1", value=score, n=20, direction="higher_is_better")],
    )
    built = SubmissionBundle(
        bundle_id="0" * 64,
        benchmark_id="bench",
        benchmark_version="1",
        task_id="primary",
        model_id=model_id,
        submission_sha256=("1" if model_id == "a" else "3") * 64,
        benchmark_sha256="0" * 64,
        evaluation_fingerprint="0" * 64,
        report=report,
    )
    fingerprint = built.expected_evaluation_fingerprint()
    return built.model_copy(update={
        "evaluation_fingerprint": fingerprint,
        "bundle_id": built.model_copy(update={"evaluation_fingerprint": fingerprint}).expected_bundle_id(),
    })


def test_publication_ranks_models_by_primary_metric():
    snapshot = publish_leaderboard([bundle("a", 0.8), bundle("b", 0.9)], task())
    assert snapshot.primary_metric == "macro_f1"
    assert [x.model_id for x in snapshot.leaderboard.entries] == ["b", "a"]
    assert snapshot.n_bundles == 2


def test_duplicate_models_are_rejected():
    with pytest.raises(ValueError, match="duplicate model_id"):
        ingest_bundles([bundle("a", 0.8), bundle("a", 0.9)], task())
