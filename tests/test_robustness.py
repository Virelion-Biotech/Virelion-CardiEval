from cardieval.evaluator import evaluate_submission
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.robustness import relative_drop, subgroup_robustness


def test_subgroups_are_reported_and_low_n_is_flagged() -> None:
    manifest = BenchmarkManifest(
        benchmark_id="demo",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b", "c", "d"],
        dataset_sha256="0" * 64,
    )
    records = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0, score=0.1, subgroup="group-a"),
        PredictionRecord(sample_id="b", y_true=1, y_pred=1, score=0.9, subgroup="group-a"),
        PredictionRecord(sample_id="c", y_true=0, y_pred=0, score=0.2, subgroup="group-b"),
        PredictionRecord(sample_id="d", y_true=1, y_pred=0, score=0.4, subgroup="group-b"),
    ]
    report = evaluate_submission(manifest, records, model_id="demo", subgroup_min_n=2)
    assert {x.subgroup for x in report.subgroups} == {"group-a", "group-b"}
    assert all(x.n == 2 for x in report.subgroups)
    assert report.benchmark_sha256 == "0" * 64
    assert any(m.name == "brier" for m in report.metrics)


def test_report_is_marked_for_tiny_subgroup() -> None:
    manifest = BenchmarkManifest(
        benchmark_id="demo",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="1" * 64,
    )
    records = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0, score=0.1, subgroup="tiny"),
        PredictionRecord(sample_id="b", y_true=1, y_pred=1, score=0.9),
    ]
    report = evaluate_submission(manifest, records, model_id="demo", subgroup_min_n=3)
    assert report.warnings
    assert "tiny" in report.warnings[0]


def test_subgroup_robustness_summary():
    summary = subgroup_robustness({"a": 0.9, "b": 0.7}, metric="macro_f1", direction="higher_is_better")
    assert summary.best == 0.9
    assert summary.worst == 0.7
    assert summary.range == 0.2


def test_relative_drop_accounts_for_metric_direction():
    assert relative_drop(0.8, 0.6, direction="higher_is_better") == 0.25
    assert relative_drop(0.2, 0.3, direction="lower_is_better") == 0.5
