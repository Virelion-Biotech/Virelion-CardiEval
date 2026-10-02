import pytest

from cardieval.evaluator import evaluate_submission, load_submission
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.registry import BenchmarkTask


def manifest():
    return BenchmarkManifest(
        benchmark_id="demo",
        version="0.1.0",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b", "c", "d"],
        dataset_sha256="0" * 64,
    )


def test_evaluation_requires_exact_sample_set():
    with pytest.raises(ValueError, match="missing"):
        evaluate_submission(
            manifest(),
            [PredictionRecord(sample_id="a", y_true=0, y_pred=0)],
            model_id="demo-model",
        )


def test_evaluation_produces_report():
    records = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0, score=0.1),
        PredictionRecord(sample_id="b", y_true=1, y_pred=1, score=0.9),
        PredictionRecord(sample_id="c", y_true=0, y_pred=0, score=0.2),
        PredictionRecord(sample_id="d", y_true=1, y_pred=0, score=0.4),
    ]
    report = evaluate_submission(manifest(), records, model_id="demo-model")
    assert report.ok
    assert {m.name for m in report.metrics} >= {"accuracy", "balanced_accuracy", "macro_f1"}


def test_duplicate_jsonl_is_rejected(tmp_path):
    path = tmp_path / "submission.jsonl"
    path.write_text(
        '{"sample_id":"a","y_true":0,"y_pred":0}\n'
        '{"sample_id":"a","y_true":0,"y_pred":0}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_submission(path)



def test_single_class_fold_omits_undefined_secondary_metric_with_warning():
    single_class_manifest = BenchmarkManifest(
        benchmark_id="single",
        version="1.0",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b", "c", "d"],
        dataset_sha256="1" * 64,
        authoritative_labels={"a": 0, "b": 0, "c": 0, "d": 0},
    )
    records = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0, score=0.1),
        PredictionRecord(sample_id="b", y_true=0, y_pred=0, score=0.2),
        PredictionRecord(sample_id="c", y_true=0, y_pred=0, score=0.3),
        PredictionRecord(sample_id="d", y_true=0, y_pred=0, score=0.4),
    ]
    task = BenchmarkTask(
        benchmark_id="single",
        version="1.0",
        task_id="single-class-secondary",
        task_type="binary_classification",
        allowed_metrics=["macro_f1", "balanced_accuracy"],
        primary_metric="macro_f1",
        primary_direction="higher_is_better",
        splits=["test"],
        requires_authoritative_labels=True,
    )
    report = evaluate_submission(
        single_class_manifest,
        records,
        model_id="model",
        task_contract=task,
    )
    assert report.ok
    assert {metric.name for metric in report.metrics} == {"macro_f1"}
    assert any(
        "balanced_accuracy" in warning and "omitted" in warning
        for warning in report.warnings
    )


def test_single_class_fold_fails_if_undefined_metric_is_primary():
    single_class_manifest = BenchmarkManifest(
        benchmark_id="single-primary",
        version="1.0",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="2" * 64,
        authoritative_labels={"a": 0, "b": 0},
    )
    records = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0),
        PredictionRecord(sample_id="b", y_true=0, y_pred=0),
    ]
    task = BenchmarkTask(
        benchmark_id="single-primary",
        version="1.0",
        task_id="single-class-primary",
        task_type="binary_classification",
        allowed_metrics=["balanced_accuracy"],
        primary_metric="balanced_accuracy",
        primary_direction="higher_is_better",
        splits=["test"],
        requires_authoritative_labels=True,
    )
    with pytest.raises(ValueError, match="Primary metric .* was not produced"):
        evaluate_submission(
            single_class_manifest,
            records,
            model_id="model",
            task_contract=task,
        )
