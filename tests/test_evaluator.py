import pytest

from cardieval.evaluator import evaluate_submission, load_submission
from cardieval.models import BenchmarkManifest, PredictionRecord


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
