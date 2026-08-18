from cardieval.evaluator import evaluate_submission
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.registry import BenchmarkTask


def make_task() -> BenchmarkTask:
    return BenchmarkTask(
        benchmark_id="bench",
        version="1.0",
        task_id="detect",
        task_type="binary_classification",
        allowed_metrics=["accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "brier", "ece"],
        primary_metric="auroc",
        primary_direction="higher_is_better",
        splits=["test"],
    )


def make_inputs():
    manifest = BenchmarkManifest(
        benchmark_id="bench",
        version="1.0",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b", "c", "d"],
        dataset_sha256="0" * 64,
    )
    records = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0, score=0.1),
        PredictionRecord(sample_id="b", y_true=0, y_pred=0, score=0.2),
        PredictionRecord(sample_id="c", y_true=1, y_pred=1, score=0.8),
        PredictionRecord(sample_id="d", y_true=1, y_pred=1, score=0.9),
    ]
    return manifest, records


def test_task_contract_controls_primary_metric():
    manifest, records = make_inputs()
    report = evaluate_submission(manifest, records, model_id="m", task_contract=make_task())
    assert report.task_id == "detect"
    assert report.primary_metric == "auroc"
    assert report.primary_direction == "higher_is_better"
    assert report.primary_value == 1.0


def test_task_contract_rejects_wrong_split():
    manifest, records = make_inputs()
    bad = make_task().model_copy(update={"splits": ["validation"]})
    try:
        evaluate_submission(manifest, records, model_id="m", task_contract=bad)
    except ValueError as exc:
        assert "not permitted" in str(exc)
    else:
        raise AssertionError("expected split mismatch to fail")


def test_task_contract_rejects_wrong_task_type():
    manifest, records = make_inputs()
    bad = make_task().model_copy(update={"task_type": "regression"})
    try:
        evaluate_submission(manifest, records, model_id="m", task_contract=bad)
    except ValueError as exc:
        assert "task_type" in str(exc)
    else:
        raise AssertionError("expected task type mismatch to fail")


def test_task_contract_rejects_missing_primary_metric():
    manifest, records = make_inputs()
    bad = make_task().model_copy(update={"primary_metric": "made_up_metric", "allowed_metrics": ["accuracy"]})
    try:
        evaluate_submission(manifest, records, model_id="m", task_contract=bad)
    except ValueError as exc:
        assert "primary_metric" in str(exc)
    else:
        raise AssertionError("expected invalid contract to fail")
