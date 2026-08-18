from cardieval.benchmark_package import BenchmarkPackage, fingerprint_directory, validate_submission_against_package, verify_package_artifacts
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.registry import BenchmarkTask


def make_package():
    manifest = BenchmarkManifest(
        benchmark_id="bench",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="0" * 64,
    )
    task = BenchmarkTask(
        benchmark_id="bench",
        version="1",
        task_id="detect",
        task_type="binary_classification",
        allowed_metrics=["accuracy"],
        primary_metric="accuracy",
        primary_direction="higher_is_better",
        splits=["test"],
    )
    return BenchmarkPackage(benchmark_id="bench", version="1", manifest=manifest, tasks=[task])


def test_package_contract_and_submission_alignment():
    package = make_package()
    validate_submission_against_package(
        package,
        [PredictionRecord(sample_id="a", y_true=1, y_pred=1), PredictionRecord(sample_id="b", y_true=0, y_pred=0)],
        task_id="detect",
    )


def test_package_rejects_wrong_submission_set():
    package = make_package()
    try:
        validate_submission_against_package(
            package,
            [PredictionRecord(sample_id="a", y_true=1, y_pred=1)],
            task_id="detect",
        )
    except ValueError as exc:
        assert "sample set mismatch" in str(exc)
    else:
        raise AssertionError("expected sample-set mismatch")


def test_fingerprint_directory_and_verify(tmp_path):
    artifact = tmp_path / "labels.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    records = fingerprint_directory(tmp_path, kinds={"json": "manifest"})
    package = make_package().model_copy(update={"artifacts": records})
    assert verify_package_artifacts(package, tmp_path) == []
    artifact.write_text('{"ok": false}', encoding="utf-8")
    errors = verify_package_artifacts(package, tmp_path)
    assert any("sha256 mismatch" in error for error in errors)
