import pytest

from cardieval.benchmark_package import BenchmarkPackage
from cardieval.bridge import PredictionSubmission, build_submission_envelope, validate_envelope
from cardieval.bundle import build_bundle
from cardieval.integrity import ArtifactRecord, build_release_manifest, verify_release_manifest
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.multiple_testing import bonferroni
from cardieval.registry import BenchmarkTask
from cardieval.evaluator import evaluate_submission


def manifest(authoritative: bool = True) -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_id="bench",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b", "c", "d"],
        dataset_sha256="0" * 64,
        authoritative_labels={"a": 0, "b": 1, "c": 0, "d": 1} if authoritative else None,
        authoritative_subgroups={
            "a": "x", "b": "x", "c": "y", "d": "y"
        } if authoritative else None,
    )


def task(required: bool = True) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark_id="bench",
        version="1",
        task_id="detect",
        task_type="binary_classification",
        allowed_metrics=["accuracy", "macro_f1", "auroc"],
        primary_metric="auroc",
        primary_direction="higher_is_better",
        splits=["test"],
        requires_authoritative_labels=required,
    )


def package() -> BenchmarkPackage:
    return BenchmarkPackage(
        benchmark_id="bench",
        version="1",
        manifest=manifest(),
        tasks=[task()],
    )


def records_with_wrong_labels() -> list[PredictionRecord]:
    return [
        PredictionRecord(sample_id="a", y_true=1, y_pred=0, score=0.9),
        PredictionRecord(sample_id="b", y_true=0, y_pred=0, score=0.8),
        PredictionRecord(sample_id="c", y_true=1, y_pred=1, score=0.2),
        PredictionRecord(sample_id="d", y_true=0, y_pred=1, score=0.1),
    ]


def test_authoritative_labels_override_submission_labels():
    report = evaluate_submission(
        manifest(),
        records_with_wrong_labels(),
        model_id="m",
        task_contract=task(),
    )
    assert report.ground_truth_source == "benchmark_manifest"
    assert report.primary_metric == "auroc"
    assert report.primary_value == 1.0
    assert {item.subgroup for item in report.subgroups} == {"x", "y"}


def test_independent_task_requires_authoritative_labels():
    with pytest.raises(ValueError, match="authoritative benchmark labels"):
        task().validate_manifest(manifest(authoritative=False))

def test_protected_submission_can_omit_ground_truth():
    records = [
        PredictionRecord(sample_id="a", y_pred=0, score=0.1),
        PredictionRecord(sample_id="b", y_pred=1, score=0.9),
        PredictionRecord(sample_id="c", y_pred=0, score=0.2),
        PredictionRecord(sample_id="d", y_pred=1, score=0.8),
    ]
    report = evaluate_submission(manifest(), records, model_id="m", task_contract=task())
    assert report.ground_truth_source == "benchmark_manifest"
    assert report.primary_value == 1.0


def test_unprotected_submission_requires_ground_truth():
    records = [PredictionRecord(sample_id="a", y_pred=0)]
    with pytest.raises(ValueError, match="missing y_true"):
        evaluate_submission(
            manifest(authoritative=False).model_copy(update={"sample_ids": ["a"]}),
            records,
            model_id="m",
            task_contract=task(required=False),
        )


def test_submission_bundle_integrity_is_detected():
    m = manifest()
    t = task()
    report = evaluate_submission(
        m,
        [
            PredictionRecord(sample_id="a", y_true=0, y_pred=0, score=0.1),
            PredictionRecord(sample_id="b", y_true=1, y_pred=1, score=0.9),
            PredictionRecord(sample_id="c", y_true=0, y_pred=0, score=0.2),
            PredictionRecord(sample_id="d", y_true=1, y_pred=1, score=0.8),
        ],
        model_id="m",
        task_contract=t,
    )
    bundle = build_bundle(m, report, task_id="detect", submission_sha256="1" * 64)
    bundle.verify_integrity()
    tampered = bundle.model_copy(update={
        "report": bundle.report.model_copy(update={"primary_value": 0.123})
    })
    with pytest.raises(ValueError, match="evaluation_fingerprint"):
        tampered.verify_integrity()


def test_bridge_rejects_message_id_tampering():
    p = package()
    submission = PredictionSubmission(
        model_id="agent-model",
        task_id="detect",
        predictions=[
            PredictionRecord(sample_id="a", y_true=0, y_pred=0),
            PredictionRecord(sample_id="b", y_true=1, y_pred=1),
            PredictionRecord(sample_id="c", y_true=0, y_pred=0),
            PredictionRecord(sample_id="d", y_true=1, y_pred=1),
        ],
    )
    envelope = build_submission_envelope(p, submission, source_role="agent")
    tampered = envelope.model_copy(update={"message_id": "1" * 64})
    with pytest.raises(ValueError, match="message_id"):
        validate_envelope(tampered, p)


def test_bridge_rejects_incomplete_sample_set():
    p = package()
    submission = PredictionSubmission(
        model_id="agent-model",
        task_id="detect",
        predictions=[
            PredictionRecord(sample_id="a", y_true=0, y_pred=0),
            PredictionRecord(sample_id="b", y_true=1, y_pred=1),
        ],
    )
    with pytest.raises(ValueError, match="sample set mismatch"):
        build_submission_envelope(p, submission, source_role="agent")


def test_release_manifest_self_hash_and_path_checks():
    artifact = ArtifactRecord(path="report.json", sha256="0" * 64, kind="report", size_bytes=0)
    manifest = build_release_manifest(
        version="1",
        benchmark_id="bench",
        benchmark_version="1",
        task_id="task",
        publication_id="pub",
        artifacts=[artifact],
    )
    assert verify_release_manifest(manifest, ".")  # file is absent, but self-check is performed
    tampered = manifest.model_copy(update={"publication_id": "tampered"})
    errors = verify_release_manifest(tampered, ".")
    assert "release manifest self-hash mismatch" in errors


def test_multiple_testing_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        bonferroni([0.1, float("nan")])
