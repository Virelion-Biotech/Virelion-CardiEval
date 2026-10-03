import json
from pathlib import Path

from jsonschema import Draft202012Validator

from cardieval.benchmark_package import BenchmarkPackage
from cardieval.bridge import PredictionSubmission, build_submission_envelope
from cardieval.bundle import build_bundle
from cardieval.integrity import ArtifactRecord, build_release_manifest
from cardieval.leaderboard import Leaderboard, LeaderboardEntry
from cardieval.models import BenchmarkManifest, EvaluationReport, MetricResult, PredictionRecord
from cardieval.publication import LeaderboardSnapshot
from cardieval.run_manifest import build_run_manifest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"


def _validate(schema_name: str, instance: dict) -> None:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(instance)


def make_manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_id="schema-bench",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="0" * 64,
    )


def make_task():
    from cardieval.registry import BenchmarkTask

    return BenchmarkTask(
        benchmark_id="schema-bench",
        version="1",
        task_id="detect",
        task_type="binary_classification",
        allowed_metrics=["accuracy"],
        primary_metric="accuracy",
        primary_direction="higher_is_better",
        splits=["test"],
    )


def test_all_checked_in_schemas_are_valid():
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_model_artifacts_validate_against_schemas(tmp_path):
    manifest = make_manifest()
    task = make_task()
    package = BenchmarkPackage(
        benchmark_id="schema-bench",
        version="1",
        manifest=manifest,
        tasks=[task],
    )
    _validate("benchmark-package.schema.json", package.model_dump(mode="json"))

    submission = [
        PredictionRecord(sample_id="a", y_true=0, y_pred=0),
        PredictionRecord(sample_id="b", y_true=1, y_pred=1),
    ]
    report = EvaluationReport(
        evaluator_version="1.4.1",
        benchmark_id="schema-bench",
        benchmark_version="1",
        benchmark_sha256="0" * 64,
        task="binary_classification",
        split="test",
        model_id="model-a",
        task_id="detect",
        primary_metric="accuracy",
        primary_value=1.0,
        primary_direction="higher_is_better",
        metrics=[MetricResult(name="accuracy", value=1.0, n=2, direction="higher_is_better")],
    )
    _validate("evaluation-report.schema.json", report.model_dump(mode="json"))

    bundle = build_bundle(
        manifest,
        report,
        task_id="detect",
        submission_sha256="1" * 64,
    )
    _validate("submission-bundle.schema.json", bundle.model_dump(mode="json"))

    leaderboard = Leaderboard(
        benchmark_id="schema-bench",
        benchmark_version="1",
        split="test",
        metric="accuracy",
        direction="higher_is_better",
        entries=[
            LeaderboardEntry(
                rank=1,
                model_id="model-a",
                score=1.0,
                n_reports=1,
                benchmarks=["schema-bench@1"],
            )
        ],
    )
    snapshot = LeaderboardSnapshot(
        benchmark_id="schema-bench",
        benchmark_version="1",
        task_id="detect",
        split="test",
        primary_metric="accuracy",
        primary_direction="higher_is_better",
        n_bundles=1,
        n_models=1,
        bundles=[bundle.bundle_id],
        leaderboard=leaderboard,
    )
    _validate("leaderboard_snapshot.schema.json", snapshot.model_dump(mode="json"))

    envelope = build_submission_envelope(
        package,
        PredictionSubmission(
            model_id="bridge-model",
            task_id="detect",
            predictions=submission,
        ),
        source_role="agent",
    )
    _validate("bridge-envelope.schema.json", envelope.model_dump(mode="json"))

    report_path = tmp_path / "report.json"
    report_path.write_text("ok", encoding="utf-8")
    artifact = ArtifactRecord(
        path="report.json",
        sha256="0" * 64,
        kind="report",
        size_bytes=2,
    )
    release = build_release_manifest(
        version="1.0.0",
        benchmark_id="schema-bench",
        benchmark_version="1",
        task_id="detect",
        publication_id="publication-1",
        artifacts=[artifact],
    )
    _validate("release_manifest.schema.json", release.model_dump(mode="json"))

    run = build_run_manifest(
        benchmark_id="schema-bench",
        benchmark_version="1",
        task_id="detect",
        model_id="model-a",
        benchmark_package_sha256="2" * 64,
        submission_sha256="3" * 64,
        evaluation_fingerprint=bundle.evaluation_fingerprint,
        bundle_id=bundle.bundle_id,
        report_path="report.json",
        bundle_path="bundle.json",
    )
    _validate("evaluation-run-manifest.schema.json", run.model_dump(mode="json"))
