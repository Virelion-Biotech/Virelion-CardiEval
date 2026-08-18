"""End-to-end CardiBench -> CardiEval evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from .benchmark_package import load_package, validate_submission_against_package, verify_package_artifacts
from .bundle import build_bundle, save_bundle
from .evaluator import evaluate_submission, load_submission, save_report, sha256_file
from .provenance import canonical_json_hash
from .run_manifest import EvaluationRunManifest, build_run_manifest, save_run_manifest


def run_evaluation(
    *,
    package_path: str | Path,
    package_root: str | Path,
    submission_path: str | Path,
    model_id: str,
    task_id: str,
    report_path: str | Path,
    bundle_path: str | Path,
    run_manifest_path: str | Path,
    require_artifact_verification: bool = True,
) -> EvaluationRunManifest:
    """Verify a benchmark package, evaluate a submission, and emit traceable artifacts."""
    package = load_package(package_path)
    if require_artifact_verification:
        errors = verify_package_artifacts(package, package_root)
        if errors:
            raise ValueError("benchmark package verification failed: " + "; ".join(errors))

    records = load_submission(submission_path)
    validate_submission_against_package(package, records, task_id=task_id)
    task = next(task for task in package.tasks if task.task_id == task_id)
    report = evaluate_submission(package.manifest, records, model_id=model_id, task_contract=task)
    save_report(report, report_path)

    submission_sha256 = sha256_file(submission_path)
    bundle = build_bundle(
        package.manifest,
        report,
        task_id=task_id,
        submission_sha256=submission_sha256,
    )
    save_bundle(bundle, bundle_path)

    package_sha256 = canonical_json_hash(package.model_dump(mode="json"))
    run_manifest = build_run_manifest(
        benchmark_id=package.benchmark_id,
        benchmark_version=package.version,
        task_id=task_id,
        model_id=model_id,
        benchmark_package_sha256=package_sha256,
        submission_sha256=submission_sha256,
        evaluation_fingerprint=bundle.evaluation_fingerprint,
        bundle_id=bundle.bundle_id,
        report_path=report_path,
        bundle_path=bundle_path,
    )
    save_run_manifest(run_manifest, run_manifest_path)
    return run_manifest
