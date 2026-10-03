"""Interoperable evaluation bundle contract for CardiBench/CardiEval."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import BenchmarkManifest, EvaluationReport
from .provenance import canonical_json_hash


class SubmissionBundle(BaseModel):
    """Self-describing bundle linking benchmark, submission, and evaluator output."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.2"
    bundle_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    submission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    report: EvaluationReport
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def fingerprint_payload(self) -> dict:
        report = self.report.model_dump(mode="json")
        report.pop("created_at", None)
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_sha256": self.benchmark_sha256,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "evaluator_version": self.report.evaluator_version,
            "primary_metric": self.report.primary_metric,
            "primary_direction": self.report.primary_direction,
            "report": report,
        }

    def expected_evaluation_fingerprint(self) -> str:
        return canonical_json_hash(self.fingerprint_payload())

    def expected_bundle_id(self) -> str:
        return canonical_json_hash({
            "benchmark_sha256": self.benchmark_sha256,
            "submission_sha256": self.submission_sha256,
            "evaluation_fingerprint": self.expected_evaluation_fingerprint(),
        })

    def verify_integrity(self) -> None:
        errors: list[str] = []
        expected_fingerprint = self.expected_evaluation_fingerprint()
        if expected_fingerprint != self.evaluation_fingerprint:
            errors.append("evaluation_fingerprint mismatch")
        expected_bundle = self.expected_bundle_id()
        if expected_bundle != self.bundle_id:
            errors.append("bundle_id mismatch")
        if self.model_id != self.report.model_id:
            errors.append("bundle model_id does not match report model_id")
        if self.benchmark_sha256 != self.report.benchmark_sha256:
            errors.append("bundle benchmark_sha256 does not match report")
        if errors:
            raise ValueError("; ".join(errors))


def build_bundle(
    manifest: BenchmarkManifest,
    report: EvaluationReport,
    *,
    task_id: str,
    submission_sha256: str,
) -> SubmissionBundle:
    """Construct a validated bundle and verify report/manifest identity."""
    if report.benchmark_id != manifest.benchmark_id:
        raise ValueError("report benchmark_id does not match manifest")
    if report.benchmark_version != manifest.version:
        raise ValueError("report benchmark_version does not match manifest")
    if report.benchmark_sha256 != manifest.dataset_sha256:
        raise ValueError("report benchmark_sha256 does not match manifest")
    if report.task_id != task_id:
        raise ValueError("report task_id does not match bundle task_id")

    bundle = SubmissionBundle(
        bundle_id="0" * 64,
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.version,
        task_id=task_id,
        model_id=report.model_id,
        submission_sha256=submission_sha256,
        benchmark_sha256=manifest.dataset_sha256,
        evaluation_fingerprint="0" * 64,
        report=report,
    )
    fingerprint = bundle.expected_evaluation_fingerprint()
    bundle = bundle.model_copy(update={
        "evaluation_fingerprint": fingerprint,
        "bundle_id": canonical_json_hash({
            "benchmark_sha256": manifest.dataset_sha256,
            "submission_sha256": submission_sha256,
            "evaluation_fingerprint": fingerprint,
        }),
    })
    bundle.verify_integrity()
    return bundle


def save_bundle(bundle: SubmissionBundle, path: str | Path) -> None:
    bundle.verify_integrity()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
