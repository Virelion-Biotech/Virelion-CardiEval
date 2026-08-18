"""Interoperable evaluation bundle contract for CardiBench/CardiEval."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import BenchmarkManifest, EvaluationReport
from .provenance import canonical_json_hash


class SubmissionBundle(BaseModel):
    """Self-describing bundle linking benchmark, submission, and evaluator output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    bundle_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    submission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    report: EvaluationReport
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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

    fingerprint = canonical_json_hash(
        {
            "benchmark_id": manifest.benchmark_id,
            "benchmark_version": manifest.version,
            "benchmark_sha256": manifest.dataset_sha256,
            "task_id": task_id,
            "model_id": report.model_id,
            "evaluator_version": report.evaluator_version,
            "report": report.model_dump(mode="json"),
        }
    )
    bundle_id = canonical_json_hash(
        {
            "benchmark_sha256": manifest.dataset_sha256,
            "submission_sha256": submission_sha256,
            "evaluation_fingerprint": fingerprint,
        }
    )
    return SubmissionBundle(
        bundle_id=bundle_id,
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.version,
        task_id=task_id,
        model_id=report.model_id,
        submission_sha256=submission_sha256,
        benchmark_sha256=manifest.dataset_sha256,
        evaluation_fingerprint=fingerprint,
        report=report,
    )


def save_bundle(bundle: SubmissionBundle, path: str | Path) -> None:
    Path(path).write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
