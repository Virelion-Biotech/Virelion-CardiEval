"""End-to-end evaluation run manifest for CardiBench -> CardiEval workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .provenance import canonical_json_hash


class EvaluationRunManifest(BaseModel):
    """Traceable record of one end-to-end evaluation event."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    benchmark_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_path: str = Field(min_length=1)
    bundle_path: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def build_run_manifest(
    *,
    benchmark_id: str,
    benchmark_version: str,
    task_id: str,
    model_id: str,
    benchmark_package_sha256: str,
    submission_sha256: str,
    evaluation_fingerprint: str,
    bundle_id: str,
    report_path: str | Path,
    bundle_path: str | Path,
) -> EvaluationRunManifest:
    payload = {
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "task_id": task_id,
        "model_id": model_id,
        "benchmark_package_sha256": benchmark_package_sha256,
        "submission_sha256": submission_sha256,
        "evaluation_fingerprint": evaluation_fingerprint,
        "bundle_id": bundle_id,
        "report_path": str(report_path),
        "bundle_path": str(bundle_path),
    }
    run_id = canonical_json_hash(payload)
    return EvaluationRunManifest(run_id=run_id, **payload)


def save_run_manifest(manifest: EvaluationRunManifest, path: str | Path) -> None:
    Path(path).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
