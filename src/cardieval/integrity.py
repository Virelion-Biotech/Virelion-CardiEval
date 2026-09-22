"""Deterministic artifact hashing and verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provenance import canonical_json_hash, safe_artifact_path, sha256_text


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> "ArtifactRecord":
        raw = Path(self.path)
        if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
            raise ValueError("release artifact path must be a clean relative path")
        return self


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.1"
    release_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    publication_id: str = Field(min_length=1)
    artifacts: list[ArtifactRecord] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def canonical_payload(self) -> dict:
        return {
            "version": self.version,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "task_id": self.task_id,
            "publication_id": self.publication_id,
            "artifacts": [
                item.model_dump(mode="json") for item in sorted(self.artifacts, key=lambda x: x.path)
            ],
        }

    def expected_manifest_sha256(self) -> str:
        return canonical_json_hash(self.canonical_payload())

    def expected_release_id(self) -> str:
        return sha256_text(f"cardieval-release:{self.expected_manifest_sha256()}")

    def verify_self(self) -> list[str]:
        errors: list[str] = []
        if self.expected_manifest_sha256() != self.manifest_sha256:
            errors.append("release manifest self-hash mismatch")
        if self.expected_release_id() != self.release_id:
            errors.append("release_id does not match manifest contents")
        return errors


def fingerprint_file(path: str | Path, *, kind: str) -> ArtifactRecord:
    file_path = Path(path)
    if not file_path.is_file() or file_path.is_symlink():
        raise ValueError(f"artifact must be a regular file: {path}")
    data = file_path.read_bytes()
    return ArtifactRecord(
        path=str(file_path),
        sha256=hashlib.sha256(data).hexdigest(),
        kind=kind,
        size_bytes=len(data),
    )


def build_release_manifest(
    *,
    version: str,
    benchmark_id: str,
    benchmark_version: str,
    task_id: str,
    publication_id: str,
    artifacts: list[ArtifactRecord],
) -> ReleaseManifest:
    records = sorted(artifacts, key=lambda item: item.path)
    if not records:
        raise ValueError("At least one release artifact is required")
    if len({item.path for item in records}) != len(records):
        raise ValueError("Duplicate artifact paths are not permitted")
    payload = {
        "version": version,
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "task_id": task_id,
        "publication_id": publication_id,
        "artifacts": [item.model_dump(mode="json") for item in records],
    }
    manifest_hash = canonical_json_hash(payload)
    return ReleaseManifest(
        release_id=sha256_text(f"cardieval-release:{manifest_hash}"),
        version=version,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        task_id=task_id,
        publication_id=publication_id,
        artifacts=records,
        manifest_sha256=manifest_hash,
    )


def verify_release_manifest(manifest: ReleaseManifest, root: str | Path = ".") -> list[str]:
    root_path = Path(root)
    errors = manifest.verify_self()
    for artifact in manifest.artifacts:
        try:
            path = safe_artifact_path(root_path, artifact.path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or non-regular artifact: {artifact.path}")
            continue
        actual = fingerprint_file(path, kind=artifact.kind)
        if actual.sha256 != artifact.sha256:
            errors.append(f"sha256 mismatch: {artifact.path}")
        if actual.size_bytes != artifact.size_bytes:
            errors.append(f"size mismatch: {artifact.path}")
    return errors
