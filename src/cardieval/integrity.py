"""Deterministic artifact hashing and verification."""

from __future__ import annotations

from pathlib import Path
import hashlib

from pydantic import BaseModel, ConfigDict, Field

from .provenance import canonical_json_hash, sha256_text


class ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    release_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    publication_id: str = Field(min_length=1)
    artifacts: list[ArtifactRecord] = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def fingerprint_file(path: str | Path, *, kind: str) -> ArtifactRecord:
    file_path = Path(path)
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
    errors: list[str] = []
    for artifact in manifest.artifacts:
        path = root_path / artifact.path
        if not path.is_file():
            errors.append(f"missing artifact: {artifact.path}")
            continue
        actual = fingerprint_file(path, kind=artifact.kind)
        if actual.sha256 != artifact.sha256:
            errors.append(f"sha256 mismatch: {artifact.path}")
        if actual.size_bytes != artifact.size_bytes:
            errors.append(f"size mismatch: {artifact.path}")
    return errors
