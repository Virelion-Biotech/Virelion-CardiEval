"""CardiBench-compatible benchmark package ingestion and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import BenchmarkManifest, PredictionRecord
from .registry import BenchmarkTask


class BenchmarkArtifact(BaseModel):
    """A declared benchmark file with an integrity hash."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class BenchmarkPackage(BaseModel):
    """Self-describing benchmark package consumed by CardiEval."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    manifest: BenchmarkManifest
    tasks: list[BenchmarkTask] = Field(min_length=1)
    artifacts: list[BenchmarkArtifact] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    def validate_contracts(self) -> None:
        """Ensure manifest and task definitions describe the same benchmark release."""
        if self.manifest.benchmark_id != self.benchmark_id:
            raise ValueError("package benchmark_id does not match manifest")
        if self.manifest.version != self.version:
            raise ValueError("package version does not match manifest version")
        keys: set[tuple[str, str, str]] = set()
        for task in self.tasks:
            key = (task.benchmark_id, task.version, task.task_id)
            if key in keys:
                raise ValueError(f"duplicate task in package: {key}")
            keys.add(key)
            task.validate_manifest(self.manifest)

        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("benchmark artifact paths must be unique")


def fingerprint_directory(root: str | Path, *, kinds: dict[str, str] | None = None) -> list[BenchmarkArtifact]:
    """Fingerprint regular files under a directory for a reproducible package manifest."""
    root_path = Path(root)
    kinds = kinds or {}
    if not root_path.is_dir():
        raise ValueError(f"benchmark root is not a directory: {root}")
    records: list[BenchmarkArtifact] = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        data = path.read_bytes()
        rel = path.relative_to(root_path).as_posix()
        records.append(
            BenchmarkArtifact(
                path=rel,
                sha256=hashlib.sha256(data).hexdigest(),
                kind=kinds.get(path.suffix.lstrip("."), "file"),
                size_bytes=len(data),
            )
        )
    return records


def load_package(path: str | Path) -> BenchmarkPackage:
    """Load and validate a serialized benchmark package."""
    package = BenchmarkPackage.model_validate_json(Path(path).read_text(encoding="utf-8"))
    package.validate_contracts()
    return package


def verify_package_artifacts(package: BenchmarkPackage, root: str | Path) -> list[str]:
    """Verify all declared package artifacts against the filesystem."""
    root_path = Path(root)
    errors: list[str] = []
    for artifact in package.artifacts:
        path = root_path / artifact.path
        if not path.is_file():
            errors.append(f"missing benchmark artifact: {artifact.path}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != artifact.sha256:
            errors.append(f"sha256 mismatch: {artifact.path}")
        if len(data) != artifact.size_bytes:
            errors.append(f"size mismatch: {artifact.path}")
    return errors


def validate_submission_against_package(
    package: BenchmarkPackage,
    records: list[PredictionRecord],
    *,
    task_id: str,
) -> None:
    """Validate a model submission against one registered package task."""
    package.validate_contracts()
    matches = [task for task in package.tasks if task.task_id == task_id]
    if len(matches) != 1:
        raise ValueError(f"package must contain exactly one task named {task_id!r}")
    matches[0].validate_manifest(package.manifest)

    expected = package.manifest.sample_set()
    observed = [record.sample_id for record in records]
    if len(observed) != len(set(observed)):
        raise ValueError("submission contains duplicate sample IDs")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        extra = sorted(observed_set - expected)
        raise ValueError(f"submission sample set mismatch; missing={missing[:5]}, extra={extra[:5]}")
