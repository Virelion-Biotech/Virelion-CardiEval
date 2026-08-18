"""Reproducible evaluation provenance and artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(payload: Any) -> str:
    """Hash JSON after deterministic canonical serialization."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded)


def evaluation_fingerprint(
    *, benchmark_id: str, benchmark_version: str, dataset_sha256: str,
    evaluator_version: str, model_id: str, config: dict[str, Any] | None = None,
) -> str:
    """Create a stable fingerprint for an evaluation configuration."""
    payload = {
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "dataset_sha256": dataset_sha256,
        "evaluator_version": evaluator_version,
        "model_id": model_id,
        "config": config or {},
    }
    return canonical_json_hash(payload)


def artifact_manifest(paths: list[str | Path]) -> dict[str, str]:
    """Return deterministic path -> SHA-256 mappings for evaluation artifacts."""
    result: dict[str, str] = {}
    for raw_path in sorted(paths, key=lambda value: str(value)):
        path = Path(raw_path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result
