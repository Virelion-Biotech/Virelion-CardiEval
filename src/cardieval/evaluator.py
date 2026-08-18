"""Independent submission validation and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import (
    METRIC_DIRECTIONS,
    accuracy,
    auprc,
    auroc,
    balanced_accuracy,
    macro_f1,
    mae,
    rmse,
)
from .models import BenchmarkManifest, EvaluationReport, MetricResult, PredictionRecord
from .stats import bootstrap_ci


CLASSIFICATION_METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
}
REGRESSION_METRICS = {"mae": mae, "rmse": rmse}


def load_submission(path: str | Path) -> list[PredictionRecord]:
    """Load a JSONL submission and reject malformed/duplicate sample records."""
    records: list[PredictionRecord] = []
    seen: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = PredictionRecord.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"Invalid submission at line {line_no}: {exc}") from exc
        if record.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        records.append(record)
    if not records:
        raise ValueError("Submission contains no prediction records")
    return records


def _assert_alignment(manifest: BenchmarkManifest, records: Sequence[PredictionRecord]) -> None:
    expected = manifest.sample_set()
    observed = {r.sample_id for r in records}
    missing = expected - observed
    extra = observed - expected
    if missing:
        raise ValueError(f"Submission missing {len(missing)} benchmark samples")
    if extra:
        raise ValueError(f"Submission contains {len(extra)} out-of-benchmark samples")
    if len(expected) != len(manifest.sample_ids):
        raise ValueError("Benchmark manifest contains duplicate sample IDs")


def _classification_metrics(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    yt = np.asarray([r.y_true for r in records])
    yp = np.asarray([r.y_pred for r in records])
    results = []
    for name, fn in CLASSIFICATION_METRICS.items():
        value = fn(yt, yp)
        low, high = bootstrap_ci(yt, yp, fn, seed=0)
        results.append(
            MetricResult(
                name=name,
                value=value,
                ci_low=low,
                ci_high=high,
                n=len(records),
                direction=METRIC_DIRECTIONS[name],
            )
        )
    scores = [r.score for r in records]
    if all(score is not None for score in scores) and len(np.unique(yt)) == 2:
        score_array = np.asarray(scores, dtype=float)
        for name, fn in (("auroc", auroc), ("auprc", auprc)):
            value = fn(yt, score_array)
            # Resample indices through a closure because AUROC/AUPRC use score instead of y_pred.
            def score_metric(a: np.ndarray, b: np.ndarray, metric=fn) -> float:
                return metric(a, b)
            low, high = bootstrap_ci(yt, score_array, score_metric, seed=0)
            results.append(
                MetricResult(
                    name=name,
                    value=value,
                    ci_low=low,
                    ci_high=high,
                    n=len(records),
                    direction=METRIC_DIRECTIONS[name],
                )
            )
    return results


def _regression_metrics(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    yt = np.asarray([float(r.y_true) for r in records])
    yp = np.asarray([float(r.y_pred) for r in records])
    results = []
    for name, fn in REGRESSION_METRICS.items():
        value = fn(yt, yp)
        low, high = bootstrap_ci(yt, yp, fn, seed=0)
        results.append(
            MetricResult(
                name=name,
                value=value,
                ci_low=low,
                ci_high=high,
                n=len(records),
                direction=METRIC_DIRECTIONS[name],
            )
        )
    return results


def evaluate_submission(
    manifest: BenchmarkManifest,
    records: Sequence[PredictionRecord],
    *,
    model_id: str,
) -> EvaluationReport:
    """Evaluate a submission without depending on model internals."""
    _assert_alignment(manifest, records)
    ordered = sorted(records, key=lambda r: manifest.sample_ids.index(r.sample_id))
    if manifest.task in {"classification", "binary_classification"}:
        metrics = _classification_metrics(ordered)
    elif manifest.task == "regression":
        metrics = _regression_metrics(ordered)
    else:
        raise NotImplementedError(f"Task type not implemented yet: {manifest.task}")
    return EvaluationReport(
        evaluator_version="0.1.0",
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.version,
        task=manifest.task,
        split=manifest.split,
        model_id=model_id,
        metrics=metrics,
    )


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 fingerprint used for dataset/artifact provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_report(report: EvaluationReport, path: str | Path) -> None:
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
