"""Independent submission validation and evaluation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ._version import __version__
from .calibration import brier_score, expected_calibration_error
from .diagnostics import (
    cohen_kappa,
    matthews_correlation,
    negative_predictive_value,
    positive_predictive_value,
    sensitivity,
    specificity,
)
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
from .models import (
    BenchmarkManifest,
    EvaluationReport,
    MetricResult,
    PredictionRecord,
    SubgroupResult,
)
from .ranking import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from .registry import BenchmarkTask
from .stats import bootstrap_ci

BASE_CLASSIFICATION_METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
}
SCORE_METRICS: dict[str, Callable] = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier_score,
    "ece": expected_calibration_error,
}
DIAGNOSTIC_METRICS: dict[str, Callable] = {
    "sensitivity": sensitivity,
    "specificity": specificity,
    "positive_predictive_value": positive_predictive_value,
    "negative_predictive_value": negative_predictive_value,
    "matthews_correlation": matthews_correlation,
    "cohen_kappa": cohen_kappa,
}
REGRESSION_METRICS = {"mae": mae, "rmse": rmse}
RANKING_METRICS = {
    "mrr": reciprocal_rank,
    "hit_rate@10": lambda y, s: hit_rate_at_k(y, s, 10),
    "ndcg@10": lambda y, s: ndcg_at_k(y, s, 10),
}


def load_submission(path: str | Path) -> list[PredictionRecord]:
    """Load JSONL prediction records and reject malformed or duplicate rows."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Submission file does not exist: {path}")
    records: list[PredictionRecord] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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
    if len(manifest.sample_ids) != len(manifest.sample_set()):
        raise ValueError("Benchmark manifest contains duplicate sample IDs")
    expected = manifest.sample_set()
    observed = {r.sample_id for r in records}
    missing = expected - observed
    extra = observed - expected
    if missing:
        raise ValueError(f"Submission missing {len(missing)} benchmark samples")
    if extra:
        raise ValueError(f"Submission contains {len(extra)} out-of-benchmark samples")


def _order_records(
    manifest: BenchmarkManifest, records: Sequence[PredictionRecord]
) -> list[PredictionRecord]:
    positions = {sample_id: i for i, sample_id in enumerate(manifest.sample_ids)}
    return sorted(records, key=lambda r: positions[r.sample_id])


def _apply_authoritative_reference(
    manifest: BenchmarkManifest, records: Sequence[PredictionRecord]
) -> tuple[list[PredictionRecord], str]:
    """Replace model-supplied reference fields with evaluator-controlled values when present."""
    if manifest.authoritative_labels is None:
        if any(record.y_true is None for record in records):
            raise ValueError(
                "submission contains missing y_true values and the benchmark provides no "
                "authoritative labels"
            )
        return list(records), "submission"
    updated: list[PredictionRecord] = []
    for record in records:
        updates = {"y_true": manifest.authoritative_labels[record.sample_id]}
        if manifest.authoritative_subgroups is not None:
            updates["subgroup"] = manifest.authoritative_subgroups[record.sample_id]
        updated.append(record.model_copy(update=updates))
    return updated, "benchmark_manifest"


def _metric_result(
    name: str,
    value: float,
    records_count: int,
    fn: Callable,
    y_true,
    prediction,
    *,
    direction: str,
) -> MetricResult:
    low: float | None = None
    high: float | None = None
    try:
        low, high = bootstrap_ci(y_true, prediction, fn, seed=0)
    except ValueError:
        # Some threshold/diagnostic metrics have undefined bootstrap replicates
        # for small or highly imbalanced groups. The point estimate remains valid.
        pass
    return MetricResult(
        name=name,
        value=float(value),
        ci_low=low,
        ci_high=high,
        n=records_count,
        direction=direction,
    )


def _classification_metrics(
    records: Sequence[PredictionRecord],
    *,
    requested_metrics: set[str] | None = None,
) -> list[MetricResult]:
    yt = np.asarray([r.y_true for r in records])
    yp = np.asarray([r.y_pred for r in records])
    requested = requested_metrics or (
        set(BASE_CLASSIFICATION_METRICS) | set(SCORE_METRICS) | set(DIAGNOSTIC_METRICS)
    )
    results: list[MetricResult] = []

    for name, fn in BASE_CLASSIFICATION_METRICS.items():
        if name not in requested:
            continue
        value = fn(yt, yp)
        results.append(
            _metric_result(
                name, value, len(records), fn, yt, yp, direction=METRIC_DIRECTIONS[name]
            )
        )

    binary_labels = set(np.unique(yt).tolist()) == {0, 1}
    scores = [r.score for r in records]
    if binary_labels and any(name in requested for name in SCORE_METRICS):
        if all(score is not None for score in scores):
            score_array = np.asarray(scores, dtype=float)
            for name, fn in SCORE_METRICS.items():
                if name not in requested:
                    continue
                value = fn(yt, score_array)
                results.append(
                    _metric_result(
                        name,
                        value,
                        len(records),
                        fn,
                        yt,
                        score_array,
                        direction=METRIC_DIRECTIONS[name],
                    )
                )

    if binary_labels:
        for name, fn in DIAGNOSTIC_METRICS.items():
            if name not in requested:
                continue
            value = float(fn(yt, yp))
            if math.isfinite(value):
            results.append(
                _metric_result(
                    name,
                    value,
                    len(records),
                    fn,
                    yt,
                    yp,
                    direction=METRIC_DIRECTIONS[name],
                )
            )
    return results


def _regression_metrics(
    records: Sequence[PredictionRecord],
    *,
    requested_metrics: set[str] | None = None,
) -> list[MetricResult]:
    yt = np.asarray([float(r.y_true) for r in records], dtype=float)
    yp = np.asarray([float(r.y_pred) for r in records], dtype=float)
    requested = requested_metrics or set(REGRESSION_METRICS)
    results: list[MetricResult] = []
    for name, fn in REGRESSION_METRICS.items():
        if name not in requested:
            continue
        value = fn(yt, yp)
        results.append(
            _metric_result(
                name, value, len(records), fn, yt, yp, direction=METRIC_DIRECTIONS[name]
            )
        )
    return results


def _ranking_metrics(
    records: Sequence[PredictionRecord],
    *,
    requested_metrics: set[str] | None = None,
) -> list[MetricResult]:
    relevance = np.asarray([float(r.y_true) for r in records], dtype=float)
    scores = [r.score for r in records]
    if not all(score is not None for score in scores):
        raise ValueError("ranking evaluation requires a score for every prediction")
    score_array = np.asarray(scores, dtype=float)
    requested = requested_metrics or set(RANKING_METRICS)
    results: list[MetricResult] = []
    for name, fn in RANKING_METRICS.items():
        if name not in requested:
            continue
        value = fn(relevance, score_array)
        results.append(
            _metric_result(
                name,
                value,
                len(records),
                fn,
                relevance,
                score_array,
                direction=METRIC_DIRECTIONS[name],
            )
        )
    return results


def _subgroup_results(
    records: Sequence[PredictionRecord],
    task: str,
    *,
    min_n: int,
) -> list[SubgroupResult]:
    if min_n < 1:
        raise ValueError("subgroup_min_n must be >= 1")
    groups: dict[str, list[PredictionRecord]] = {}
    for record in records:
        if record.subgroup is not None:
            groups.setdefault(record.subgroup, []).append(record)
    results: list[SubgroupResult] = []
    for name, group in sorted(groups.items()):
        warning = (
            None
            if len(group) >= min_n
            else f"subgroup has n={len(group)} below recommended minimum n={min_n}"
        )
        try:
            if task in {"classification", "binary_classification"}:
                metrics = _classification_metrics(group)
            elif task == "regression":
                metrics = _regression_metrics(group)
            elif task == "ranking":
                metrics = _ranking_metrics(group)
            else:
                metrics = []
                warning = f"{warning + '; ' if warning else ''}subgroup metrics not implemented for {task}"
        except ValueError as exc:
            metrics = []
            warning = f"{warning + '; ' if warning else ''}{exc}"
        results.append(SubgroupResult(subgroup=name, n=len(group), metrics=metrics, warning=warning))
    return results


def evaluate_submission(
    manifest: BenchmarkManifest,
    records: Sequence[PredictionRecord],
    *,
    model_id: str,
    subgroup_min_n: int = 10,
    task_contract: BenchmarkTask | None = None,
) -> EvaluationReport:
    """Evaluate a submission with exact sample alignment and optional task enforcement."""
    if not model_id.strip():
        raise ValueError("model_id must not be blank")
    _assert_alignment(manifest, records)
    if task_contract is not None:
        task_contract.validate_manifest(manifest)

    referenced, ground_truth_source = _apply_authoritative_reference(manifest, records)
    ordered = _order_records(manifest, referenced)
    requested_metrics = set(task_contract.allowed_metrics) if task_contract is not None else None

    if manifest.task in {"classification", "binary_classification"}:
        metrics = _classification_metrics(ordered, requested_metrics=requested_metrics)
    elif manifest.task == "regression":
        metrics = _regression_metrics(ordered, requested_metrics=requested_metrics)
    elif manifest.task == "ranking":
        metrics = _ranking_metrics(ordered, requested_metrics=requested_metrics)
    else:
        raise NotImplementedError(f"Task type not implemented yet: {manifest.task}")

    task_id = None
    primary_metric = None
    primary_value = None
    primary_direction = None
    warnings: list[str] = []

    if task_contract is not None:
        task_id = task_contract.task_id
        primary_metric = task_contract.primary_metric
        primary_direction = task_contract.primary_direction
        metric_by_name = {metric.name: metric for metric in metrics}
        primary = metric_by_name.get(primary_metric)
        if primary is None:
            raise ValueError(f"Primary metric {primary_metric!r} was not produced by evaluator")
        if primary.direction != primary_direction:
            raise ValueError(f"Primary metric {primary_metric!r} direction does not match task contract")
        primary_value = primary.value
    if ground_truth_source == "submission":
        warnings.append(
            "Ground truth comes from the submission records; this run is not an independent "
            "evaluator-controlled label assessment."
        )

    subgroups = _subgroup_results(ordered, manifest.task, min_n=subgroup_min_n)
    warnings.extend(
        f"Subgroup '{item.subgroup}': {item.warning}"
        for item in subgroups
        if item.warning
    )
    report = EvaluationReport(
        evaluator_version=__version__,
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.version,
        benchmark_sha256=manifest.dataset_sha256,
        task=manifest.task,
        split=manifest.split,
        model_id=model_id,
        task_id=task_id,
        primary_metric=primary_metric,
        primary_value=primary_value,
        primary_direction=primary_direction,
        ground_truth_source=ground_truth_source,
        metrics=metrics,
        subgroups=subgroups,
        warnings=warnings,
    )
    if task_contract is not None:
        task_contract.validate_report_contract(report)
    return report


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_report(report: EvaluationReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
