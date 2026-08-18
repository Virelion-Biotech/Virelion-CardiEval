"""Independent submission validation and evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibration import brier_score, expected_calibration_error
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
from .models import BenchmarkManifest, EvaluationReport, MetricResult, PredictionRecord, SubgroupResult
from .ranking import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from .registry import BenchmarkTask
from .stats import bootstrap_ci

CLASSIFICATION_METRICS = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "macro_f1": macro_f1}
REGRESSION_METRICS = {"mae": mae, "rmse": rmse}


def load_submission(path: str | Path) -> list[PredictionRecord]:
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


def _order_records(manifest: BenchmarkManifest, records: Sequence[PredictionRecord]) -> list[PredictionRecord]:
    positions = {sample_id: i for i, sample_id in enumerate(manifest.sample_ids)}
    return sorted(records, key=lambda r: positions[r.sample_id])


def _classification_metrics(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    yt = np.asarray([r.y_true for r in records])
    yp = np.asarray([r.y_pred for r in records])
    results: list[MetricResult] = []
    for name, fn in CLASSIFICATION_METRICS.items():
        value = fn(yt, yp)
        low, high = bootstrap_ci(yt, yp, fn, seed=0)
        results.append(MetricResult(name=name, value=value, ci_low=low, ci_high=high, n=len(records), direction=METRIC_DIRECTIONS[name]))
    scores = [r.score for r in records]
    if all(score is not None for score in scores) and len(np.unique(yt)) == 2:
        score_array = np.asarray(scores, dtype=float)
        for name, fn in (("auroc", auroc), ("auprc", auprc)):
            value = fn(yt, score_array)
            low, high = bootstrap_ci(yt, score_array, fn, seed=0)
            results.append(MetricResult(name=name, value=value, ci_low=low, ci_high=high, n=len(records), direction=METRIC_DIRECTIONS[name]))
        for name, fn in (("brier", brier_score), ("ece", expected_calibration_error)):
            value = fn(yt, score_array)
            low, high = bootstrap_ci(yt, score_array, fn, seed=0)
            results.append(MetricResult(name=name, value=value, ci_low=low, ci_high=high, n=len(records), direction="lower_is_better"))
    return results


def _regression_metrics(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    yt = np.asarray([float(r.y_true) for r in records])
    yp = np.asarray([float(r.y_pred) for r in records])
    results: list[MetricResult] = []
    for name, fn in REGRESSION_METRICS.items():
        value = fn(yt, yp)
        low, high = bootstrap_ci(yt, yp, fn, seed=0)
        results.append(MetricResult(name=name, value=value, ci_low=low, ci_high=high, n=len(records), direction=METRIC_DIRECTIONS[name]))
    return results


def _ranking_metrics(records: Sequence[PredictionRecord]) -> list[MetricResult]:
    relevance = np.asarray([float(r.y_true) for r in records])
    scores = [r.score for r in records]
    if not all(score is not None for score in scores):
        raise ValueError("ranking evaluation requires a score for every prediction")
    score_array = np.asarray(scores, dtype=float)
    specs = [
        ("mrr", reciprocal_rank),
        ("hit_rate@10", lambda y, s: hit_rate_at_k(y, s, 10)),
        ("ndcg@10", lambda y, s: ndcg_at_k(y, s, 10)),
    ]
    results: list[MetricResult] = []
    for name, fn in specs:
        value = fn(relevance, score_array)
        low, high = bootstrap_ci(relevance, score_array, fn, seed=0)
        results.append(MetricResult(name=name, value=value, ci_low=low, ci_high=high, n=len(records), direction="higher_is_better"))
    return results


def _subgroup_results(records: Sequence[PredictionRecord], task: str, *, min_n: int) -> list[SubgroupResult]:
    groups: dict[str, list[PredictionRecord]] = {}
    for record in records:
        if record.subgroup is not None:
            groups.setdefault(record.subgroup, []).append(record)
    results: list[SubgroupResult] = []
    for name, group in sorted(groups.items()):
        warning = None if len(group) >= min_n else f"subgroup has n={len(group)} below recommended minimum n={min_n}"
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
    """Evaluate a submission, optionally enforcing a registered task contract."""
    _assert_alignment(manifest, records)
    if task_contract is not None:
        task_contract.validate_manifest(manifest)
    ordered = _order_records(manifest, records)
    if manifest.task in {"classification", "binary_classification"}:
        metrics = _classification_metrics(ordered)
    elif manifest.task == "regression":
        metrics = _regression_metrics(ordered)
    elif manifest.task == "ranking":
        metrics = _ranking_metrics(ordered)
    else:
        raise NotImplementedError(f"Task type not implemented yet: {manifest.task}")

    if task_contract is not None:
        disallowed = sorted({metric.name for metric in metrics} - set(task_contract.allowed_metrics))
        if disallowed:
            raise ValueError(f"Evaluator produced metrics not allowed by task contract: {disallowed}")
        metric_by_name = {metric.name: metric for metric in metrics}
        primary = metric_by_name.get(task_contract.primary_metric)
        if primary is None:
            raise ValueError(f"Primary metric {task_contract.primary_metric!r} was not produced by evaluator")
        primary_metric = task_contract.primary_metric
        primary_value = primary.value
        primary_direction = task_contract.primary_direction
        task_id = task_contract.task_id
    else:
        primary_metric = None
        primary_value = None
        primary_direction = None
        task_id = None

    subgroups = _subgroup_results(ordered, manifest.task, min_n=subgroup_min_n)
    warnings = [f"Subgroup '{item.subgroup}': {item.warning}" for item in subgroups if item.warning]
    return EvaluationReport(
        evaluator_version="0.4.0",
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
        metrics=metrics,
        subgroups=subgroups,
        warnings=warnings,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_report(report: EvaluationReport, path: str | Path) -> None:
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
