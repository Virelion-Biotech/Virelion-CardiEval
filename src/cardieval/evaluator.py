"""Independent submission validation and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ._version import __version__
from .calibration import brier_score, expected_calibration_error
from .calibration_curves import calibration_curve
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
from .stats import bootstrap_ci, binomial_accuracy_ci

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
    if len(observed) != len(records):
        raise ValueError("Submission contains duplicate sample IDs")
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
    updated: list[PredictionRecord] = []
    for record in records:
        updates = {}
        if manifest.authoritative_labels is not None:
            updates["y_true"] = manifest.authoritative_labels[record.sample_id]
        if manifest.authoritative_subgroups is not None:
            updates["subgroup"] = manifest.authoritative_subgroups[record.sample_id]
        updated.append(record.model_copy(update=updates))
    return updated, "benchmark_manifest" if manifest.authoritative_labels is not None else "submission"


def _metric_result(
    name: str,
    value: float,
    records_count: int,
    fn: Callable,
    y_true,
    prediction,
    *,
    direction: str,
    clusters=None,
    warning_sink: list[str] | None = None,
    details: dict | None = None,
) -> MetricResult:
    low: float | None = None
    high: float | None = None
    n_independent = len(set(clusters)) if clusters is not None else records_count
    method = "cluster_percentile_bootstrap" if clusters is not None else "percentile_bootstrap"
    uncertainty_warning = None
    details = dict(details or {})
    try:
        if name == "accuracy" and (clusters is None or n_independent == records_count):
            low, high = binomial_accuracy_ci(y_true, prediction, method="exact")
            method = "clopper_pearson_binomial"
        else:
            if n_independent < 2:
                raise ValueError("at least two independent units are required for bootstrap uncertainty")
            draws = {}
            low, high = bootstrap_ci(y_true, prediction, fn, seed=0, clusters=clusters, diagnostics=draws)
            details["bootstrap"] = draws
            if draws["rejected_resamples"]:
                uncertainty_warning = f"Metric {name!r} CI conditions on valid resamples; {draws['rejected_resamples']} draws rejected."
                if warning_sink is not None:
                    warning_sink.append(uncertainty_warning)
            if low == high:
                raise ValueError("degenerate bootstrap distribution; interval withheld")
    except ValueError as exc:
        low = high = None
        uncertainty_warning = f"Metric {name!r} CI unavailable: {exc}"
        if warning_sink is not None:
            warning_sink.append(uncertainty_warning)
    return MetricResult(
        name=name,
        value=float(value),
        ci_low=low,
        ci_high=high,
        n=records_count,
        direction=direction,
        ci_method=method,
        confidence_level=0.95 if low is not None else None,
        n_independent=n_independent,
        resampling_unit="cluster" if clusters is not None else "record",
        uncertainty_warning=uncertainty_warning,
        details=details,
    )


def _classification_metrics(
    records: Sequence[PredictionRecord], *, requested_metrics=None, warning_sink=None,
    cluster_map=None, ece_bins=10,
) -> list[MetricResult]:
    yt = np.asarray([r.y_true for r in records])
    yp = np.asarray([r.y_pred for r in records])
    requested = requested_metrics if requested_metrics is not None else (
        set(BASE_CLASSIFICATION_METRICS) | set(SCORE_METRICS) | set(DIAGNOSTIC_METRICS)
    )
    binary = set(np.unique(yt).tolist()).issubset({0, 1})
    clusters = [cluster_map[r.sample_id] for r in records] if cluster_map is not None else None
    functions = dict(BASE_CLASSIFICATION_METRICS)
    if binary:
        functions.update(SCORE_METRICS)
        functions.update(DIAGNOSTIC_METRICS)
    results = []
    for name in sorted(requested):
        if name not in functions:
            if warning_sink is not None:
                warning_sink.append(f"Metric {name!r} was omitted: incompatible label domain")
            continue
        fn = functions[name]
        details = {}
        if name in SCORE_METRICS:
            if any(r.score is None for r in records):
                if warning_sink is not None:
                    warning_sink.append(f"Metric {name!r} was omitted: missing probability scores")
                continue
            prediction = np.asarray([r.score for r in records], dtype=float)
        else:
            prediction = yp
        if name == 'ece':
            def fn(y, score):
                return expected_calibration_error(y, score, n_bins=ece_bins)
        try:
            value = float(fn(yt, prediction))
            if not math.isfinite(value):
                raise ValueError('undefined for this class/decision distribution')
            if name == 'ece':
                details = {'n_bins': ece_bins, 'binning': 'equal_width',
                           'bins': [b.model_dump() for b in calibration_curve(yt, prediction, n_bins=ece_bins)],
                           'interpretation': 'Binned ECE alone cannot establish calibration.'}
            elif name == 'auprc':
                details = {'estimator': 'average_precision'}
        except ValueError as exc:
            if warning_sink is not None:
                warning_sink.append(f"Metric {name!r} was omitted: {exc}")
            continue
        results.append(_metric_result(name, value, len(records), fn, yt, prediction,
            direction=METRIC_DIRECTIONS[name], clusters=clusters,
            warning_sink=warning_sink, details=details))
    return results


def _regression_metrics(records, *, requested_metrics=None, cluster_map=None, warning_sink=None):
    yt = np.asarray([float(r.y_true) for r in records])
    yp = np.asarray([float(r.y_pred) for r in records])
    clusters = [cluster_map[r.sample_id] for r in records] if cluster_map is not None else None
    requested = requested_metrics if requested_metrics is not None else set(REGRESSION_METRICS)
    return [_metric_result(name, fn(yt, yp), len(records), fn, yt, yp,
        direction=METRIC_DIRECTIONS[name], clusters=clusters, warning_sink=warning_sink)
        for name, fn in REGRESSION_METRICS.items() if name in requested]


def _ranking_metrics(records, *, requested_metrics=None, query_map=None, cluster_map=None, warning_sink=None):
    if any(r.score is None for r in records):
        raise ValueError("ranking evaluation requires a score for every prediction")
    requested = requested_metrics if requested_metrics is not None else set(RANKING_METRICS)
    groups = {}
    for record in records:
        query = query_map[record.sample_id] if query_map is not None else '__single_list__'
        groups.setdefault(query, []).append(record)
    if query_map is None and warning_sink is not None:
        warning_sink.append('Ranking has no authoritative query map: interpreted as one fixed query; no population CI.')
    results = []
    for name, fn in RANKING_METRICS.items():
        if name not in requested:
            continue
        values, units = [], []
        for query, group in sorted(groups.items()):
            values.append(fn([r.y_true for r in group], [r.score for r in group]))
            if cluster_map is not None:
                owners = {cluster_map[r.sample_id] for r in group}
                if len(owners) != 1:
                    raise ValueError('each ranking query must belong to exactly one authoritative cluster')
                units.append(next(iter(owners)))
        result = _metric_result(name, float(np.mean(values)), len(values),
            lambda y, scores: float(np.mean(scores)), np.zeros(len(values)), values,
            direction=METRIC_DIRECTIONS[name], clusters=units if cluster_map is not None else None,
            warning_sink=warning_sink,
            details={'n_queries': len(groups), 'aggregation': 'equal_query_mean',
                     'tie_policy': 'uniform_expected_rank', 'ndcg_gain': 'linear'})
        result.n = len(records)
        result.resampling_unit = 'cluster' if cluster_map is not None else 'query'
        results.append(result)
    return results


def _subgroup_results(records, task, *, min_n, requested_metrics=None, cluster_map=None, query_map=None, ece_bins=10):
    if min_n < 1:
        raise ValueError('subgroup_min_n must be >= 1')
    groups = {}
    for record in records:
        if record.subgroup is not None:
            groups.setdefault(record.subgroup, []).append(record)
    if task == 'ranking' and query_map is not None:
        memberships = {}
        for record in records:
            memberships.setdefault(query_map[record.sample_id], set()).add(record.subgroup)
        if any(len(values) != 1 for values in memberships.values()):
            raise ValueError('ranking subgroup assignments must not split a query candidate list')
    results = []
    for name, group in sorted(groups.items()):
        independent = len({cluster_map[r.sample_id] for r in group}) if cluster_map is not None else (
            len({query_map[r.sample_id] for r in group}) if task == 'ranking' and query_map is not None else len(group))
        notes = []
        if independent < min_n:
            notes.append(f'subgroup has n={independent} independent units below recommended minimum n={min_n}')
        if task in {'classification', 'binary_classification'}:
            metrics = _classification_metrics(group, requested_metrics=requested_metrics,
                cluster_map=cluster_map, warning_sink=notes, ece_bins=ece_bins)
        elif task == 'regression':
            metrics = _regression_metrics(group, requested_metrics=requested_metrics,
                cluster_map=cluster_map, warning_sink=notes)
        elif task == 'ranking':
            metrics = _ranking_metrics(group, requested_metrics=requested_metrics,
                query_map=query_map, cluster_map=cluster_map, warning_sink=notes)
        else:
            metrics = []
            notes.append(f'subgroup metrics not implemented for {task}')
        results.append(SubgroupResult(subgroup=name, n=len(group), metrics=metrics,
                                      warning='; '.join(notes) if notes else None))
    return results


def evaluate_submission(
    manifest: BenchmarkManifest,
    records: Sequence[PredictionRecord],
    *,
    model_id: str,
    subgroup_min_n: int = 10,
    task_contract: BenchmarkTask | None = None,
    ece_bins: int = 10,
) -> EvaluationReport:
    """Evaluate a submission with exact sample alignment and optional task enforcement."""
    if not model_id.strip():
        raise ValueError("model_id must not be blank")
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins < 2:
        raise ValueError("ece_bins must be an integer >= 2")
    _assert_alignment(manifest, records)
    if task_contract is not None:
        task_contract.validate_manifest(manifest)

    referenced, ground_truth_source = _apply_authoritative_reference(manifest, records)
    ordered = _order_records(manifest, referenced)
    if manifest.task == "binary_classification":
        if any(r.y_true not in (0, 1) or r.y_pred not in (0, 1) for r in ordered):
            raise ValueError("binary_classification requires truth and predicted labels 0/1")
    requested_metrics = set(task_contract.allowed_metrics) if task_contract is not None else None
    warnings: list[str] = []
    if manifest.authoritative_clusters is None and manifest.task != "ranking":
        warnings.append("Uncertainty assumes independent records; no authoritative subject/cluster map was supplied.")

    if manifest.task in {"classification", "binary_classification"}:
        metrics = _classification_metrics(
            ordered,
            requested_metrics=requested_metrics,
            warning_sink=warnings,
            cluster_map=manifest.authoritative_clusters,
            ece_bins=ece_bins,
        )
    elif manifest.task == "regression":
        metrics = _regression_metrics(ordered, requested_metrics=requested_metrics,
            cluster_map=manifest.authoritative_clusters, warning_sink=warnings)
    elif manifest.task == "ranking":
        metrics = _ranking_metrics(ordered, requested_metrics=requested_metrics,
            query_map=manifest.authoritative_queries, cluster_map=manifest.authoritative_clusters,
            warning_sink=warnings)
    else:
        raise NotImplementedError(f"Task type not implemented yet: {manifest.task}")

    task_id = None
    primary_metric = None
    primary_value = None
    primary_direction = None

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

    subgroups = _subgroup_results(
        ordered,
        manifest.task,
        min_n=subgroup_min_n,
        requested_metrics=requested_metrics,
        cluster_map=manifest.authoritative_clusters,
        query_map=manifest.authoritative_queries,
        ece_bins=ece_bins,
    )
    warnings.extend(
        f"Subgroup '{item.subgroup}': {item.warning}"
        for item in subgroups
        if item.warning
    )
    reference = {
        "task": manifest.task, "split": manifest.split,
        "samples": sorted((r.sample_id, r.y_true, r.subgroup) for r in ordered),
        "clusters": manifest.authoritative_clusters,
        "queries": manifest.authoritative_queries,
        "ece_bins": ece_bins,
        "allowed_metrics": sorted(requested_metrics) if requested_metrics is not None else None,
    }
    reference_sha256 = hashlib.sha256(json.dumps(reference, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    report = EvaluationReport(
        evaluator_version=__version__,
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.version,
        benchmark_sha256=manifest.dataset_sha256,
        reference_sha256=reference_sha256,
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
