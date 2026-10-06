import itertools
import numpy as np
import pytest
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.evaluator import evaluate_submission
from cardieval.registry import BenchmarkTask
from cardieval.ranking import reciprocal_rank, hit_rate_at_k, ndcg_at_k
from cardieval.calibration_curves import calibration_curve
from cardieval.bundle import SubmissionBundle, build_bundle
from cardieval.publication import publish_leaderboard
from pathlib import Path


def inputs(labels=(0, 1), metrics=("accuracy",), **kwargs):
    ids = [f"s{i}" for i in range(len(labels))]
    manifest = BenchmarkManifest(
        benchmark_id="repair",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=ids,
        dataset_sha256="1" * 64,
        authoritative_labels=dict(zip(ids, labels)),
        **kwargs,
    )
    task = BenchmarkTask(
        benchmark_id="repair",
        version="1",
        task_id="test",
        task_type="binary_classification",
        splits=["test"],
        allowed_metrics=list(metrics),
        primary_metric=metrics[0],
        primary_direction="higher_is_better",
        requires_authoritative_labels=True,
    )
    records = [
        PredictionRecord(sample_id=id, y_pred=label, score=0.1 + 0.8 * label)
        for id, label in zip(ids, labels)
    ]
    return manifest, task, records


def test_direct_api_duplicate_and_invalid_binary_rejected():
    manifest, task, records = inputs()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_submission(manifest, records + [records[0]], model_id="m", task_contract=task)
    with pytest.raises(ValueError, match="labels 0/1"):
        manifest, task, records = inputs((2, 3))
        evaluate_submission(manifest, records, model_id="m", task_contract=task)


def test_accuracy_boundary_interval_contains_near_perfect_truth():
    manifest, task, records = inputs((1,) * 100)
    metric = evaluate_submission(manifest, records, model_id="m", task_contract=task).metrics[0]
    assert metric.ci_low < 0.99 < metric.ci_high
    assert metric.ci_method == "clopper_pearson_binomial" and metric.confidence_level == 0.95


def test_single_class_subgroups_retain_defined_metrics_and_warn_undefined():
    labels = (0,) * 10 + (1,) * 10
    subgroups = {f"s{i}": str(label) for i, label in enumerate(labels)}
    manifest, task, records = inputs(
        labels,
        metrics=("accuracy", "brier", "ece", "auroc", "sensitivity", "specificity"),
        authoritative_subgroups=subgroups,
    )
    report = evaluate_submission(manifest, records, model_id="m", task_contract=task)
    for group in report.subgroups:
        values = {m.name: m.value for m in group.metrics}
        assert values["brier"] == pytest.approx(0.01)
        assert "ece" in values
        assert values["specificity" if group.subgroup == "0" else "sensitivity"] == 1
        assert "auroc" not in values
        assert "auroc" in group.warning


def test_authoritative_subgroups_apply_without_authoritative_labels():
    manifest, task, records = inputs()
    manifest = manifest.model_copy(
        update={
            "authoritative_labels": None,
            "authoritative_subgroups": {"s0": "fixed", "s1": "fixed"},
        }
    )
    records = [r.model_copy(update={"y_true": r.y_pred, "subgroup": "poison"}) for r in records]
    task = task.model_copy(update={"requires_authoritative_labels": False})
    report = evaluate_submission(manifest, records, model_id="m", task_contract=task)
    assert [g.subgroup for g in report.subgroups] == ["fixed"]


def test_cluster_map_controls_units_and_uncertainty():
    labels = (1,) * 40
    manifest, task, records = inputs(
        labels, authoritative_clusters={f"s{i}": str(i // 10) for i in range(40)}
    )
    records = [r.model_copy(update={"y_pred": int(i // 10 < 3)}) for i, r in enumerate(records)]
    metric = evaluate_submission(manifest, records, model_id="m", task_contract=task).metrics[0]
    assert metric.value == 0.75 and metric.n == 40 and metric.n_independent == 4
    assert (
        metric.resampling_unit == "cluster" and metric.ci_method == "cluster_percentile_bootstrap"
    )
    assert metric.ci_high - metric.ci_low >= 0.5


def test_mrr_averages_queries_and_resamples_queries():
    ids = ["a", "b", "c", "d"]
    truth = [1, 0, 0, 1]
    manifest = BenchmarkManifest(
        benchmark_id="rank",
        version="1",
        task="ranking",
        split="test",
        sample_ids=ids,
        dataset_sha256="2" * 64,
        authoritative_labels=dict(zip(ids, truth)),
        authoritative_queries=dict(zip(ids, ["A", "A", "B", "B"])),
    )
    records = [
        PredictionRecord(sample_id=id, y_pred=0, score=score)
        for id, score in zip(ids, [0.9, 0.8, 0.7, 0.6])
    ]
    report = evaluate_submission(manifest, records, model_id="m")
    metric = next(m for m in report.metrics if m.name == "mrr")
    assert metric.value == 0.75 and metric.n_independent == 2 and metric.resampling_unit == "query"
    # Query-specific score shifts must not change query order or result.
    records[2] = records[2].model_copy(update={"score": 100.7})
    records[3] = records[3].model_copy(update={"score": 100.6})
    assert (
        next(
            m
            for m in evaluate_submission(manifest, records, model_id="m").metrics
            if m.name == "mrr"
        ).value
        == 0.75
    )


def test_ranking_subgroups_cannot_split_a_query():
    manifest = BenchmarkManifest(
        benchmark_id="rank",
        version="1",
        task="ranking",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="2" * 64,
        authoritative_labels={"a": 1, "b": 0},
        authoritative_queries={"a": "q", "b": "q"},
        authoritative_subgroups={"a": "x", "b": "y"},
    )
    records = [
        PredictionRecord(sample_id="a", y_pred=0, score=0.9),
        PredictionRecord(sample_id="b", y_pred=0, score=0.1),
    ]
    with pytest.raises(ValueError, match="must not split"):
        evaluate_submission(manifest, records, model_id="m")


def test_expected_tie_metrics_against_all_permutations():
    truth = [0, 1, 2]
    permutations = list(itertools.permutations(truth))
    check_rr = np.mean([1 / (next(i for i, y in enumerate(p) if y > 0) + 1) for p in permutations])
    check_hit = np.mean([p[0] > 0 for p in permutations])
    discounts = 1 / np.log2(np.arange(2, 5))
    ideal = np.sum(np.array([2, 1, 0]) * discounts)
    check_ndcg = np.mean([np.sum(np.array(p) * discounts) / ideal for p in permutations])
    assert reciprocal_rank(truth, [1, 1, 1]) == pytest.approx(check_rr)
    assert hit_rate_at_k(truth, [1, 1, 1], 1) == pytest.approx(check_hit)
    assert ndcg_at_k(truth, [1, 1, 1], 3) == pytest.approx(check_ndcg)


def test_empty_calibration_bins_are_unknown_and_ece_metadata_visible():
    assert all(
        b.mean_predicted is None and b.observed_rate is None
        for b in calibration_curve([0, 1], [0, 1])
        if b.n == 0
    )
    manifest, task, records = inputs(metrics=("accuracy", "ece"))
    report = evaluate_submission(manifest, records, model_id="m", task_contract=task, ece_bins=20)
    details = next(m for m in report.metrics if m.name == "ece").details
    assert details["n_bins"] == 20 and len(details["bins"]) == 20


def test_reference_label_hash_prevents_forged_same_dataset_leaderboard():
    manifest, task, records = inputs()
    report = evaluate_submission(manifest, records, model_id="a", task_contract=task)
    changed = manifest.model_copy(update={"authoritative_labels": {"s0": 1, "s1": 0}})
    other = evaluate_submission(changed, records, model_id="b", task_contract=task)
    assert report.benchmark_sha256 == other.benchmark_sha256
    assert report.reference_sha256 != other.reference_sha256
    bundles = [
        build_bundle(manifest, report, task_id="test", submission_sha256="3" * 64),
        build_bundle(changed, other, task_id="test", submission_sha256="4" * 64),
    ]
    with pytest.raises(ValueError, match="reference"):
        publish_leaderboard(bundles, task)


def test_old_bundle_fingerprint_still_verifies():
    path = Path(__file__).parent / "fixtures/legacy_bundle_1_4_1.json"
    bundle = SubmissionBundle.model_validate_json(path.read_text(encoding="utf-8"))
    bundle.verify_integrity()
