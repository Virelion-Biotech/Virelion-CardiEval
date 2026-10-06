import pytest

from cardieval.bundle import build_bundle
from cardieval.leaderboard import build_leaderboard
from cardieval.models import BenchmarkManifest, EvaluationReport, MetricResult
from cardieval.publication import LeaderboardSnapshot, ingest_bundles, publish_leaderboard
from cardieval.registry import BenchmarkTask
from cardieval.robustness import subgroup_robustness


def task(expected_hash=None):
    return BenchmarkTask(benchmark_id="bench", version="1", task_id="binary",
        task_type="binary_classification", allowed_metrics=["accuracy"],
        primary_metric="accuracy", primary_direction="higher_is_better", splits=["test"],
        expected_dataset_sha256=expected_hash)


def report(model="a", digest="0", **updates):
    fields = dict(evaluator_version="1", benchmark_id="bench", benchmark_version="1",
        benchmark_sha256=digest * 64, reference_sha256="5" * 64, task="binary_classification", split="test",
        model_id=model, task_id="binary", primary_metric="accuracy", primary_value=.8,
        primary_direction="higher_is_better",
        metrics=[MetricResult(name="accuracy", value=.8, n=2, direction="higher_is_better")])
    fields.update(updates)
    return EvaluationReport(**fields)


def manifest(digest="0"):
    return BenchmarkManifest(benchmark_id="bench", version="1", task="binary_classification",
        split="test", sample_ids=["x", "y"], dataset_sha256=digest * 64)


def bundle(model="a", digest="0"):
    return build_bundle(manifest(digest), report(model, digest), task_id="binary",
        submission_sha256=("3" if model == "a" else "4") * 64)


def board(reports, **kwargs):
    return build_leaderboard(reports, metric="accuracy", direction="higher_is_better", **kwargs)


def test_mixed_hashes_rejected_by_raw_leaderboard_and_strict_publication():
    with pytest.raises(ValueError, match="dataset hash"):
        board([report(), report("b", "1")])
    with pytest.raises(ValueError, match="dataset hash"):
        ingest_bundles([bundle(), bundle("b", "1")], task())


@pytest.mark.parametrize("updates", [{"task_id": "other"}, {"task": "classification"},
    {"errors": ["invalid predictions"]}])
def test_incompatible_or_failed_reports_cannot_be_ranked(updates):
    with pytest.raises(ValueError):
        board([report(), report("b", **updates)])


def test_duplicates_require_explicit_aggregation_opt_in():
    with pytest.raises(ValueError, match="duplicate model_id"):
        board([report(), report()])
    assert board([report(), report()], allow_repeated_models=True).entries[0].n_reports == 2


def test_optional_expected_hash_pins_manifest_report_and_publication():
    pinned = task("1" * 64)
    with pytest.raises(ValueError, match="expected dataset hash"):
        pinned.validate_manifest(manifest())
    with pytest.raises(ValueError, match="expected dataset hash"):
        pinned.validate_report_contract(report())
    with pytest.raises(ValueError, match="expected dataset hash"):
        publish_leaderboard([bundle()], pinned)
    pinned.validate_manifest(manifest("1"))
    pinned.validate_report_contract(report(digest="1"))


def test_snapshot_records_hash_and_rejects_mismatched_metadata():
    snapshot = publish_leaderboard([bundle(), bundle("b")], task())
    assert snapshot.benchmark_sha256 == "0" * 64
    assert snapshot.leaderboard.benchmark_sha256 == "0" * 64
    corrupted = snapshot.model_dump()
    corrupted["benchmark_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="dataset hashes"):
        LeaderboardSnapshot.model_validate(corrupted)


def test_legacy_snapshot_can_be_read_without_hash_metadata():
    snapshot = publish_leaderboard([bundle()], task()).model_dump()
    snapshot.pop("benchmark_sha256")
    snapshot["leaderboard"].pop("benchmark_sha256")
    snapshot["leaderboard"].pop("task_id")
    snapshot["schema_version"] = "1.1"
    assert LeaderboardSnapshot.model_validate(snapshot).benchmark_sha256 is None


def test_loss_metric_range_is_nonnegative_spread():
    summary = subgroup_robustness({"a": .1, "b": .4}, metric="mae", direction="lower_is_better")
    assert summary.best == .1
    assert summary.worst == .4
    assert summary.range == pytest.approx(.3)


def test_unbound_legacy_reports_require_reevaluation_before_publication():
    legacy = report(reference_sha256=None)
    with pytest.raises(ValueError, match="reference hash"):
        board([legacy])
    unbound = build_bundle(manifest(), legacy, task_id="binary", submission_sha256="3"*64)
    with pytest.raises(ValueError, match="reference hash"):
        publish_leaderboard([unbound], task())
