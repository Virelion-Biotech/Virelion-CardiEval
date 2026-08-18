from cardieval.publication import LeaderboardSnapshot
from cardieval.publication_history import compare_snapshots, snapshot_hash
from cardieval.leaderboard import Leaderboard, LeaderboardEntry


def snapshot(a_score: float, b_score: float) -> LeaderboardSnapshot:
    leaderboard = Leaderboard(
        benchmark_id="bench",
        benchmark_version="1",
        split="test",
        metric="auroc",
        direction="higher_is_better",
        entries=[
            LeaderboardEntry(rank=1 if a_score >= b_score else 2, model_id="a", score=a_score, n_reports=1, benchmarks=["bench@1"]),
            LeaderboardEntry(rank=1 if b_score > a_score else 2, model_id="b", score=b_score, n_reports=1, benchmarks=["bench@1"]),
        ],
    )
    return LeaderboardSnapshot(
        benchmark_id="bench",
        benchmark_version="1",
        task_id="task",
        split="test",
        primary_metric="auroc",
        primary_direction="higher_is_better",
        n_bundles=2,
        n_models=2,
        bundles=["bundle-a", "bundle-b"],
        leaderboard=leaderboard,
    )


def test_snapshot_hash_is_deterministic():
    assert snapshot_hash(snapshot(0.9, 0.8)) == snapshot_hash(snapshot(0.9, 0.8))


def test_compare_snapshots_reports_rank_and_score_change():
    comparison = compare_snapshots(snapshot(0.9, 0.8), snapshot(0.85, 0.95))
    by_model = {item.model_id: item for item in comparison.deltas}
    assert by_model["a"].rank_change == -1
    assert by_model["b"].rank_change == 1
    assert by_model["a"].score_change == -0.05
    assert by_model["b"].score_change == 0.15
    assert comparison.changed_models == 2


def test_compare_snapshots_rejects_incompatible_contract():
    previous = snapshot(0.9, 0.8)
    current = snapshot(0.9, 0.8).model_copy(update={"primary_metric": "accuracy"})
    try:
        compare_snapshots(previous, current)
    except ValueError as exc:
        assert "primary_metric" in str(exc)
    else:
        raise AssertionError("expected incompatible snapshots to fail")
