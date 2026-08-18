from cardieval.integrity import ArtifactRecord, build_release_manifest, fingerprint_file, verify_release_manifest
from cardieval.leaderboard import Leaderboard, LeaderboardEntry
from cardieval.publication import LeaderboardSnapshot
from cardieval.publication_history import compare_snapshots, snapshot_hash
from cardieval.scorecard import build_scorecard


def snapshot(scores):
    entries = [
        LeaderboardEntry(rank=i + 1, model_id=model, score=score, n_reports=1)
        for i, (model, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1]))
    ]
    return LeaderboardSnapshot(
        benchmark_id="bench",
        benchmark_version="1",
        task_id="task",
        split="test",
        primary_metric="auroc",
        primary_direction="higher_is_better",
        n_bundles=len(entries),
        n_models=len(entries),
        bundles=[f"b-{x.model_id}" for x in entries],
        leaderboard=Leaderboard(
            benchmark_id="bench",
            benchmark_version="1",
            split="test",
            metric="auroc",
            direction="higher_is_better",
            entries=entries,
        ),
    )


def test_snapshot_comparison_tracks_new_removed_and_moved_models():
    old = snapshot({"a": 0.8, "b": 0.7})
    new = snapshot({"a": 0.6, "c": 0.9})
    comparison = compare_snapshots(old, new)
    states = {item.model_id: item for item in comparison.deltas}
    assert set(states) == {"a", "b", "c"}
    assert states["b"].current_rank is None
    assert states["c"].previous_rank is None
    assert comparison.current_snapshot_hash == snapshot_hash(new)


def test_release_manifest_is_deterministic():
    artifact = ArtifactRecord(path="report.json", sha256="0" * 64, kind="report", size_bytes=10)
    first = build_release_manifest(
        version="1.0.0", benchmark_id="bench", benchmark_version="1", task_id="task", publication_id="pub", artifacts=[artifact]
    )
    second = build_release_manifest(
        version="1.0.0", benchmark_id="bench", benchmark_version="1", task_id="task", publication_id="pub", artifacts=[artifact]
    )
    assert first.release_id == second.release_id
    assert first.manifest_sha256 == second.manifest_sha256


def test_release_verification_detects_tampering(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("stable", encoding="utf-8")
    record = fingerprint_file(path, kind="report")
    manifest = build_release_manifest(
        version="1.0.0", benchmark_id="bench", benchmark_version="1", task_id="task", publication_id="pub", artifacts=[record]
    )
    assert verify_release_manifest(manifest, tmp_path) == []
    path.write_text("tampered", encoding="utf-8")
    errors = verify_release_manifest(manifest, tmp_path)
    assert any("sha256 mismatch" in error for error in errors)


def test_scorecard_ranks_models_across_snapshots():
    first = snapshot({"a": 0.9, "b": 0.8})
    second = snapshot({"a": 0.7, "b": 0.6})
    scorecard = build_scorecard([first, second])
    assert scorecard.models[0].model_id == "a"
    assert scorecard.models[0].n_benchmarks == 2
