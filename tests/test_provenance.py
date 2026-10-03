from pathlib import Path

from cardieval.provenance import artifact_manifest, canonical_json_hash, evaluation_fingerprint


def test_canonical_json_hash_is_order_independent():
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})


def test_evaluation_fingerprint_changes_with_model():
    common = dict(
        benchmark_id="bench",
        benchmark_version="1",
        dataset_sha256="0" * 64,
        evaluator_version="0.4.0",
        config={"seed": 0},
    )
    assert evaluation_fingerprint(model_id="a", **common) != evaluation_fingerprint(model_id="b", **common)


def test_artifact_manifest_is_deterministic(tmp_path: Path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")
    one = artifact_manifest([second, first])
    two = artifact_manifest([first, second])
    assert one == two
    assert len(one) == 2
