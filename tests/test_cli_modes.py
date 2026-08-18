from cardieval.cli import main
from cardieval.integrity import build_release_manifest, fingerprint_file


def test_verify_cli_success(monkeypatch, tmp_path, capsys):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("ok", encoding="utf-8")
    record = fingerprint_file(artifact, kind="json")
    manifest = build_release_manifest(
        version="1.0.0",
        benchmark_id="bench",
        benchmark_version="1",
        task_id="task",
        publication_id="pub",
        artifacts=[record.model_copy(update={"path": "artifact.json"})],
    )
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["cardieval", "verify", "--manifest", str(manifest_path), "--root", str(tmp_path)],
    )
    assert main() == 0
    assert '"ok": true' in capsys.readouterr().out
