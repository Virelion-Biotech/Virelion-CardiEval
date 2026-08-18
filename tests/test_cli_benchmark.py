from cardieval.cli import main


def test_verify_benchmark_cli(tmp_path, monkeypatch, capsys):
    package_path = tmp_path / "package.json"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("cardieval", encoding="utf-8")
    package_path.write_text(
        '{'
        '"benchmark_id":"bench",'
        '"version":"1",'
        '"manifest":{'
        '"benchmark_id":"bench","version":"1","task":"binary_classification",'
        '"split":"test","sample_ids":["a"],"dataset_sha256":"' + '0' * 64 + '"},'
        '"tasks":[{'
        '"benchmark_id":"bench","version":"1","task_id":"detect",'
        '"task_type":"binary_classification","allowed_metrics":["accuracy"],'
        '"primary_metric":"accuracy","primary_direction":"higher_is_better",'
        '"splits":["test"]}],'
        '"artifacts":[{'
        '"path":"artifact.txt","sha256":"' + __import__('hashlib').sha256(b'cardieval').hexdigest() + '",'
        '"kind":"text","size_bytes":9}]}' ,
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["cardieval", "verify-benchmark", "--package", str(package_path), "--root", str(tmp_path)])
    assert main() == 0
    assert '"ok": true' in capsys.readouterr().out
