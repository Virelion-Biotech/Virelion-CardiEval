from cardieval.benchmark_package import BenchmarkPackage
from cardieval.models import BenchmarkManifest
from cardieval.pipeline import run_evaluation
from cardieval.registry import BenchmarkTask


def test_end_to_end_pipeline(tmp_path):
    package = BenchmarkPackage(
        benchmark_id="bench",
        version="1",
        manifest=BenchmarkManifest(
            benchmark_id="bench",
            version="1",
            task="binary_classification",
            split="test",
            sample_ids=["a", "b"],
            dataset_sha256="0" * 64,
        ),
        tasks=[
            BenchmarkTask(
                benchmark_id="bench",
                version="1",
                task_id="detect",
                task_type="binary_classification",
                allowed_metrics=["accuracy", "balanced_accuracy", "macro_f1"],
                primary_metric="accuracy",
                primary_direction="higher_is_better",
                splits=["test"],
            )
        ],
    )
    package_path = tmp_path / "package.json"
    package_path.write_text(package.model_dump_json(), encoding="utf-8")
    submission = tmp_path / "submission.jsonl"
    submission.write_text(
        '{"sample_id":"a","y_true":0,"y_pred":0}\n'
        '{"sample_id":"b","y_true":1,"y_pred":1}\n',
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    bundle_path = tmp_path / "bundle.json"
    run_path = tmp_path / "run.json"
    run = run_evaluation(
        package_path=package_path,
        package_root=tmp_path,
        submission_path=submission,
        model_id="demo",
        task_id="detect",
        report_path=report_path,
        bundle_path=bundle_path,
        run_manifest_path=run_path,
        require_artifact_verification=False,
    )
    assert len(run.run_id) == 64
    assert report_path.exists()
    assert bundle_path.exists()
    assert run_path.exists()
