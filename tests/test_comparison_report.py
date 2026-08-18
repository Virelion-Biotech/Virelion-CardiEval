from cardiEval.comparison_report import build_comparison_report
from cardiEval.metrics import accuracy


def test_comparison_report_is_fingerprintable():
    report = build_comparison_report(
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        accuracy,
        metric_name="accuracy",
        direction="higher_is_better",
        benchmark_id="bench",
        benchmark_version="1",
        dataset_sha256="0" * 64,
        evaluator_version="0.5.0",
        model_a="a",
        model_b="b",
        n_resamples=100,
        seed=7,
    )
    assert len(report.fingerprint) == 64
    assert report.model_a == "a"
    assert report.model_b == "b"
    assert report.comparison["metric"] == "accuracy"
