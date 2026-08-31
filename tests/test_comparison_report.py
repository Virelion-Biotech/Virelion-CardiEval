from cardieval.comparison_report import build_comparison_report
from cardieval.metrics import accuracy


def test_comparison_report_is_fingerprintable():
    report = build_comparison_report(
        model_a="a",
        model_b="b",
        metric="accuracy",
        value_a=0.8,
        value_b=0.7,
        difference=0.1,
        ci_low=0.0,
        ci_high=0.2,
        direction="higher_is_better",
    )
    assert report.fingerprint
    assert report.model_a == "a"
