import numpy as np

from cardieval.comparison import compare_predictions
from cardieval.metrics import accuracy, mae


def test_paired_accuracy_comparison_is_reproducible() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    pred_a = np.array([0, 1, 0, 0, 1, 0])
    pred_b = np.array([0, 1, 1, 0, 0, 0])
    result_a = compare_predictions(
        y_true,
        pred_a,
        pred_b,
        accuracy,
        metric_name="accuracy",
        n_resamples=300,
        seed=7,
    )
    result_b = compare_predictions(
        y_true,
        pred_a,
        pred_b,
        accuracy,
        metric_name="accuracy",
        n_resamples=300,
        seed=7,
    )
    assert result_a == result_b
    assert result_a.winner == "model_a"
    assert result_a.wilcoxon_pvalue is None


def test_wilcoxon_requires_explicit_samplewise_score() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    pred_a = np.array([0.0, 1.0, 2.0, 2.0])
    pred_b = np.array([0.5, 1.0, 2.5, 3.5])
    result = compare_predictions(
        y_true,
        pred_a,
        pred_b,
        mae,
        metric_name="mae",
        samplewise_score=lambda y, p: np.abs(y - p),
        n_resamples=300,
        seed=4,
    )
    assert result.wilcoxon_pvalue is not None
