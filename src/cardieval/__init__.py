"""CardiEval: independent evaluation for cardiac challenge models."""

from .calibration import brier_score, expected_calibration_error
from .comparison import compare_predictions
from .evaluator import EvaluationReport, evaluate_submission
from .models import BenchmarkManifest, MetricResult, ModelComparison, PredictionRecord, SubgroupResult

__all__ = [
    "BenchmarkManifest",
    "EvaluationReport",
    "MetricResult",
    "ModelComparison",
    "PredictionRecord",
    "SubgroupResult",
    "brier_score",
    "compare_predictions",
    "evaluate_submission",
    "expected_calibration_error",
]

__version__ = "0.2.0"
