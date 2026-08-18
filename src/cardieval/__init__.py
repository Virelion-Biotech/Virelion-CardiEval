"""CardiEval: independent evaluation for cardiac challenge models."""

from .evaluator import EvaluationReport, evaluate_submission
from .models import BenchmarkManifest, MetricResult, PredictionRecord

__all__ = [
    "BenchmarkManifest",
    "EvaluationReport",
    "MetricResult",
    "PredictionRecord",
    "evaluate_submission",
]

__version__ = "0.1.0"
