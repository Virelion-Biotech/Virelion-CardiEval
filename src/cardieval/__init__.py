"""CardiEval: independent evaluation for cardiac challenge models."""

from .calibration import brier_score, expected_calibration_error
from .comparison import compare_predictions
from .evaluator import EvaluationReport, evaluate_submission
from .leaderboard import Leaderboard, LeaderboardEntry, build_leaderboard
from .models import BenchmarkManifest, MetricResult, ModelComparison, PredictionRecord, SubgroupResult
from .multiple_testing import benjamini_hochberg, bonferroni
from .ranking import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from .registry import BenchmarkTask, TaskRegistry
from .robustness import relative_drop, subgroup_robustness

__all__ = [
    "BenchmarkManifest",
    "BenchmarkTask",
    "EvaluationReport",
    "Leaderboard",
    "LeaderboardEntry",
    "MetricResult",
    "ModelComparison",
    "PredictionRecord",
    "SubgroupResult",
    "TaskRegistry",
    "benjamini_hochberg",
    "brier_score",
    "bonferroni",
    "build_leaderboard",
    "compare_predictions",
    "evaluate_submission",
    "expected_calibration_error",
    "hit_rate_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "relative_drop",
    "subgroup_robustness",
]

__version__ = "0.3.0"
