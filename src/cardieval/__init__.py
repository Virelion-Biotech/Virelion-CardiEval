"""CardiEval: independent evaluation for cardiac challenge models."""

from .calibration import brier_score, expected_calibration_error
from .calibration_curves import CalibrationBin, calibration_curve
from .comparison import compare_predictions
from .confidence import paired_difference_ci
from .evaluator import EvaluationReport, evaluate_submission
from .leaderboard import Leaderboard, LeaderboardEntry, build_leaderboard
from .models import BenchmarkManifest, MetricResult, ModelComparison, PredictionRecord, SubgroupResult
from .multiple_testing import benjamini_hochberg, bonferroni
from .ranking import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from .provenance import artifact_manifest, canonical_json_hash, evaluation_fingerprint, sha256_text
from .registry import BenchmarkTask, TaskRegistry
from .robustness import relative_drop, subgroup_robustness
from .stress import StressResult, aggregate_stress, compare_stress

__all__ = [
    "BenchmarkManifest",
    "BenchmarkTask",
    "CalibrationBin",
    "CalibrationBin",
    "EvaluationReport",
    "Leaderboard",
    "LeaderboardEntry",
    "MetricResult",
    "ModelComparison",
    "PredictionRecord",
    "StressResult",
    "SubgroupResult",
    "TaskRegistry",
    "aggregate_stress",
    "artifact_manifest",
    "benjamini_hochberg",
    "brier_score",
    "canonical_json_hash",
    "bonferroni",
    "build_leaderboard",
    "calibration_curve",
    "compare_predictions",
    "compare_stress",
    "evaluate_submission",
    "evaluation_fingerprint",
    "expected_calibration_error",
    "hit_rate_at_k",
    "ndcg_at_k",
    "paired_difference_ci",
    "reciprocal_rank",
    "relative_drop",
    "sha256_text",
    "subgroup_robustness",
]

__version__ = "0.5.0"
