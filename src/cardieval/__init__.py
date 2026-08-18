"""CardiEval: independent evaluation for cardiac challenge models."""

from .benchmark_package import BenchmarkArtifact, BenchmarkPackage, fingerprint_directory, load_package, validate_submission_against_package, verify_package_artifacts
from .bridge import BridgeCapabilities, BridgeEnvelope, PredictionSubmission, build_submission_envelope, negotiate_capabilities, validate_envelope
from .bundle import SubmissionBundle, build_bundle, save_bundle
from .calibration import brier_score, expected_calibration_error
from .calibration_curves import CalibrationBin, calibration_curve
from .comparison import compare_predictions
from .comparison_report import ComparisonReport, build_comparison_report
from .confidence import paired_difference_ci
from .decision import ComparisonDecision, QualityGate, ReleaseGateReport, decide_comparison, evaluate_release_gates
from .diagnostics import cohen_kappa, confusion_matrix_counts, matthews_correlation, negative_predictive_value, positive_predictive_value, sensitivity, specificity
from .evaluator import EvaluationReport, evaluate_submission
from .integrity import ArtifactRecord, ReleaseManifest, build_release_manifest, fingerprint_file, verify_release_manifest
from .leaderboard import Leaderboard, LeaderboardEntry, build_leaderboard
from .models import BenchmarkManifest, MetricResult, ModelComparison, PredictionRecord, SubgroupResult
from .multiple_testing import benjamini_hochberg, bonferroni
from .pipeline import run_evaluation
from .publication import LeaderboardSnapshot, ingest_bundles, load_bundle, publish_leaderboard, save_snapshot
from .publication_history import LeaderboardDelta, PublicationComparison, compare_snapshots, load_snapshot, save_comparison, snapshot_hash
from .ranking import hit_rate_at_k, ndcg_at_k, reciprocal_rank
from .provenance import artifact_manifest, canonical_json_hash, evaluation_fingerprint, sha256_text
from .registry import BenchmarkTask, TaskRegistry
from .robustness import relative_drop, subgroup_robustness
from .run_manifest import EvaluationRunManifest, build_run_manifest, save_run_manifest
from .scorecard import BenchmarkScore, ModelScorecard, Scorecard, build_scorecard
from .stress import StressResult, aggregate_stress, compare_stress

__all__ = [
    "ArtifactRecord", "BenchmarkArtifact", "BenchmarkManifest", "BenchmarkPackage", "BenchmarkScore",
    "BenchmarkTask", "BridgeCapabilities", "BridgeEnvelope", "CalibrationBin", "ComparisonDecision",
    "ComparisonReport", "EvaluationReport", "EvaluationRunManifest", "Leaderboard", "LeaderboardDelta",
    "LeaderboardEntry", "LeaderboardSnapshot", "MetricResult", "ModelComparison", "ModelScorecard",
    "PredictionRecord", "PredictionSubmission", "PublicationComparison", "QualityGate", "ReleaseGateReport",
    "ReleaseManifest", "Scorecard", "StressResult", "SubgroupResult", "SubmissionBundle", "TaskRegistry",
    "aggregate_stress", "artifact_manifest", "benjamini_hochberg", "brier_score", "bonferroni", "build_bundle",
    "build_comparison_report", "build_leaderboard", "build_release_manifest", "build_run_manifest", "build_scorecard",
    "build_submission_envelope", "calibration_curve", "canonical_json_hash", "cohen_kappa", "compare_predictions",
    "compare_snapshots", "compare_stress", "confusion_matrix_counts", "decide_comparison", "evaluate_release_gates",
    "evaluate_submission", "evaluation_fingerprint", "fingerprint_directory", "fingerprint_file", "hit_rate_at_k",
    "ingest_bundles", "load_bundle", "load_package", "load_snapshot", "matthews_correlation", "negotiate_capabilities",
    "negative_predictive_value", "ndcg_at_k", "paired_difference_ci", "positive_predictive_value", "publish_leaderboard",
    "reciprocal_rank", "relative_drop", "run_evaluation", "save_bundle", "save_comparison", "save_run_manifest",
    "save_snapshot", "sensitivity", "sha256_text", "snapshot_hash", "specificity", "subgroup_robustness",
    "validate_envelope", "validate_submission_against_package", "verify_package_artifacts", "verify_release_manifest",
]

__version__ = "1.4.0"
