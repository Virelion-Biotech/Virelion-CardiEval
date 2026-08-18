from cardieval.leaderboard import build_leaderboard
from cardieval.multiple_testing import benjamini_hochberg, bonferroni
from cardieval.models import EvaluationReport, MetricResult


def report(model_id: str, score: float) -> EvaluationReport:
    return EvaluationReport(
        evaluator_version="0.2.0",
        benchmark_id="bench",
        benchmark_version="1.0",
        task="classification",
        split="test",
        model_id=model_id,
        metrics=[MetricResult(
            name="macro_f1",
            value=score,
            n=10,
            direction="higher_is_better",
        )],
    )


def test_leaderboard_ranks_and_averages_models():
    board = build_leaderboard(
        [report("a", 0.8), report("a", 0.9), report("b", 0.85)],
        metric="macro_f1",
        direction="higher_is_better",
    )
    assert [entry.model_id for entry in board.entries] == ["a", "b"]
    assert board.entries[0].score == 0.85
    assert board.entries[0].n_reports == 2
    assert board.entries[0].rank == 1


def test_pvalue_corrections_preserve_order():
    p = [0.01, 0.02, 0.5]
    assert bonferroni(p) == [0.03, 0.06, 1.0]
    adjusted = benjamini_hochberg(p)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
