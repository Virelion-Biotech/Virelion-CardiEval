from cardieval.evaluator import evaluate_submission
from cardieval.models import BenchmarkManifest, PredictionRecord


def test_ranking_task_is_evaluated():
    manifest = BenchmarkManifest(
        benchmark_id="ranking-demo",
        version="1",
        task="ranking",
        split="test",
        sample_ids=["a", "b", "c", "d"],
        dataset_sha256="2" * 64,
    )
    records = [
        PredictionRecord(sample_id="a", y_true=3, y_pred=0, score=0.9),
        PredictionRecord(sample_id="b", y_true=0, y_pred=0, score=0.2),
        PredictionRecord(sample_id="c", y_true=2, y_pred=0, score=0.8),
        PredictionRecord(sample_id="d", y_true=1, y_pred=0, score=0.7),
    ]
    report = evaluate_submission(manifest, records, model_id="ranker")
    names = {metric.name for metric in report.metrics}
    assert {"mrr", "hit_rate@10", "ndcg@10"}.issubset(names)
    assert report.ok
