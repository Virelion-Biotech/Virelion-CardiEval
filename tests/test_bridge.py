from cardieval.bridge import (
    BridgeCapabilities,
    PredictionSubmission,
    build_submission_envelope,
    negotiate_capabilities,
    validate_envelope,
)
from cardieval.benchmark_package import BenchmarkPackage
from cardieval.models import BenchmarkManifest, PredictionRecord
from cardieval.registry import BenchmarkTask


def package():
    manifest = BenchmarkManifest(
        benchmark_id="bench",
        version="1",
        task="binary_classification",
        split="test",
        sample_ids=["a", "b"],
        dataset_sha256="0" * 64,
    )
    task = BenchmarkTask(
        benchmark_id="bench",
        version="1",
        task_id="detect",
        task_type="binary_classification",
        allowed_metrics=["accuracy"],
        primary_metric="accuracy",
        primary_direction="higher_is_better",
        splits=["test"],
    )
    return BenchmarkPackage(benchmark_id="bench", version="1", manifest=manifest, tasks=[task])


def submission():
    return PredictionSubmission(
        model_id="agent-model",
        task_id="detect",
        predictions=[
            PredictionRecord(sample_id="a", y_true=0, y_pred=0),
            PredictionRecord(sample_id="b", y_true=1, y_pred=1),
        ],
    )


def test_round_trip_envelope():
    envelope = build_submission_envelope(package(), submission(), source_role="agent")
    restored = validate_envelope(envelope, package(), expected_source_role="agent")
    assert restored.model_id == "agent-model"
    assert restored.task_id == "detect"


def test_tampered_payload_is_rejected():
    envelope = build_submission_envelope(package(), submission(), source_role="vex")
    tampered = envelope.model_copy(update={"payload": {**envelope.payload, "model_id": "tampered"}})
    try:
        validate_envelope(tampered, package())
    except ValueError as exc:
        assert "integrity" in str(exc)
    else:
        raise AssertionError("expected payload tampering to fail")


def test_capability_negotiation():
    agent = BridgeCapabilities(role="agent", payload_types=["prediction_submission"], task_types=["binary_classification"])
    evaluator = BridgeCapabilities(role="eval", payload_types=["prediction_submission"], task_types=["binary_classification", "regression"])
    negotiated = negotiate_capabilities(agent, evaluator)
    assert negotiated["payload_types"] == ["prediction_submission"]
    assert negotiated["task_types"] == ["binary_classification"]
