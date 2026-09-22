"""Versioned CardiBridge exchange contracts for Agent/Vex -> CardiEval handoff."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark_package import BenchmarkPackage, load_package, validate_submission_against_package
from .models import PredictionRecord
from .provenance import canonical_json_hash

BridgeRole = Literal["agent", "vex", "eval"]
BridgePayloadType = Literal["challenge_population", "observation", "prediction_submission"]


class BridgeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = "1.0"
    role: BridgeRole
    payload_types: list[BridgePayloadType] = Field(min_length=1)
    task_types: list[str] = Field(min_length=1)
    max_schema_version: str = "1.0"

    @model_validator(mode="after")
    def validate_capabilities(self) -> "BridgeCapabilities":
        if len(set(self.payload_types)) != len(self.payload_types):
            raise ValueError("payload_types must be unique")
        if len(set(self.task_types)) != len(self.task_types):
            raise ValueError("task_types must be unique")
        if self.max_schema_version != self.protocol_version:
            raise ValueError("max_schema_version must equal protocol_version")
        return self


class BridgeEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    message_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_role: BridgeRole
    target_role: BridgeRole
    payload_type: BridgePayloadType
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict


class PredictionSubmission(BaseModel):
    """Portable submission payload accepted from CardiAgent/CardiVex adapters."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "1.0"
    model_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    predictions: list[PredictionRecord] = Field(min_length=1)


def _expected_message_id(envelope: BridgeEnvelope) -> str:
    return canonical_json_hash({
        "benchmark": f"{envelope.benchmark_id}@{envelope.benchmark_version}",
        "task_id": envelope.task_id,
        "payload_sha256": envelope.payload_sha256,
        "source_role": envelope.source_role,
    })


def build_submission_envelope(
    package: BenchmarkPackage,
    submission: PredictionSubmission,
    *,
    source_role: Literal["agent", "vex"],
) -> BridgeEnvelope:
    """Create a validated Agent/Vex -> Eval envelope tied to one benchmark task."""
    package.validate_contracts()
    task = next((item for item in package.tasks if item.task_id == submission.task_id), None)
    if task is None:
        raise ValueError(f"unknown task_id: {submission.task_id!r}")
    task.validate_manifest(package.manifest)
    validate_submission_against_package(package, submission.predictions, task_id=submission.task_id)
    payload = submission.model_dump(mode="json")
    digest = canonical_json_hash(payload)
    return BridgeEnvelope(
        message_id=canonical_json_hash({
            "benchmark": f"{package.benchmark_id}@{package.version}",
            "task_id": submission.task_id,
            "payload_sha256": digest,
            "source_role": source_role,
        }),
        source_role=source_role,
        target_role="eval",
        payload_type="prediction_submission",
        benchmark_id=package.benchmark_id,
        benchmark_version=package.version,
        task_id=submission.task_id,
        payload_sha256=digest,
        payload=payload,
    )


def validate_envelope(
    envelope: BridgeEnvelope,
    package: BenchmarkPackage,
    *,
    expected_source_role: BridgeRole | None = None,
) -> PredictionSubmission:
    """Validate an incoming envelope before it enters CardiEval."""
    package.validate_contracts()
    if expected_source_role is not None and envelope.source_role != expected_source_role:
        raise ValueError("bridge source role mismatch")
    if envelope.source_role == "eval":
        raise ValueError("CardiEval does not accept eval-originated submission envelopes")
    if envelope.target_role != "eval":
        raise ValueError("CardiEval accepts only envelopes targeted at eval")
    if envelope.benchmark_id != package.benchmark_id or envelope.benchmark_version != package.version:
        raise ValueError("bridge benchmark identity mismatch")
    if envelope.payload_type != "prediction_submission":
        raise ValueError("unsupported payload_type for CardiEval")
    payload_hash = canonical_json_hash(envelope.payload)
    if payload_hash != envelope.payload_sha256:
        raise ValueError("bridge payload integrity check failed")
    if _expected_message_id(envelope) != envelope.message_id:
        raise ValueError("bridge message_id integrity check failed")
    submission = PredictionSubmission.model_validate(envelope.payload)
    if submission.task_id != envelope.task_id:
        raise ValueError("bridge task_id mismatch")
    task = next((item for item in package.tasks if item.task_id == submission.task_id), None)
    if task is None:
        raise ValueError(f"unknown task_id: {submission.task_id!r}")
    task.validate_manifest(package.manifest)
    validate_submission_against_package(package, submission.predictions, task_id=submission.task_id)
    return submission


def negotiate_capabilities(local: BridgeCapabilities, remote: BridgeCapabilities) -> dict[str, list[str] | str]:
    """Return shared CardiBridge capabilities or fail closed on version mismatch."""
    if local.role == remote.role:
        raise ValueError("capability negotiation requires distinct roles")
    if local.protocol_version != remote.protocol_version:
        raise ValueError("no compatible CardiBridge protocol version")
    shared_payloads = sorted(set(local.payload_types) & set(remote.payload_types))
    shared_tasks = sorted(set(local.task_types) & set(remote.task_types))
    if not shared_payloads or not shared_tasks:
        raise ValueError("no shared CardiBridge capabilities")
    return {
        "protocol_version": local.protocol_version,
        "payload_types": shared_payloads,
        "task_types": shared_tasks,
    }


def load_package_for_bridge(path: str) -> BenchmarkPackage:
    return load_package(path)
