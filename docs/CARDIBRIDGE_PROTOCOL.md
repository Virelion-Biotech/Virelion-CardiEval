# CardiBridge Protocol

CardiBridge is the exchange boundary between Virelion components and CardiEval. The protocol is intentionally narrow: it transports a typed prediction submission, preserves benchmark/task identity, and verifies payload integrity before evaluation.

## Roles

`agent` produces a challenge-aware prediction submission; `vex` may transform observations into a prediction submission; `eval` is the accepting evaluator.

## Envelope

A `BridgeEnvelope` contains:

- protocol/schema version
- message ID
- source and target roles
- payload type
- benchmark ID/version
- task ID
- canonical SHA-256 payload digest
- typed payload

CardiEval currently accepts only `prediction_submission` envelopes targeted at `eval`.

## Submission payload

`PredictionSubmission` contains a stable `model_id`, `task_id`, and the same `PredictionRecord` objects used by the native evaluator. This means bridge-originated submissions are subject to the identical sample-set and task-contract validation as local submissions.

## Fail-closed validation

Before a bridge submission can enter evaluation, CardiEval verifies:

1. the package contract;
2. source/target role expectations;
3. benchmark ID/version;
4. supported payload type;
5. canonical payload hash;
6. payload task ID;
7. task existence and manifest compatibility.

A tampered or incompatible envelope is rejected rather than normalized silently.

## Capability negotiation

`BridgeCapabilities` advertises supported payload types and task types. `negotiate_capabilities()` returns the intersection or fails if there is no compatible payload/task combination.

## CLI validation

```bash
cardieval bridge-validate \
  --package benchmark-package.json \
  --envelope submission-envelope.json \
  --source-role agent
```

This validates the envelope without running the full evaluator.

## Versioning

The protocol begins at schema `1.0`. Producers should include the schema version and must not assume unsupported fields will be ignored; CardiEval uses strict Pydantic models for protocol objects.

## Security boundary

The payload SHA-256 value provides integrity detection, not authenticity. Digital signatures, key distribution, identity, authorization, and transport security belong to the deployment layer rather than the evaluation schema itself.
