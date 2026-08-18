# CardiBench ↔ CardiEval Integration Contract

CardiBench owns benchmark curation and release identity. CardiEval owns independent evaluation and publication.

## Benchmark package

A package is a JSON contract containing:

- `benchmark_id` and release `version`
- an exact `BenchmarkManifest`
- one or more `BenchmarkTask` definitions
- optional integrity records for package artifacts
- optional string metadata

CardiEval requires the package ID/version to match the manifest and requires every task to validate against the manifest.

## Submission flow

```text
CardiBench package
      ↓
verify-benchmark
      ↓
model submission JSONL
      ↓
validate sample IDs + task contract
      ↓
evaluate_submission
      ↓
EvaluationReport
      ↓
SubmissionBundle
      ↓
leaderboard publication
```

## Integrity

Package artifacts may be fingerprinted with SHA-256 and declared with their size. `verify-benchmark` checks the declaration against the local filesystem before evaluation.

This is integrity verification, not digital signature verification. Signing and key management belong outside this schema.

## Required invariants

A valid package must satisfy:

1. Package benchmark ID = manifest benchmark ID.
2. Package release version = manifest version.
3. Every task references the same benchmark ID/version.
4. Every task's type and permitted split are compatible with the manifest.
5. Task IDs are unique within the package.
6. Artifact paths are unique.
7. A submission contains exactly the manifest's sample IDs for the selected task.
8. Duplicate or unknown sample IDs are rejected.

## CLI

```bash
cardieval verify-benchmark \
  --package benchmark-package.json \
  --root ./benchmark-release
```

Then evaluate against the package's manifest/task files and emit a standard `SubmissionBundle` for publication.
