"""Command-line entrypoint for CardiEval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import build_bundle, save_bundle
from .evaluator import evaluate_submission, load_submission, save_report, sha256_file
from .models import BenchmarkManifest
from .registry import BenchmarkTask


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cardieval", description="Independently evaluate a model submission"
    )
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSON")
    parser.add_argument("--submission", required=True, help="Prediction JSONL")
    parser.add_argument("--model-id", required=True, help="Stable model identifier")
    parser.add_argument("--task-file", help="Versioned BenchmarkTask JSON contract")
    parser.add_argument("--task-id", help="Task ID when no task file is supplied")
    parser.add_argument("--output", default="cardieval-report.json", help="Output report JSON")
    parser.add_argument("--bundle-output", help="Optional interoperable evaluation bundle JSON")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    submission_path = Path(args.submission)
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    submission = load_submission(submission_path)
    task = (
        BenchmarkTask.model_validate_json(Path(args.task_file).read_text(encoding="utf-8"))
        if args.task_file
        else None
    )
    if task is None and args.bundle_output:
        parser.error("--bundle-output requires --task-file")
    if task is not None and args.task_id and args.task_id != task.task_id:
        parser.error("--task-id does not match --task-file task_id")

    report = evaluate_submission(
        manifest,
        submission,
        model_id=args.model_id,
        task_contract=task,
    )
    save_report(report, args.output)

    bundle_path = None
    bundle_id = None
    evaluation_fingerprint = None
    if args.bundle_output:
        bundle = build_bundle(
            manifest,
            report,
            task_id=task.task_id,
            submission_sha256=sha256_file(submission_path),
        )
        save_bundle(bundle, args.bundle_output)
        bundle_path = args.bundle_output
        bundle_id = bundle.bundle_id
        evaluation_fingerprint = bundle.evaluation_fingerprint

    print(
        json.dumps(
            {
                "ok": report.ok,
                "output": args.output,
                "bundle_output": bundle_path,
                "bundle_id": bundle_id,
                "evaluation_fingerprint": evaluation_fingerprint,
                "task_id": report.task_id,
                "primary_metric": report.primary_metric,
                "primary_value": report.primary_value,
                "metrics": {m.name: m.value for m in report.metrics},
            }
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
