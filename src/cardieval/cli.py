"""Command-line entrypoint for CardiEval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import build_bundle, save_bundle
from .evaluator import evaluate_submission, load_submission, save_report, sha256_file
from .models import BenchmarkManifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="cardieval", description="Independently evaluate a model submission")
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSON")
    parser.add_argument("--submission", required=True, help="Prediction JSONL")
    parser.add_argument("--model-id", required=True, help="Stable model identifier")
    parser.add_argument("--task-id", default="default", help="Registered benchmark task identifier")
    parser.add_argument("--output", default="cardieval-report.json", help="Output report JSON")
    parser.add_argument("--bundle-output", help="Optional interoperable evaluation bundle JSON")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    submission_path = Path(args.submission)
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    submission = load_submission(submission_path)
    report = evaluate_submission(manifest, submission, model_id=args.model_id)
    save_report(report, args.output)

    bundle_path = None
    if args.bundle_output:
        bundle = build_bundle(
            manifest,
            report,
            task_id=args.task_id,
            submission_sha256=sha256_file(submission_path),
        )
        save_bundle(bundle, args.bundle_output)
        bundle_path = args.bundle_output

    print(
        json.dumps(
            {
                "ok": report.ok,
                "output": args.output,
                "bundle_output": bundle_path,
                "metrics": {m.name: m.value for m in report.metrics},
                "evaluation_fingerprint": (
                    build_bundle(
                        manifest,
                        report,
                        task_id=args.task_id,
                        submission_sha256=sha256_file(submission_path),
                    ).evaluation_fingerprint
                    if bundle_path
                    else None
                ),
            }
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
