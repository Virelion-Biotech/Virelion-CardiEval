"""Command-line entrypoint for CardiEval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import build_bundle, save_bundle
from .evaluator import evaluate_submission, load_submission, save_report, sha256_file
from .models import BenchmarkManifest
from .publication import load_bundle as load_publication_bundle, publish_leaderboard, save_snapshot
from .publication_history import compare_snapshots, load_snapshot, save_comparison
from .registry import BenchmarkTask


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cardieval", description="Independently evaluate and publish model submissions"
    )
    parser.add_argument("--manifest", help="Benchmark manifest JSON")
    parser.add_argument("--submission", help="Prediction JSONL")
    parser.add_argument("--model-id", help="Stable model identifier")
    parser.add_argument("--task-file", help="Versioned BenchmarkTask JSON contract")
    parser.add_argument("--task-id", help="Task ID when no task file is supplied")
    parser.add_argument("--output", default="cardieval-report.json", help="Output report JSON")
    parser.add_argument("--bundle-output", help="Optional interoperable evaluation bundle JSON")
    parser.add_argument("--publish-bundle-dir", help="Directory containing SubmissionBundle JSON files")
    parser.add_argument("--snapshot-output", help="Output JSON for a published leaderboard snapshot")
    parser.add_argument("--compare-snapshot", help="Previous leaderboard snapshot JSON to compare against")
    parser.add_argument("--comparison-output", help="Output JSON for a historical publication comparison")
    args = parser.parse_args()

    if args.publish_bundle_dir or args.compare_snapshot:
        if not args.task_file:
            parser.error("publication operations require --task-file")
        task = BenchmarkTask.model_validate_json(Path(args.task_file).read_text(encoding="utf-8"))
        if args.publish_bundle_dir:
            if not args.snapshot_output:
                parser.error("--publish-bundle-dir requires --snapshot-output")
            bundle_paths = sorted(Path(args.publish_bundle_dir).glob("*.json"))
            if not bundle_paths:
                parser.error("no .json bundle files found in --publish-bundle-dir")
            bundles = [load_publication_bundle(path) for path in bundle_paths]
            snapshot = publish_leaderboard(bundles, task)
            save_snapshot(snapshot, args.snapshot_output)
            result = {
                "operation": "publish",
                "snapshot_output": args.snapshot_output,
                "benchmark_id": snapshot.benchmark_id,
                "task_id": snapshot.task_id,
                "primary_metric": snapshot.primary_metric,
                "n_models": snapshot.n_models,
            }
            if args.compare_snapshot:
                previous = load_snapshot(args.compare_snapshot)
                comparison = compare_snapshots(previous, snapshot)
                comparison_output = args.comparison_output or "cardieval-publication-comparison.json"
                save_comparison(comparison, comparison_output)
                result["comparison_output"] = comparison_output
                result["changed_models"] = comparison.changed_models
            print(json.dumps(result))
            return 0

        current = load_snapshot(args.snapshot_output) if args.snapshot_output else None
        if current is None:
            parser.error("--compare-snapshot alone requires --snapshot-output as the current snapshot")
        previous = load_snapshot(args.compare_snapshot)
        comparison = compare_snapshots(previous, current)
        comparison_output = args.comparison_output or "cardieval-publication-comparison.json"
        save_comparison(comparison, comparison_output)
        print(json.dumps({"operation": "compare", "comparison_output": comparison_output, "changed_models": comparison.changed_models}))
        return 0

    if not args.manifest or not args.submission or not args.model_id:
        parser.error("evaluation requires --manifest, --submission, and --model-id")

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
                "operation": "evaluate",
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
