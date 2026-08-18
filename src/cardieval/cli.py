"""Command-line entrypoint for CardiEval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark_package import load_package, verify_package_artifacts
from .bundle import build_bundle, save_bundle
from .evaluator import evaluate_submission, load_submission, save_report, sha256_file
from .integrity import ReleaseManifest, verify_release_manifest
from .models import BenchmarkManifest
from .publication import load_bundle, publish_leaderboard, save_snapshot
from .publication_history import compare_snapshots, load_snapshot, save_comparison
from .registry import BenchmarkTask


def _evaluate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cardieval", description="Independently evaluate a model submission")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--task-file")
    parser.add_argument("--task-id")
    parser.add_argument("--output", default="cardieval-report.json")
    parser.add_argument("--bundle-output")
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    submission_path = Path(args.submission)
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    submission = load_submission(submission_path)
    task = BenchmarkTask.model_validate_json(Path(args.task_file).read_text(encoding="utf-8")) if args.task_file else None
    if task is None and args.bundle_output:
        parser.error("--bundle-output requires --task-file")
    if task is not None and args.task_id and args.task_id != task.task_id:
        parser.error("--task-id does not match --task-file task_id")
    report = evaluate_submission(manifest, submission, model_id=args.model_id, task_contract=task)
    save_report(report, args.output)
    bundle_id = None
    fingerprint = None
    if args.bundle_output:
        bundle = build_bundle(manifest, report, task_id=task.task_id, submission_sha256=sha256_file(submission_path))
        save_bundle(bundle, args.bundle_output)
        bundle_id = bundle.bundle_id
        fingerprint = bundle.evaluation_fingerprint
    print(json.dumps({
        "ok": report.ok,
        "output": args.output,
        "bundle_output": args.bundle_output,
        "bundle_id": bundle_id,
        "evaluation_fingerprint": fingerprint,
        "task_id": report.task_id,
        "primary_metric": report.primary_metric,
        "primary_value": report.primary_value,
        "metrics": {m.name: m.value for m in report.metrics},
    }))
    return 0 if report.ok else 1


def _publish(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cardieval publish")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--bundles-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    task = BenchmarkTask.model_validate_json(Path(args.task_file).read_text(encoding="utf-8"))
    paths = sorted(Path(args.bundles_dir).glob("*.json"))
    bundles = [load_bundle(path) for path in paths]
    snapshot = publish_leaderboard(bundles, task)
    save_snapshot(snapshot, args.output)
    print(json.dumps({"ok": True, "output": args.output, "n_models": snapshot.n_models, "primary_metric": snapshot.primary_metric}))
    return 0


def _compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cardieval compare")
    parser.add_argument("--previous", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    comparison = compare_snapshots(load_snapshot(args.previous), load_snapshot(args.current))
    save_comparison(comparison, args.output)
    print(json.dumps({"ok": True, "output": args.output, "changed_models": comparison.changed_models}))
    return 0


def _verify(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cardieval verify")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    manifest = ReleaseManifest.model_validate_json(Path(args.manifest).read_text(encoding="utf-8"))
    errors = verify_release_manifest(manifest, args.root)
    print(json.dumps({"ok": not errors, "errors": errors}))
    return 0 if not errors else 1


def _verify_benchmark(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="cardieval verify-benchmark")
    parser.add_argument("--package", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    package = load_package(args.package)
    errors = verify_package_artifacts(package, args.root)
    print(json.dumps({
        "ok": not errors,
        "benchmark_id": package.benchmark_id,
        "version": package.version,
        "tasks": [task.task_id for task in package.tasks],
        "errors": errors,
    }))
    return 0 if not errors else 1


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "publish":
        return _publish(argv[1:])
    if argv and argv[0] == "compare":
        return _compare(argv[1:])
    if argv and argv[0] == "verify":
        return _verify(argv[1:])
    if argv and argv[0] == "verify-benchmark":
        return _verify_benchmark(argv[1:])
    return _evaluate(argv)


if __name__ == "__main__":
    raise SystemExit(main())
