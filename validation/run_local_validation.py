#!/usr/bin/env python3
"""
Run a full local CardiEval validation on the synthetic (or your own) benchmark.

Prerequisites
-------------
From the repository root:

  pip install -e ".[dev]"          # or: pip install numpy pydantic scipy scikit-learn pytest
  python validation/generate_synthetic_benchmark.py

Then:

  python validation/run_local_validation.py

What it does
------------
1. Loads manifest + task + two submissions (baseline + strong).
2. Runs evaluate_submission for each model.
3. Writes reports + bundles under validation/outputs/.
4. Prints a short summary and a pass/fail checklist.
5. Optionally runs pytest if --pytest is given.

You can point it at your own data with --data-dir.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _ensure_package() -> None:
    """Make src/cardieval importable when run from a source checkout."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def run(data_dir: Path, out_dir: Path, run_pytest: bool) -> int:
    _ensure_package()

    from cardieval.bundle import build_bundle, save_bundle
    from cardieval.evaluator import evaluate_submission, load_submission, save_report, sha256_file
    from cardieval.models import BenchmarkManifest
    from cardieval.registry import BenchmarkTask

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = out_dir / "bundles"
    bundles_dir.mkdir(exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    task_path = data_dir / "task.json"
    if not manifest_path.exists() or not task_path.exists():
        print(f"Missing {manifest_path} or {task_path}")
        print("Run:  python validation/generate_synthetic_benchmark.py")
        return 1

    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    task = BenchmarkTask.model_validate_json(task_path.read_text(encoding="utf-8"))

    submissions = {
        "baseline": data_dir / "submissions" / "baseline.jsonl",
        "strong": data_dir / "submissions" / "strong.jsonl",
    }
    for name, path in submissions.items():
        if not path.exists():
            print(f"Missing submission: {path}")
            return 1

    summary = {"models": {}, "ok": True}

    for model_id, sub_path in submissions.items():
        records = load_submission(sub_path)
        report = evaluate_submission(
            manifest,
            records,
            model_id=model_id,
            task_contract=task,
        )
        report_path = out_dir / f"report_{model_id}.json"
        save_report(report, report_path)

        bundle = build_bundle(
            manifest,
            report,
            task_id=task.task_id,
            submission_sha256=sha256_file(sub_path),
        )
        bundle_path = bundles_dir / f"bundle_{model_id}.json"
        save_bundle(bundle, bundle_path)

        metrics = {m.name: m.value for m in report.metrics}
        summary["models"][model_id] = {
            "ok": report.ok,
            "primary_metric": report.primary_metric,
            "primary_value": report.primary_value,
            "metrics": metrics,
            "n_subgroups": len(report.subgroups),
            "warnings": report.warnings,
            "report": str(report_path),
            "bundle": str(bundle_path),
            "bundle_id": bundle.bundle_id,
            "evaluation_fingerprint": bundle.evaluation_fingerprint,
        }
        if not report.ok:
            summary["ok"] = False

        print(f"\n=== {model_id} ===")
        print(f"  primary ({report.primary_metric}): {report.primary_value}")
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "auprc", "brier", "ece"):
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")
        if report.warnings:
            print(f"  warnings: {len(report.warnings)}")
        print(f"  report -> {report_path}")
        print(f"  bundle -> {bundle_path}")

    # Sanity checks (synthetic data should show strong > baseline on AUROC)
    base_auroc = summary["models"]["baseline"]["metrics"].get("auroc")
    strong_auroc = summary["models"]["strong"]["metrics"].get("auroc")
    if base_auroc is not None and strong_auroc is not None:
        if strong_auroc > base_auroc:
            print("\n[PASS] strong model AUROC > baseline AUROC (expected for synthetic generator)")
        else:
            print("\n[WARN] strong model AUROC did not exceed baseline; check generator seed / n")
            summary["ok"] = False

    # Try optional publication if available
    try:
        from cardieval.publication import load_bundle, publish_leaderboard, save_snapshot

        bundles = [load_bundle(p) for p in sorted(bundles_dir.glob("*.json"))]
        snapshot = publish_leaderboard(bundles, task)
        snap_path = out_dir / "leaderboard.json"
        save_snapshot(snapshot, snap_path)
        summary["leaderboard"] = str(snap_path)
        print(f"\nLeaderboard snapshot -> {snap_path}  (n_models={snapshot.n_models})")
    except Exception as exc:
        print(f"\n[INFO] publication step skipped or failed: {exc}")

    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    if run_pytest:
        print("\n--- running pytest ---")
        root = Path(__file__).resolve().parents[1]
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=root)
        summary["pytest_exit_code"] = rc
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if rc != 0:
            print(f"pytest exited with code {rc}")
            return rc

    print("\nDone. Local validation artifacts are under:", out_dir.resolve())
    return 0 if summary["ok"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CardiEval validation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("validation/data"),
        help="Directory containing manifest.json, task.json, submissions/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation/outputs"),
        help="Where to write reports and bundles",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Also run the package test suite after evaluation",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.data_dir, args.out_dir, args.pytest))


if __name__ == "__main__":
    main()
