#!/usr/bin/env python3
"""Run a strict end-to-end CardiEval validation on the synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _ensure_package() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def run(data_dir: Path, out_dir: Path, run_pytest: bool) -> int:
    _ensure_package()

    from cardieval.benchmark_package import load_package
    from cardieval.evaluator import load_submission
    from cardieval.models import EvaluationReport
    from cardieval.pipeline import run_evaluation
    from cardieval.publication import load_bundle, publish_leaderboard, save_snapshot

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = out_dir / "bundles"
    bundles_dir.mkdir(exist_ok=True)

    package_path = data_dir / "package.json"
    if not package_path.exists():
        print("Missing package.json; run generate_synthetic_benchmark.py")
        return 1

    package = load_package(package_path)
    task = next(item for item in package.tasks if item.task_id == package.tasks[0].task_id)
    task_id = task.task_id

    submissions = {
        "baseline": data_dir / "submissions" / "baseline.jsonl",
        "strong": data_dir / "submissions" / "strong.jsonl",
    }
    summary = {"models": {}, "ok": True}
    for model_id, sub_path in submissions.items():
        if not sub_path.exists():
            print(f"Missing submission: {sub_path}")
            return 1

        records = load_submission(sub_path)
        if any(record.y_true is not None for record in records):
            print(f"[FAIL] protected submission {model_id} unexpectedly contains y_true")
            summary["ok"] = False

        report_path = out_dir / f"report_{model_id}.json"
        bundle_path = bundles_dir / f"bundle_{model_id}.json"
        run_path = out_dir / f"run_{model_id}.json"

        run_evaluation(
            package_path=package_path,
            package_root=data_dir,
            submission_path=sub_path,
            model_id=model_id,
            task_id=task_id,
            report_path=report_path,
            bundle_path=bundle_path,
            run_manifest_path=run_path,
            require_artifact_verification=True,
        )

        report = EvaluationReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        bundle = load_bundle(bundle_path)
        bundle.verify_integrity()

        metrics = {m.name: m.value for m in report.metrics}
        summary["models"][model_id] = {
            "ok": report.ok,
            "ground_truth_source": report.ground_truth_source,
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
        if not report.ok or report.ground_truth_source != "benchmark_manifest":
            summary["ok"] = False

    base_auroc = summary["models"]["baseline"]["metrics"].get("auroc")
    strong_auroc = summary["models"]["strong"]["metrics"].get("auroc")
    if base_auroc is None or strong_auroc is None or strong_auroc <= base_auroc:
        print("[FAIL] synthetic AUROC sanity check failed")
        summary["ok"] = False
    else:
        print("[PASS] synthetic AUROC sanity check")

    try:
        bundles = [load_bundle(p) for p in sorted(bundles_dir.glob("*.json"))]
        snapshot = publish_leaderboard(bundles, task)
        snap_path = out_dir / "leaderboard.json"
        save_snapshot(snapshot, snap_path)
        summary["leaderboard"] = str(snap_path)
        print(f"[PASS] publication snapshot created (n_models={snapshot.n_models})")
    except Exception as exc:
        print(f"[FAIL] publication: {exc}")
        summary["ok"] = False

    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if run_pytest:
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q"])
        summary["pytest_exit_code"] = rc
        summary["ok"] = summary["ok"] and rc == 0
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if rc != 0:
            return rc

    print(f"Validation summary: {summary_path}")
    return 0 if summary["ok"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict CardiEval local validation")
    parser.add_argument("--data-dir", type=Path, default=Path("validation/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("validation/outputs"))
    parser.add_argument("--pytest", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.data_dir, args.out_dir, args.pytest))


if __name__ == "__main__":
    main()
