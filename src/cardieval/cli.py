"""Command-line entrypoint for CardiEval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import evaluate_submission, load_submission, save_report
from .models import BenchmarkManifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="cardieval", description="Independently evaluate a model submission")
    parser.add_argument("--manifest", required=True, help="Benchmark manifest JSON")
    parser.add_argument("--submission", required=True, help="Prediction JSONL")
    parser.add_argument("--model-id", required=True, help="Stable model identifier")
    parser.add_argument("--output", default="cardieval-report.json", help="Output report JSON")
    args = parser.parse_args()

    manifest = BenchmarkManifest.model_validate_json(Path(args.manifest).read_text(encoding="utf-8"))
    submission = load_submission(args.submission)
    report = evaluate_submission(manifest, submission, model_id=args.model_id)
    save_report(report, args.output)
    print(json.dumps({"ok": report.ok, "output": args.output, "metrics": {m.name: m.value for m in report.metrics}}))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
