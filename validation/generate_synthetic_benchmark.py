#!/usr/bin/env python3
"""Generate a reproducible synthetic benchmark for local CardiEval validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 500, seed: int = 42, out_dir: Path = Path("validation/data")) -> dict:
    if n < 50:
        raise ValueError("n must be >= 50 for meaningful subgroup / CI checks")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "submissions").mkdir(parents=True, exist_ok=True)

    sample_ids = [f"sample-{i:04d}" for i in range(1, n + 1)]
    latent = rng.normal(0.0, 1.0, size=n)
    subgroup = np.where(rng.random(n) < 0.55, "male", "female")
    prevalence_shift = np.where(subgroup == "male", 0.15, -0.10)
    p_true = _sigmoid(0.8 * latent + prevalence_shift)
    y_true = (rng.random(n) < p_true).astype(int)

    noise_base = rng.normal(0.0, 1.1, size=n)
    score_base = _sigmoid(0.55 * latent + noise_base)
    y_pred_base = (score_base >= 0.5).astype(int)

    noise_strong = rng.normal(0.0, 0.55, size=n)
    score_strong = _sigmoid(1.15 * latent + noise_strong)
    y_pred_strong = (score_strong >= 0.5).astype(int)

    def write_jsonl(path: Path, scores: np.ndarray, preds: np.ndarray) -> None:
        lines = []
        for sid, yp, sc in zip(sample_ids, preds, scores):
            lines.append(json.dumps(
                {
                    "sample_id": sid,
                    "y_pred": int(yp),
                    "score": float(round(sc, 6)),
                },
                separators=(",", ":"),
            ))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_jsonl(out_dir / "submissions" / "baseline.jsonl", score_base, y_pred_base)
    write_jsonl(out_dir / "submissions" / "strong.jsonl", score_strong, y_pred_strong)

    id_blob = "\n".join(sample_ids).encode("utf-8")
    dataset_sha256 = hashlib.sha256(id_blob).hexdigest()
    manifest = {
        "benchmark_id": "cardiac-synthetic-validation",
        "version": "1.0.0",
        "task": "binary_classification",
        "split": "test",
        "sample_ids": sample_ids,
        "dataset_sha256": dataset_sha256,
        "label_schema": {"0": "negative", "1": "positive"},
        "metadata": {
            "purpose": "local CardiEval validation (synthetic, reproducible)",
            "generator": "validation/generate_synthetic_benchmark.py",
            "seed": str(seed),
            "n": str(n),
            "ground_truth_policy": "evaluator-controlled synthetic labels",
            "not_clinical": "true",
        },
        "authoritative_labels": {sid: int(y) for sid, y in zip(sample_ids, y_true)},
        "authoritative_subgroups": {sid: str(sg) for sid, sg in zip(sample_ids, subgroup)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    task = {
        "benchmark_id": "cardiac-synthetic-validation",
        "version": "1.0.0",
        "task_id": "binary-challenge-detection",
        "task_type": "binary_classification",
        "allowed_metrics": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auroc",
            "auprc",
            "brier",
            "ece",
            "sensitivity",
            "specificity",
        ],
        "primary_metric": "auroc",
        "primary_direction": "higher_is_better",
        "splits": ["validation", "test"],
        "description": "Synthetic binary cardiac challenge detection task for local validation.",
        "requires_authoritative_labels": True,
    }
    (out_dir / "task.json").write_text(
        json.dumps(task, indent=2) + "\n", encoding="utf-8"
    )

    package = {
        "schema_version": "1.1",
        "benchmark_id": "cardiac-synthetic-validation",
        "version": "1.0.0",
        "manifest": manifest,
        "tasks": [task],
        "metadata": {
            "purpose": "strict synthetic CardiEval package integration test"
        },
    }
    (out_dir / "package.json").write_text(
        json.dumps(package, indent=2) + "\n", encoding="utf-8"
    )

    def rough_auroc(yt: np.ndarray, scores: np.ndarray) -> float:
        pos = scores[yt == 1]
        neg = scores[yt == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        greater = 0.0
        for p in pos:
            greater += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(greater / (len(pos) * len(neg)))

    meta = {
        "n": n,
        "seed": seed,
        "prevalence": float(y_true.mean()),
        "subgroup_counts": {
            "male": int((subgroup == "male").sum()),
            "female": int((subgroup == "female").sum()),
        },
        "approx_auroc": {
            "baseline": rough_auroc(y_true, score_base),
            "strong": rough_auroc(y_true, score_strong),
        },
        "note": "approx_auroc is a sanity-check estimate; CardiEval uses sklearn-based AUROC.",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote synthetic benchmark to {out_dir.resolve()}")
    print(
        f"  n={n}  seed={seed}  prevalence={meta['prevalence']:.3f}  "
        f"AUROC baseline={meta['approx_auroc']['baseline']:.3f} "
        f"strong={meta['approx_auroc']['strong']:.3f}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CardiEval validation benchmark"
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("validation/data"))
    args = parser.parse_args()
    try:
        generate(n=args.n, seed=args.seed, out_dir=args.out_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
