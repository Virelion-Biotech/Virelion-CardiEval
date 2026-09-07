#!/usr/bin/env python3
"""
Generate a reproducible synthetic cardiac-style binary classification benchmark
for local CardiEval validation.

Produces:
  - manifest.json
  - task.json
  - submissions/baseline.jsonl   (weaker model)
  - submissions/strong.jsonl     (stronger model)
  - meta.json                    (expected approximate metrics for sanity checks)

Usage:
  python validation/generate_synthetic_benchmark.py
  python validation/generate_synthetic_benchmark.py --n 800 --seed 42 --out-dir validation/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n: int = 500,
    seed: int = 42,
    out_dir: Path = Path("validation/data"),
) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "submissions").mkdir(parents=True, exist_ok=True)

    sample_ids = [f"sample-{i:04d}" for i in range(1, n + 1)]

    # Synthetic latent risk + two demographic-like subgroups
    latent = rng.normal(0.0, 1.0, size=n)
    subgroup = np.where(rng.random(n) < 0.55, "male", "female")
    # Mild prevalence difference by subgroup (realistic for many cardiac tasks)
    prevalence_shift = np.where(subgroup == "male", 0.15, -0.10)
    p_true = _sigmoid(0.8 * latent + prevalence_shift)
    y_true = (rng.random(n) < p_true).astype(int)

    # Baseline model: noisier scores
    noise_base = rng.normal(0.0, 1.1, size=n)
    score_base = _sigmoid(0.55 * latent + noise_base)
    y_pred_base = (score_base >= 0.5).astype(int)

    # Strong model: better signal, less noise
    noise_strong = rng.normal(0.0, 0.55, size=n)
    score_strong = _sigmoid(1.15 * latent + noise_strong)
    y_pred_strong = (score_strong >= 0.5).astype(int)

    def write_jsonl(path: Path, scores: np.ndarray, preds: np.ndarray) -> None:
        lines = []
        for sid, yt, yp, sc, sg in zip(sample_ids, y_true, preds, scores, subgroup):
            lines.append(
                json.dumps(
                    {
                        "sample_id": sid,
                        "y_true": int(yt),
                        "y_pred": int(yp),
                        "score": float(round(sc, 6)),
                        "subgroup": str(sg),
                    },
                    separators=(",", ":"),
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_jsonl(out_dir / "submissions" / "baseline.jsonl", score_base, y_pred_base)
    write_jsonl(out_dir / "submissions" / "strong.jsonl", score_strong, y_pred_strong)

    # Deterministic placeholder hash of the sample ID list (not real data bytes)
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
            "leakage_policy": "labels and scores are generated together; not for clinical use",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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
    }
    (out_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")

    # Rough expected metrics (for human sanity checks only)
    def rough_auroc(yt: np.ndarray, scores: np.ndarray) -> float:
        # Simple Mann-Whitney style estimate without sklearn dependency
        pos = scores[yt == 1]
        neg = scores[yt == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # P(score_pos > score_neg) + 0.5 * P(equal)
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
        "note": "approx_auroc is a simple rank estimate; CardiEval uses sklearn-based AUROC.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote synthetic benchmark to {out_dir.resolve()}")
    print(f"  n={n}  seed={seed}  prevalence={meta['prevalence']:.3f}")
    print(f"  approx AUROC  baseline={meta['approx_auroc']['baseline']:.3f}  strong={meta['approx_auroc']['strong']:.3f}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CardiEval validation benchmark")
    parser.add_argument("--n", type=int, default=500, help="Number of samples (default 500)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation/data"),
        help="Output directory",
    )
    args = parser.parse_args()
    if args.n < 50:
        raise SystemExit("--n must be >= 50 for meaningful subgroup / CI checks")
    generate(n=args.n, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
