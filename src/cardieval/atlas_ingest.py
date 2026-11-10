"""Load structured clinical-trial evidence embedded in CardiAtlas evidence records.

Only explicit context.clinical_trial_evidence objects are ingested. Free-text claims
are never parsed into endpoints or numerical outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clinical_evidence import ClinicalEndpointResult, ClinicalTrialEvidence


def load_atlas_clinical_trials(path: str | Path) -> list[ClinicalTrialEvidence]:
    """Load and validate CardiEval trial objects from a CardiAtlas JSONL export."""
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in CardiAtlas bundle at line {line_number}: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"invalid CardiAtlas record at line {line_number}: id is required")
        if record["id"] in records:
            raise ValueError(f"duplicate CardiAtlas record id: {record['id']}")
        records[record["id"]] = record

    products = {
        record_id: record for record_id, record in records.items()
        if record.get("record_type") == "therapeutic_product"
    }
    trials: list[ClinicalTrialEvidence] = []
    seen_trials: set[tuple[str, str]] = set()
    for record_id, record in records.items():
        if record.get("record_type") != "evidence":
            continue
        context = record.get("context") or {}
        payload = context.get("clinical_trial_evidence") if isinstance(context, dict) else None
        if payload is None:
            continue
        if not isinstance(payload, dict):
            raise ValueError(f"{record_id}: clinical_trial_evidence must be an object")
        product_id = payload.get("product_id")
        product = products.get(product_id)
        if product is None:
            raise ValueError(f"{record_id}: referenced therapeutic product {product_id!r} is missing")
        if record_id not in product.get("evidence_ids", []):
            raise ValueError(f"{record_id}: evidence is not linked from product {product_id}")
        key = (payload.get("trial_id", ""), product_id)
        if key in seen_trials:
            raise ValueError(f"duplicate clinical trial evidence source for {key[0]} / {product_id}")
        seen_trials.add(key)
        endpoints = []
        for endpoint in payload.get("endpoints", []):
            item = dict(endpoint)
            if item.get("confidence_interval") is not None:
                item["confidence_interval"] = tuple(item["confidence_interval"])
            endpoints.append(ClinicalEndpointResult(**item))
        trial = ClinicalTrialEvidence(
            trial_id=payload["trial_id"],
            product_id=product_id,
            phase=payload["phase"],
            design=payload["design"],
            comparator=payload.get("comparator"),
            randomised=payload["randomised"],
            blinded=payload.get("blinded"),
            endpoints=tuple(endpoints),
            source_ids=(record_id,),
            source_type=payload["source_type"],
            source_verified=payload.get("source_verified", False),
            follow_up=payload.get("follow_up"),
            limitations=tuple(payload.get("limitations", [])),
        )
        trials.append(trial)
    return trials


def build_clinical_evidence_report(trials: list[ClinicalTrialEvidence]) -> dict[str, Any]:
    """Create a transparent summary; never infer a primary endpoint from secondary data."""
    return {
        "schema_version": "1.0",
        "trial_count": len(trials),
        "trials": [
            {
                **trial.to_dict(),
                "primary_endpoint_status": trial.primary_endpoint_status,
                "subgroup_caveats": list(trial.subgroup_caveats),
            }
            for trial in trials
        ],
        "interpretation_policy": {
            "unknown_is_not_zero": True,
            "free_text_is_not_parsed_into_endpoint_values": True,
            "subgroup_results_do_not_override_primary_endpoint": True,
            "source_type_and_source_ids_are_retained": True,
        },
    }


def write_clinical_evidence_report(
    atlas_bundle_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    trials = load_atlas_clinical_trials(atlas_bundle_path)
    report = build_clinical_evidence_report(trials)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
