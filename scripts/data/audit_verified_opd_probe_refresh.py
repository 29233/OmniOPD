#!/usr/bin/env python3
"""Audit completeness and media outputs of a verified-OPD probe refresh."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def audit_condition(
    path: Path,
    expected_ids: set[str],
    expected_temperature: float,
    expected_top_p: float,
    check_masks: bool,
    allow_invalid_seg: bool,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    ids = [str(row.get("sample_id") or "") for row in rows]
    errors: Counter[str] = Counter()
    missing_masks = 0
    missing_seg_features = 0
    ious = []
    dices = []
    for row in rows:
        if row.get("status") != "ok":
            errors["status"] += 1
        seg_count = int(row.get("seg_count") or 0)
        invalid_prediction = allow_invalid_seg and seg_count != 1
        if seg_count != 1 and not invalid_prediction:
            errors["seg_count"] += 1
        if int(row.get("frames_scored") or 0) != 10 and not invalid_prediction:
            errors["frames_scored"] += 1
        if len(row.get("frame_ious") or []) != 10 and not invalid_prediction:
            errors["frame_ious"] += 1
        if len(row.get("frame_dices") or []) != 10 and not invalid_prediction:
            errors["frame_dices"] += 1
        mask_paths = [Path(value) for value in row.get("mask_paths") or []]
        if len(mask_paths) != 10 and not invalid_prediction:
            errors["mask_paths"] += 1
        elif check_masks and not invalid_prediction:
            missing_masks += sum(not mask.is_file() for mask in mask_paths)
        seg_feature = Path(str(row.get("seg_feature_file") or ""))
        if not seg_feature.is_file() and not invalid_prediction:
            missing_seg_features += 1
        if float(row.get("generation_temperature")) != expected_temperature:
            errors["temperature"] += 1
        if float(row.get("generation_top_p")) != expected_top_p:
            errors["top_p"] += 1
        if (row.get("iou") is None or row.get("dice") is None) and not invalid_prediction:
            errors["metric"] += 1
        elif not invalid_prediction:
            ious.append(float(row["iou"]))
            dices.append(float(row["dice"]))
    id_set = set(ids)
    result = {
        "path": str(path.resolve()),
        "rows": len(rows),
        "unique_sample_ids": len(id_set),
        "duplicate_sample_ids": len(ids) - len(id_set),
        "missing_expected_ids": len(expected_ids - id_set),
        "unexpected_ids": len(id_set - expected_ids),
        "errors": dict(errors),
        "missing_mask_files": missing_masks,
        "missing_seg_feature_references": missing_seg_features,
        "allowed_invalid_seg_trajectories": sum(
            int(row.get("seg_count") or 0) != 1 for row in rows
        ) if allow_invalid_seg else 0,
        "mean_iou": sum(ious) / len(ious) if ious else None,
        "mean_dice": sum(dices) / len(dices) if dices else None,
    }
    result["valid"] = (
        len(rows) == len(expected_ids)
        and result["unique_sample_ids"] == len(expected_ids)
        and not result["missing_expected_ids"]
        and not result["unexpected_ids"]
        and not errors
        and not missing_masks
        and not missing_seg_features
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--student-rollouts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-mask-file-check", action="store_true")
    args = parser.parse_args()

    selected_path = args.probe_root / "selected" / "refavs_train.jsonl"
    if not selected_path.is_file():
        selected_path = args.probe_root / "conditions" / "student" / "refavs_train.jsonl"
    selected = read_jsonl(selected_path)
    selected_ids = {str(row.get("sample_id") or "") for row in selected}
    if len(selected_ids) != len(selected):
        raise SystemExit("Selected probe source contains duplicate sample IDs")

    conditions = {}
    for index in range(args.student_rollouts):
        conditions[f"student_{index}"] = audit_condition(
            args.probe_root / f"student_{index}" / "train.jsonl",
            selected_ids,
            expected_temperature=1.0,
            expected_top_p=0.9,
            check_masks=not args.skip_mask_file_check,
            allow_invalid_seg=True,
        )
    conditions["audio_teacher"] = audit_condition(
        args.probe_root / "audio_teacher" / "train.jsonl",
        selected_ids,
        expected_temperature=0.0,
        expected_top_p=1.0,
        check_masks=not args.skip_mask_file_check,
        # Missing/multiple SEG is a genuine model failure, not an incomplete
        # infrastructure run. Keep it as a verifier-negative trajectory
        # (mask IoU=0 in prepare_verified_routed_opd_data), just like Student
        # failures, instead of aborting a full training-set refresh.
        allow_invalid_seg=True,
    )
    conditions["visual_teacher"] = audit_condition(
        args.probe_root / "visual_teacher" / "train.jsonl",
        selected_ids,
        expected_temperature=0.0,
        expected_top_p=1.0,
        check_masks=not args.skip_mask_file_check,
        allow_invalid_seg=True,
    )

    data_manifest_path = args.data_root / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    annotated = read_jsonl(args.data_root / "annotated.jsonl")
    train = read_jsonl(args.data_root / "train.jsonl")
    train_ids = Counter(str(row.get("sample_id") or "") for row in train)
    data_result = {
        "manifest": str(data_manifest_path.resolve()),
        "annotated_rows": len(annotated),
        "annotated_unique_sample_ids": len({
            str(row.get("sample_id") or "") for row in annotated
        }),
        "train_rows": len(train),
        "train_unique_sample_ids": len(train_ids),
        "train_duplicate_rows": len(train) - len(train_ids),
        "train_max_repeat": max(train_ids.values(), default=0),
        "bucket_counts": dict(Counter(str(row.get("difficulty_bucket")) for row in train)),
        "source_bucket_counts": data_manifest.get("source_bucket_counts"),
    }
    data_result["valid"] = (
        len(annotated) == len(selected)
        and data_result["annotated_unique_sample_ids"] == len(selected)
        and len(train) == int(data_manifest.get("sampled_rows", -1))
    )
    report = {
        "schema": 1,
        "purpose": "verified-OPD expanded probe completeness audit",
        "probe_root": str(args.probe_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "selected_rows": len(selected),
        "selected_unique_sample_ids": len(selected_ids),
        "mask_file_check": not args.skip_mask_file_check,
        "conditions": conditions,
        "data": data_result,
    }
    report["valid"] = data_result["valid"] and all(
        result["valid"] for result in conditions.values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit("Expanded verified-OPD probe audit failed")


if __name__ == "__main__":
    main()
