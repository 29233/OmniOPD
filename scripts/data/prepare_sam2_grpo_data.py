#!/usr/bin/env python3
"""Prepare prompt-only structured REF-AVS rows for SAM2 GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from swift_plugin.sam2_grpo_rewards import class_name_from_sample_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    messages = []
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            break
        messages.append(dict(message))
    if not messages or messages[-1].get("role") != "user":
        raise ValueError(f"row {index} does not end its prompt with a user message")
    user_prompt = str(messages[-1].get("content") or "")
    if not user_prompt.startswith("Task:\n"):
        raise ValueError(f"row {index} does not use the structured SFT prompt")
    if "<video><audio>" not in user_prompt:
        raise ValueError(f"row {index} lacks the expected video/audio placeholders")

    sample_id = str(row.get("sample_id") or "")
    class_name = str(row.get("class_name") or "").strip() or class_name_from_sample_id(sample_id)
    # REF-AVS appends negative/no-source rows with category "null" to val.
    # They have masks for dataset bookkeeping but no semantic class that can
    # satisfy the requested category reward, so exclude them explicitly.
    if class_name.strip().lower() == "null":
        return None
    if not class_name:
        raise ValueError(f"row {index} has no recoverable class name")
    frames = list(row.get("sam_frame_paths") or [])
    masks = list(row.get("mask_paths") or [])
    if len(frames) != 10 or len(masks) != 10:
        raise ValueError(f"row {index} must contain 10 SAM frames and masks")
    if not row.get("videos") or not row.get("audios"):
        raise ValueError(f"row {index} lacks rollout video or audio")

    return {
        "messages": messages,
        "videos": row["videos"],
        "audios": row["audios"],
        "sam_frame_paths": frames,
        "mask_paths": masks,
        "class_name": class_name,
        "sample_id": sample_id,
    }


def convert_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped_null_class = 0
    sample_ids = set()
    classes = set()
    with source.open(encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for index, line in enumerate(input_handle):
            if not line.strip():
                continue
            row = convert_row(json.loads(line), index)
            if row is None:
                skipped_null_class += 1
                continue
            if row["sample_id"] in sample_ids:
                raise ValueError(f"duplicate sample_id: {row['sample_id']}")
            sample_ids.add(row["sample_id"])
            classes.add(row["class_name"])
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(destination.resolve()),
        "output_sha256": sha256(destination),
        "rows": count,
        "skipped_null_class_rows": skipped_null_class,
        "unique_sample_ids": len(sample_ids),
        "classes": sorted(classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--val-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "purpose": "SAM2 SFT-baseline GRPO with class-exact and ten-frame mask-mIoU rewards",
        "assistant_labels_removed": True,
        "class_name_fallback": "sample_id.rsplit('_', 2)[-2]",
        "media": {
            "stored_frames": 10,
            "clip_seconds": 10,
            "dataset_nominal_fps": 1.0,
            "exact_sampling_instants_verified": False,
        },
        "splits": {
            "train": convert_file(args.train_source, args.output_dir / "train.jsonl"),
            "val": convert_file(args.val_source, args.output_dir / "val.jsonl"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
