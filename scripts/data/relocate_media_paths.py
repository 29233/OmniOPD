#!/usr/bin/env python3
"""Relocate the audited REF-AVS JSONL media paths without changing its labels.

Does not copy media. Map only the known path-valued fields and fail if any
of those fields has an unexpected absolute root, so training cannot silently
mix media hosts.
The original file is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PATH_FIELDS = ("audios", "sam_frame_paths", "mask_paths", "teacher_audio_paths",
               "teacher_image_paths", "teacher_mask_paths")
REQUIRED_FIELDS = ("videos", "audios", "sam_frame_paths", "mask_paths")


def relocate_record(record: dict, old_root: Path, new_root: Path,
                    *, check_files: bool, validated: set[str]) -> dict:
    missing = set(REQUIRED_FIELDS) - record.keys()
    if missing:
        raise ValueError(f"{record.get('sample_id')}: missing media fields: {sorted(missing)}")
    if (len(record["videos"]) != 1 or len(record["videos"][0]) != 10
            or len(record["sam_frame_paths"]) != 10 or len(record["mask_paths"]) != 10
            or len(record["audios"]) != 1):
        raise ValueError(f"{record.get('sample_id')}: expected 10 frames/masks and one WAV")

    def mapped(value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(old_root):
            raise ValueError(f"Unexpected media root for {record.get('sample_id')}: {value}")
        result = str(new_root / path.relative_to(old_root))
        if check_files and result not in validated:
            if not Path(result).is_file():
                raise FileNotFoundError(result)
            validated.add(result)
        return result

    result = dict(record)
    result["videos"] = [[mapped(value) for value in item] for item in record["videos"]]
    for name in PATH_FIELDS:
        if name in result and result[name] is not None:
            if not isinstance(result[name], list):
                raise TypeError(f"{name} must be a list of media paths")
            result[name] = [mapped(value) for value in result[name]]
    if result["sam_frame_paths"] != result["videos"][0]:
        raise ValueError(f"{record.get('sample_id')}: SAM and Omni video frame lists disagree")
    return result


def relocate(source: Path, output: Path, old_root: Path, new_root: Path,
             *, check_files: bool = True) -> dict:
    source, output = source.resolve(), output.absolute()
    old_root, new_root = old_root.resolve(), new_root.absolute()
    if not source.is_file() or output.exists() or output == source:
        raise ValueError("Source must exist; output must be a new path")
    manifest_path = output.with_suffix(".media_manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    src_hash, dst_hash = hashlib.sha256(), hashlib.sha256()
    validated: set[str] = set()
    rows = 0
    try:
        with source.open("rb") as inp, temporary.open("wb") as dst:
            for line in inp:
                src_hash.update(line)
                if not line.strip():
                    continue
                record = relocate_record(json.loads(line), old_root, new_root,
                                         check_files=check_files, validated=validated)
                encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
                dst.write(encoded)
                dst_hash.update(encoded)
                rows += 1
        if rows == 0:
            raise ValueError("Empty REF-AVS input manifest")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    result = {
        "schema": 1, "purpose": "REF-AVS media path relocation only",
        "source": str(source), "source_sha256": src_hash.hexdigest(),
        "output": str(output), "output_sha256": dst_hash.hexdigest(),
        "rows": rows, "media_root_before": str(old_root), "media_root_after": str(new_root),
        "check_files": check_files, "unique_media_files_checked": len(validated),
        "label_and_message_contents_unchanged": True,
        "note": "Preserve split identity; never use test_s/test_u to build training rows.",
    }
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--skip-file-check", action="store_true",
                        help="For staging only; outputs must not be used for training until files are verified")
    args = parser.parse_args()
    print(json.dumps(relocate(args.source, args.output, args.old_root, args.new_root,
                              check_files=not args.skip_file_check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
