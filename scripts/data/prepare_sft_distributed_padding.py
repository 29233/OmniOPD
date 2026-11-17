#!/usr/bin/env python3
"""Preserve full sample coverage with ms-swift's floor-sharded batch sampler."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prepare(source: Path, output: Path, world_size: int) -> dict:
    rows = [json.loads(line) for line in source.open() if line.strip()]
    ids = [str(row["sample_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("The unpadded SFT source must contain unique sample IDs")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    needed = (-len(rows)) % world_size
    # Standard distributed padding, choosing a different modality whenever
    # available. No original sample is deleted or rewritten.
    candidates = []
    seen_modalities = set()
    for index, row in enumerate(rows):
        modality = (row.get("template_match") or {}).get("template_reference")
        if modality not in seen_modalities:
            seen_modalities.add(modality)
            candidates.append(index)
    initial_candidates = set(candidates)
    candidates.extend(index for index in range(len(rows)) if index not in initial_candidates)
    padding = [rows[candidates[index % len(candidates)]] for index in range(needed)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows + padding:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_unique_samples": len(rows),
        "physical_rows": len(rows) + needed,
        "world_size": world_size,
        "padding_rows": needed,
        "padding_sample_ids": [row["sample_id"] for row in padding],
        "purpose": "avoid ms-swift BatchSamplerShard's total_samples // world_size dropping tail samples",
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.world_size), indent=2))


if __name__ == "__main__":
    main()
