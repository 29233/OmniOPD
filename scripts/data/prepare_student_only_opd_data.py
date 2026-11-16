#!/usr/bin/env python3
"""Initial difficulty mixture using completed Student probes only.

No Teacher result is required or synthesized. Hard Student groups are NOT
called KD-rescue; only the online Teacher can establish a rescue license.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.data.prepare_verified_routed_opd_data import (
    allocate_counts, gt_class, probe_score, prompt_only, read_jsonl, sha256,
    weighted_bucket_sample,
)
from scripts.data.audit_verified_opd_probe_refresh import audit_condition

RATIOS = {"rl_active": .55, "hard": .30, "intermediate": .10, "mastered": .05}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--student-probes", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    source = read_jsonl(args.source)
    ids = {r["sample_id"] for r in source}
    if len(ids) != len(source):
        raise ValueError("Duplicate source IDs")
    probes = []
    audits = []
    for path in args.student_probes:
        rows = read_jsonl(path)
        mapping = {r["sample_id"]: r for r in rows}
        if len(mapping) != len(rows) or set(mapping) != ids:
            raise ValueError(f"Incomplete or duplicate Student coverage: {path}")
        report = audit_condition(path, ids, 1.0, .9, True, True)
        if not report["valid"]:
            raise ValueError(f"Invalid Student probe: {report}")
        audits.append(report)
        probes.append(mapping)
    annotated = []
    for raw in source:
        sid = raw["sample_id"]
        scores = [probe_score(p[sid], gt_class(raw)) for p in probes]
        rewards = [s["reward"] for s in scores]
        mean, std = float(np.mean(rewards)), float(np.std(rewards))
        bucket = ("rl_active" if std > 1e-4 else "mastered" if mean >= .85
                  else "hard" if mean <= .10 else "intermediate")
        row = prompt_only(raw)
        for key in list(row):
            if key.startswith(("audio_teacher_", "visual_teacher_")):
                del row[key]
        row.update(class_name=gt_class(raw), difficulty_bucket=bucket,
                   student_probe_count=len(probes), student_reward_mean=mean,
                   student_reward_std=std,
                   student_class_success_rate=float(np.mean([s["class_correct"] for s in scores])),
                   student_iou_mean=float(np.mean([s["iou"] for s in scores])))
        annotated.append(row)
    # Keep source modality proportions. Balance difficulty/class *within*
    # modality, so rare Visual/Semantic rows cannot disappear by chance.
    by_modality = defaultdict(list)
    for row in annotated:
        by_modality[row["template_match"]["template_reference"]].append(row)
    rng = random.Random(20261103)
    sampled = []
    for modality, rows in by_modality.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["difficulty_bucket"]].append(row)
        quotas = allocate_counts(len(rows), RATIOS)
        for name in list(quotas):
            if quotas[name] and not buckets[name]:
                count, quotas[name] = quotas[name], 0
                dest = next(k for k in RATIOS if buckets[k])
                quotas[dest] += count
        for name, count in quotas.items():
            sampled.extend(weighted_bucket_sample(buckets[name], count, rng, .5))
    rng.shuffle(sampled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("annotated", annotated), ("train", sampled)]:
        with (args.output_dir / f"{name}.jsonl").open("w") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": 1, "purpose": "Student-only initial mixture; online current-policy Teachers",
        "source": str(args.source.resolve()), "source_sha256": sha256(args.source),
        "student_probes": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.student_probes],
        "rows": len(annotated), "sampled_rows": len(sampled),
        "sampled_unique_ids": len({r["sample_id"] for r in sampled}),
        "source_bucket_counts": dict(Counter(r["difficulty_bucket"] for r in annotated)),
        "target_ratios": RATIOS,
        "realized_ratios": {k: sum(r["difficulty_bucket"] == k for r in sampled) / len(sampled) for k in RATIOS},
        "modality_counts": dict(Counter(r["template_match"]["template_reference"] for r in sampled)),
        "offline_teacher_required": False,
        "sampling": {"with_replacement": True, "seed": 20261103, "within_modality": True},
        "train_sha256": sha256(args.output_dir / "train.jsonl"),
        "difficulty_note": "Initial SFT difficulty only; not a current-policy license. No cached KD-rescue labels.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(json.dumps({"valid": True, "student_conditions": audits}, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
