#!/usr/bin/env python3
"""Build prompt-only verified-OPD rows and an empirical difficulty mixture.

The input probe files are ordinary AURORA evaluation JSONL files.  Student
probes may contain multiple rows per sample (different rollout seeds); each
privileged teacher file must contain one cached probe per sample.

Difficulty is based on *training signal*, not nominal template grade:

* ``rl_active``: student rollout rewards have non-zero variance;
* ``kd_rescue``: student rollouts are uniformly poor but at least one teacher
  predicts the GT category;
* ``mastered``: student reward is consistently high;
* ``hard_unverified``: uniformly poor and neither teacher predicts the GT;
* ``intermediate``: everything else.

The sampled mixture preserves all rows in ``annotated.jsonl`` and materializes
an explicitly weighted ``train.jsonl`` for ms-swift, which has no per-row
weighted sampler in this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from swift_plugin.sam2_grpo_rewards import (
    canonical_class_name,
    class_name_from_sample_id,
    extract_target_class,
)


DEFAULT_RATIOS = {
    "rl_active": 0.45,
    "kd_rescue": 0.30,
    "intermediate": 0.15,
    "mastered": 0.075,
    "hard_unverified": 0.025,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gt_class(row: dict[str, Any]) -> str:
    value = canonical_class_name(row.get("class_name"))
    if value:
        return value
    return canonical_class_name(class_name_from_sample_id(str(row.get("sample_id") or "")))


def probe_score(row: dict[str, Any], target: str) -> dict[str, Any]:
    rollout = str(row.get("rollout") or row.get("completion") or "")
    prediction = extract_target_class(rollout)
    valid_seg = (
        row.get("status", "ok") == "ok"
        and int(row.get("seg_count", 1) or 0) == 1
        and row.get("iou") is not None
    )
    iou = float(row.get("iou") or 0.0) if valid_seg else 0.0
    iou = min(max(iou, 0.0), 1.0)
    correct = bool(target and prediction == target)
    # Category correctness gates mask credit.  This prevents an inconsistent
    # textual target from receiving a high segmentation reward accidentally.
    reward = 0.5 * float(correct) + 0.5 * float(correct) * iou
    return {
        "completion": rollout,
        "predicted_class": prediction,
        "class_correct": correct,
        "iou": iou,
        "seg_valid": valid_seg,
        "reward": reward,
    }


def group_student_probes(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"Student probe row in {path} has no sample_id")
        grouped[sample_id].append(row)
    return grouped


def unique_probes(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in result:
            raise ValueError(f"Teacher probe file must contain one unique row per sample: {sample_id!r}")
        result[sample_id] = row
    return result


def prompt_only(row: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            break
        messages.append(dict(message))
    if not messages or messages[-1].get("role") != "user":
        raise ValueError(f"{row.get('sample_id')} has no prompt-only user turn")
    result = {key: value for key, value in row.items() if key != "messages"}
    result["messages"] = messages
    return result


def parse_ratios(value: str) -> dict[str, float]:
    ratios = {}
    for part in value.split(","):
        name, weight = part.split("=", 1)
        ratios[name.strip()] = float(weight)
    if set(ratios) != set(DEFAULT_RATIOS):
        raise ValueError(f"Difficulty ratios must cover {sorted(DEFAULT_RATIOS)}")
    if any(weight < 0 for weight in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError("Difficulty ratios must be nonnegative with positive total")
    total = sum(ratios.values())
    return {key: value / total for key, value in ratios.items()}


def difficulty_bucket(
    rewards: list[float],
    teacher_scores: list[dict[str, Any]],
    std_threshold: float,
    hard_threshold: float,
    easy_threshold: float,
) -> str:
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    if std > std_threshold:
        return "rl_active"
    if mean >= easy_threshold:
        return "mastered"
    if mean <= hard_threshold:
        if any(score["class_correct"] for score in teacher_scores):
            return "kd_rescue"
        return "hard_unverified"
    return "intermediate"


def eligible_difficulty_teachers(
    row: dict[str, Any], scores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Use the same Teacher eligibility as the online modality router."""
    modality = (row.get("template_match") or {}).get("template_reference")
    # An audio-Teacher success cannot rescue a non-audio sample in training,
    # because that Teacher is deliberately disabled by the online router.
    return scores[1:] if modality in {"Visual", "Semantic"} else scores


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in ratios.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def weighted_bucket_sample(
    rows: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    class_balance_power: float,
) -> list[dict[str, Any]]:
    if not rows or count <= 0:
        return []
    frequencies = Counter(gt_class(row) for row in rows)
    weights = [
        1.0 / max(frequencies[gt_class(row)], 1) ** class_balance_power
        for row in rows
    ]
    return rng.choices(rows, weights=weights, k=count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--student-probes", type=Path, required=True)
    parser.add_argument("--audio-teacher-probes", type=Path, required=True)
    parser.add_argument("--visual-teacher-probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--std-threshold", type=float, default=1e-4)
    parser.add_argument("--hard-threshold", type=float, default=0.10)
    parser.add_argument("--easy-threshold", type=float, default=0.85)
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument(
        "--difficulty-ratios",
        default=",".join(f"{key}={value}" for key, value in DEFAULT_RATIOS.items()),
    )
    args = parser.parse_args()
    ratios = parse_ratios(args.difficulty_ratios)
    source = read_jsonl(args.source)
    student = group_student_probes(args.student_probes)
    audio = unique_probes(args.audio_teacher_probes)
    visual = unique_probes(args.visual_teacher_probes)
    output_rows = []
    bucket_counts: Counter[str] = Counter()
    missing = []

    for raw in source:
        sample_id = str(raw.get("sample_id") or "")
        if sample_id not in student or sample_id not in audio or sample_id not in visual:
            missing.append(sample_id)
            continue
        target = gt_class(raw)
        if not target or target == "null":
            continue
        student_scores = [probe_score(row, target) for row in student[sample_id]]
        teacher_scores = [
            probe_score(audio[sample_id], target),
            probe_score(visual[sample_id], target),
        ]
        rewards = [score["reward"] for score in student_scores]
        bucket = difficulty_bucket(
            rewards,
            eligible_difficulty_teachers(raw, teacher_scores),
            args.std_threshold,
            args.hard_threshold,
            args.easy_threshold,
        )
        row = prompt_only(raw)
        row.update({
            "class_name": target,
            "difficulty_bucket": bucket,
            "student_probe_count": len(student_scores),
            "student_reward_mean": float(np.mean(rewards)),
            "student_reward_std": float(np.std(rewards)),
            "student_class_success_rate": float(np.mean([
                score["class_correct"] for score in student_scores
            ])),
            "student_iou_mean": float(np.mean([
                score["iou"] for score in student_scores
            ])),
            "audio_teacher_completion": teacher_scores[0]["completion"],
            "audio_teacher_predicted_class": teacher_scores[0]["predicted_class"],
            "audio_teacher_class_correct": teacher_scores[0]["class_correct"],
            "audio_teacher_iou": teacher_scores[0]["iou"],
            "audio_teacher_seg_valid": teacher_scores[0]["seg_valid"],
            "visual_teacher_completion": teacher_scores[1]["completion"],
            "visual_teacher_predicted_class": teacher_scores[1]["predicted_class"],
            "visual_teacher_class_correct": teacher_scores[1]["class_correct"],
            "visual_teacher_iou": teacher_scores[1]["iou"],
            "visual_teacher_seg_valid": teacher_scores[1]["seg_valid"],
        })
        output_rows.append(row)
        bucket_counts[bucket] += 1

    if missing:
        raise SystemExit(
            f"Probe coverage is incomplete for {len(missing)} source rows; first={missing[:8]}"
        )
    if not output_rows:
        raise SystemExit("No verified OPD rows were produced")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = args.output_dir / "annotated.jsonl"
    with annotated_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(output_rows) if args.output_samples < 0 else args.output_samples
    desired = allocate_counts(total, ratios)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        buckets[row["difficulty_bucket"]].append(row)
    # Reallocate an empty bucket's quota to RL-active, then intermediate, then
    # any nonempty bucket.  The manifest records the realized mixture.
    for name, count in list(desired.items()):
        if count and not buckets[name]:
            desired[name] = 0
            destination = next(
                key for key in ("rl_active", "intermediate", "kd_rescue", "mastered", "hard_unverified")
                if buckets[key]
            )
            desired[destination] += count
    rng = random.Random(args.seed)
    sampled = []
    for name in DEFAULT_RATIOS:
        sampled.extend(
            weighted_bucket_sample(
                buckets[name], desired[name], rng, args.class_balance_power
            )
        )
    rng.shuffle(sampled)
    train_path = args.output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    realized = Counter(row["difficulty_bucket"] for row in sampled)
    manifest = {
        "schema": 1,
        "purpose": "verified routed OPD + verifiable reward empirical difficulty mixture",
        "sources": {
            "data": str(args.source.resolve()),
            "student_probes": str(args.student_probes.resolve()),
            "audio_teacher_probes": str(args.audio_teacher_probes.resolve()),
            "visual_teacher_probes": str(args.visual_teacher_probes.resolve()),
        },
        "source_sha256": {
            "data": sha256(args.source),
            "student_probes": sha256(args.student_probes),
            "audio_teacher_probes": sha256(args.audio_teacher_probes),
            "visual_teacher_probes": sha256(args.visual_teacher_probes),
        },
        "rows": len(output_rows),
        "sampled_rows": len(sampled),
        "source_bucket_counts": dict(bucket_counts),
        "target_ratios": ratios,
        "realized_bucket_counts": dict(realized),
        "realized_ratios": {
            key: realized[key] / max(len(sampled), 1) for key in DEFAULT_RATIOS
        },
        "difficulty": {
            "reward": "0.5*class_exact + 0.5*class_exact*mean_10_frame_iou",
            "std_threshold": args.std_threshold,
            "hard_threshold": args.hard_threshold,
            "easy_threshold": args.easy_threshold,
            "student_probe_requirement": "multiple stochastic rollouts recommended",
            "kd_rescue": "zero/low-variance hard student group with at least one GT-correct teacher probe",
            "modality_eligibility": "Visual/Semantic difficulty ignores audio Teacher, matching online routing",
        },
        "sampling": {
            "seed": args.seed,
            "class_balance_power": args.class_balance_power,
            "with_replacement": True,
        },
        "outputs": {
            "annotated": str(annotated_path.resolve()),
            "annotated_sha256": sha256(annotated_path),
            "train": str(train_path.resolve()),
            "train_sha256": sha256(train_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
