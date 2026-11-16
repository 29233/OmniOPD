#!/usr/bin/env python3
"""Degrade REF-AVS audio tracks with additive Gaussian noise (offline, deterministic).

Generates the STUDENT-side audio for audio-degrade OPSD: the student hears a
noisy copy, the teacher hears the original (see swift_plugin/aurora_opsd.py
``_patch_teacher_audio`` and the ``audio-degrade`` branch of
scripts/prepare_refavs_opsd.py, which consumes these files via
``--degraded-audio-name``).

Output: ``media/<vid>/audio_snr{N}db.wav`` next to the original ``audio.wav``
(same directory, same duration, same sample rate — only noise is added, so the
degraded and original waveforms yield IDENTICAL audio token counts on the
Qwen2.5-Omni side, keeping teacher/student sequences aligned). Multiple SNR
levels coexist via the filename suffix; run once per level to build a sweep.

Determinism: per-file RNG seeded from sha256(f"{seed}:{vid}:{snr_db}"), so
``--force`` regeneration is bit-identical.

Usage:
  python scripts/degrade_refavs_audio.py --snr-db 10            # one level
  for snr in 20 10 5 0; do                                      # full sweep
    python scripts/degrade_refavs_audio.py --snr-db "$snr"
  done
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def degraded_name(snr_db: float) -> str:
    """Canonical degraded filename: audio_snr10db.wav (integer SNR, no decimals)."""
    return f"audio_snr{int(round(snr_db))}db.wav"


def collect_audio_paths(jsonl_paths: list[str]) -> list[Path]:
    """Distinct audio.wav paths referenced by the given dataset rows."""
    seen: set[str] = set()
    ordered: list[Path] = []
    for jsonl in jsonl_paths:
        path = Path(jsonl)
        if not path.is_file():
            print(f"[warn] {jsonl} not found, skipping", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                audios = row.get("audios") or []
                if not audios:
                    continue
                audio = str(audios[0])
                if audio not in seen:
                    seen.add(audio)
                    ordered.append(Path(audio))
    return ordered


def degrade_one(src: Path, dst: Path, snr_db: float, seed: int) -> dict:
    """Add deterministic Gaussian noise to one file at the target SNR."""
    # Read WITHOUT resampling: swift's load_audio resamples to 16k on load,
    # so the degraded file must keep the original sampling rate.
    x, sr = sf.read(str(src), always_2d=False)
    if x.ndim > 1:  # stereo -> mono (Ref-AVS tracks are already mono; be safe)
        x = x.mean(axis=1)

    # Per-file deterministic seed so --force regeneration is bit-identical.
    vid = src.parent.name
    digest = hashlib.sha256(f"{seed}:{vid}:{snr_db}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

    p_sig = float(np.mean(x.astype(np.float64) ** 2))
    near_silent = p_sig < 1e-8
    p_ref = max(p_sig, 1e-8)
    if near_silent:
        print(f"[warn] near-silent track, noise floored: {src}", file=sys.stderr)
    noise_scale = math.sqrt(p_ref / (10.0 ** (snr_db / 10.0)))
    noise = rng.standard_normal(x.shape[0]) * noise_scale

    y = np.clip(x + noise.astype(x.dtype), -1.0, 1.0)
    clipped = float(np.mean(np.abs(x + noise.astype(x.dtype)) > 1.0))
    if clipped > 0.01:
        print(f"[warn] {clipped:.1%} samples clipped at {snr_db} dB: {src}", file=sys.stderr)

    sf.write(str(dst), y, sr, subtype="PCM_16")

    # Verify the achieved SNR of what we actually wrote (post-clip, post-quant).
    y_read, _ = sf.read(str(dst), always_2d=False)
    if y_read.ndim > 1:
        y_read = y_read.mean(axis=1)
    p_noise_actual = float(np.mean((y_read.astype(np.float64) - x.astype(np.float64)) ** 2))
    achieved = 10.0 * math.log10(p_ref / max(p_noise_actual, 1e-12))
    if y_read.shape[0] != x.shape[0]:
        raise RuntimeError(f"length changed: {src} {x.shape[0]} -> {dst} {y_read.shape[0]}")

    return {
        "src": str(src),
        "dst": str(dst),
        "samples": int(x.shape[0]),
        "sr": int(sr),
        "near_silent": near_silent,
        "clipped_frac": clipped,
        "snr_target_db": snr_db,
        "snr_achieved_db": achieved,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl", action="append", dest="jsonl_paths",
        default=[str(root / "outputs" / "data" / "refavs" / "refavs_full.jsonl"),
                 str(root / "outputs" / "data" / "refavs" / "refavs_val.jsonl")],
        help="dataset rows whose audios[0] should be degraded (repeatable)")
    parser.add_argument("--snr-db", type=float, default=10.0, help="target signal-to-noise ratio in dB")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="regenerate even if the output exists")
    parser.add_argument("--max-files", type=int, default=-1, help="cap for smoke tests")
    args = parser.parse_args()

    sources = collect_audio_paths(args.jsonl_paths)
    if args.max_files > 0:
        sources = sources[: args.max_files]
    if not sources:
        print("[error] no audio paths collected; check --jsonl", file=sys.stderr)
        sys.exit(1)

    stats = {"files_total": len(sources), "written": 0, "skipped_existing": 0,
             "near_silent": 0, "missing_src": 0, "snr_deviation_max": 0.0}
    for src in sources:
        if not src.is_file():
            print(f"[error] source audio missing: {src}", file=sys.stderr)
            stats["missing_src"] += 1
            continue
        dst = src.parent / degraded_name(args.snr_db)
        if dst.is_file() and not args.force:
            stats["skipped_existing"] += 1
            continue
        info = degrade_one(src, dst, args.snr_db, args.seed)
        stats["written"] += 1
        stats["near_silent"] += int(info["near_silent"])
        stats["snr_deviation_max"] = max(
            stats["snr_deviation_max"], abs(info["snr_achieved_db"] - args.snr_db))

    print(json.dumps(stats, indent=2))
    if stats["missing_src"]:
        sys.exit(1)
    if stats["snr_deviation_max"] > 0.3:
        print(f"[warn] max SNR deviation {stats['snr_deviation_max']:.2f} dB exceeds 0.3 dB",
              file=sys.stderr)


if __name__ == "__main__":
    main()
