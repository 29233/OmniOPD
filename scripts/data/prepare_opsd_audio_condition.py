#!/usr/bin/env python3
"""Build deterministic audio-degraded files for current-Student OPSD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]


def seed_for(source: Path, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{source.resolve()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def read_audio(path: Path):
    data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    return data, int(rate)


def write_audio(path: Path, data: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1.0, 1.0), rate, subtype="PCM_16")
    check, check_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if check_rate != rate or check.shape != data.shape:
        raise RuntimeError(f"audio shape/rate changed: {path} {data.shape}/{rate} -> {check.shape}/{check_rate}")


def save_metadata(path: Path, config: dict, records: dict) -> None:
    """Checkpoint metadata atomically, including during audio preparation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump({**config, 'files': list(records.values())}, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_orphan(destination: Path, expected: np.ndarray, rate: int) -> None:
    """Verify deterministic PCM16 WAV bytes without overwriting the cache."""
    fd, temporary = tempfile.mkstemp(prefix='.audio-cache-verify-', suffix='.wav')
    os.close(fd)
    try:
        write_audio(Path(temporary), expected, rate)
        if hashlib.sha256(Path(temporary).read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise RuntimeError(
                f'Orphan cache does not match requested audio settings: {destination}; '
                'choose a new output name or use --force explicitly.'
            )
    finally:
        os.unlink(temporary)


def mute_intervals(data: np.ndarray, rate: int, fraction: float, block_ms: int, rng: np.random.Generator):
    count = data.shape[0]
    target = int(round(count * fraction))
    block = max(1, int(round(rate * block_ms / 1000.0)))
    starts = np.arange(0, count, block, dtype=np.int64)
    order = rng.permutation(len(starts))
    muted = np.zeros(count, dtype=bool)
    remaining = target
    intervals = []
    for index in order:
        if remaining <= 0:
            break
        start = int(starts[index])
        end = min(count, start + min(block, remaining))
        muted[start:end] = True
        intervals.append([start, end])
        remaining -= end - start
    output = data.copy()
    output[muted] = 0.0
    return output, intervals, float(muted.mean())


def noise(data: np.ndarray, snr_db: float, rng: np.random.Generator):
    signal_rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if signal_rms <= 1e-12:
        return data.copy(), None
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    eps = rng.normal(0.0, noise_rms, size=data.shape).astype(np.float32)
    raw = data + eps
    clipped = float(np.mean(np.abs(raw) > 1.0))
    return np.clip(raw, -1.0, 1.0), clipped


def collect(manifests: list[Path]) -> list[Path]:
    seen = set()
    paths = []
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                for raw in json.loads(line).get("audios", [])[:1]:
                    path = Path(raw).resolve()
                    if str(path) not in seen:
                        seen.add(str(path)); paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["mute", "mute50", "noise5db"], required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--block-ms", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument('--recover-orphan-cache', action='store_true',
                        help='Rebuild missing metadata only after deterministic WAV hash verification')
    parser.add_argument("--preserve-cache-records", action="store_true",
                        help="Keep verified metadata entries outside the selected manifest")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.wav'):
        parser.error('output-name must be a WAV filename, without a directory')
    if args.block_ms <= 0 or not np.isfinite(args.snr_db):
        parser.error('block-ms must be positive and snr-db must be finite')
    if not args.manifest:
        args.manifest = [ROOT / "outputs/data/refavs/refavs_full.jsonl", ROOT / "outputs/data/refavs/refavs_val.jsonl"]
    if not 0 < args.fraction <= 1:
        parser.error("fraction must be in (0, 1]")
    paths = collect(args.manifest)
    if args.max_files > 0:
        paths = paths[:args.max_files]
    config = {"condition": args.condition, "seed": args.seed, "output_name": args.output_name}
    if args.condition in {"mute", "mute50"}:
        config.update(fraction=args.fraction, block_ms=args.block_ms)
    else:
        config.update(snr_db=args.snr_db)
    previous_records = {}
    if args.metadata.exists() and not args.force:
        previous = json.loads(args.metadata.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise RuntimeError(
                f"Existing metadata {args.metadata} was made with a different "
                f"configuration; choose new output/metadata paths or use --force explicitly."
            )
        previous_records = {item['source']: item for item in previous.get('files', [])}
    records = []
    cached_records = previous_records.copy() if args.preserve_cache_records else {}
    for index, source in enumerate(paths):
        destination = source.with_name(args.output_name)
        if destination.resolve() == source.resolve():
            raise ValueError(f'Refusing to overwrite original audio: {source}')
        orphan = False
        if destination.exists() and not args.force:
            existing = previous_records.get(str(source))
            if not existing or existing.get('destination') != str(destination):
                if not args.recover_orphan_cache:
                    raise RuntimeError(f'No matching metadata for {destination}; use --recover-orphan-cache, a new output name or --force.')
                orphan = True
            else:
                for key, path in [('source_sha256', source), ('destination_sha256', destination)]:
                    if existing.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise RuntimeError(f'Cached audio changed: {path}; use --force to regenerate explicitly.')
                records.append(existing)
                cached_records[str(source)] = existing
                continue
        data, rate = read_audio(source)
        if not data.size:
            raise ValueError(f'Empty audio: {source}')
        rng = np.random.default_rng(seed_for(source, args.seed))
        item = {"source": str(source), "destination": str(destination), "sample_rate": rate, "samples": int(data.shape[0])}
        if args.condition in {"mute", "mute50"}:
            degraded, intervals, fraction = mute_intervals(data, rate, args.fraction, args.block_ms, rng)
            item.update(muted_fraction=fraction, block_ms=args.block_ms, intervals=intervals)
        else:
            degraded, clipped = noise(data, args.snr_db, rng)
            item.update(snr_db=args.snr_db, clipped_fraction_before_clip=clipped)
        if orphan:
            verify_orphan(destination, degraded, rate)
            item['recovered_metadata'] = True
        else:
            write_audio(destination, degraded, rate)
        item["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        item["destination_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        records.append(item)
        cached_records[str(source)] = item
        save_metadata(args.metadata, config, cached_records)
        if (index + 1) % 100 == 0:
            print(f'[{args.condition}] {index + 1}/{len(paths)} files', flush=True)
    save_metadata(args.metadata, config, cached_records)
    print(f"Prepared {len(records)} audio files for {args.condition}: {args.metadata}", flush=True)


if __name__ == "__main__":
    main()
