#!/usr/bin/env python3
"""Two-stage REF-AVS evaluation for Qwen2.5-Omni and Qwen3-Omni.

Stage ``generate`` runs the LLM and stores the complete response plus the
post-[SEG] projected prompt embedding.  Stage ``segment`` loads only SAM and
the checkpoint mask decoder, then scores those saved embeddings.  The latter
stage does not load the language model or replay the assistant response.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_name(value: str, index: int) -> str:
    text = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in str(value))
    return text.strip(".") or f"case-{index:06d}"


def _unwrap(model):
    current = model
    for _ in range(8):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            base = current.get_base_model()
            if base is current:
                break
            current = base
            continue
        break
    return current


def _load_rows(data_dir: Path, split: str, max_samples: int, shard: int, shards: int):
    path = data_dir / f"refavs_{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if max_samples > 0:
        rows = rows[:max_samples]
    return [(i, row) for i, row in enumerate(rows) if i % shards == shard]


def _build_engine(args):
    os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "0")
    os.environ.setdefault("USE_AUDIO_IN_VIDEO", "0")
    os.environ.setdefault("FPS", str(args.video_fps))
    os.environ.setdefault("FPS_MAX_FRAMES", "10" if args.family == "qwen3" else "768")
    os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "128" if args.family == "qwen3" else "768")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024" if args.family == "qwen3" else "16384")
    os.environ.setdefault("AURORA_VIDEO_FPS", str(args.video_fps))
    os.environ["AURORA_NUM_SAM_FRAMES"] = str(args.num_frames)
    os.environ.setdefault(
        "AURORA_SAM_CHECKPOINT",
        "models/sam_vit_h_4b8939.pth",
    )
    if args.family == "qwen3":
        from scripts.eval.test_qwen3_omni_refavs import _prepare_adapter_for_eval, _restore_saved_modules
        import swift_plugin.aurora_qwen3_omni  # noqa: F401
        from swift.infer_engine import TransformersEngine

        adapter, temporary = _prepare_adapter_for_eval(args.checkpoint)
        max_memory = {i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())}
        engine = TransformersEngine(
            str(args.model), adapters=[str(adapter)], torch_dtype=torch.bfloat16,
            model_type="aurora_qwen3_omni", template_type="aurora_qwen3_omni",
            new_special_tokens=["[SEG]"], attn_impl=args.attn_impl,
            device_map="auto", max_memory=max_memory, download_model=False,
            max_batch_size=args.batch_size,
        )
        if temporary is not None:
            engine._aurora_adapter_tmp = temporary
            _restore_saved_modules(engine.model, args.checkpoint)
        engine.model.eval()
        return engine

    if args.family == "qwen2_5":
        import swift_plugin.aurora_qwen2_5_omni  # noqa: F401
        from scripts.eval.test_qwen2_5_omni_refavs import _build_engine

        parsed = SimpleNamespace(
            model=str(args.model), checkpoint=Path(args.checkpoint),
            device=f"cuda:{os.environ.get('LOCAL_RANK', '0')}", attn_impl=args.attn_impl,
        )
        engine, _, _ = _build_engine(parsed)
        engine.max_batch_size = args.batch_size
        engine.model.eval()
        return engine
    raise ValueError(f"unsupported family: {args.family}")


@torch.inference_mode()
def _generate_batch(engine, requests, request_config, seg_id: int):
    """Generate a batch and capture one post-[SEG] hidden state per item."""
    captured: dict[str, Any] = {"post": {}}
    original_forward = engine.model.forward
    base = _unwrap(engine.model)
    thinker = getattr(base, "thinker", None)
    original_thinker_forward = thinker.forward if thinker is not None else None
    original_generate = engine.template.generate
    # Encoding must fail atomically: a dropped item would shift captured row IDs.
    engine.strict = True

    def capture(original, *args, **kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args and torch.is_tensor(args[0]):
            ids = args[0]
        active = None
        if torch.is_tensor(ids) and ids.ndim >= 2 and ids.shape[-1] == 1:
            active = ids[..., -1].eq(seg_id)
        if active is not None and bool(active.any()):
            kwargs["output_hidden_states"] = True
        output = original(*args, **kwargs)
        if active is not None and bool(active.any()) and getattr(output, "hidden_states", None):
            hidden = output.hidden_states[-1][:, -1, :].detach()
            for index in torch.nonzero(active, as_tuple=False).flatten().tolist():
                captured["post"][index] = hidden[index]
        return output

    @functools.wraps(original_forward)
    def outer_forward(*args, **kwargs):
        return capture(original_forward, *args, **kwargs)

    @functools.wraps(original_thinker_forward)
    def thinker_forward(*args, **kwargs):
        return capture(original_thinker_forward, *args, **kwargs)

    def generate(model, *args, **kwargs):
        result = original_generate(model, *args, **kwargs)
        captured["output"] = result
        return result

    engine.model.forward = outer_forward
    if thinker is not None:
        thinker.forward = thinker_forward
    engine.template.generate = generate

    def hook(kwargs):
        kwargs["generation_config"].return_dict_in_generate = True
        kwargs["generation_config"].output_hidden_states = False
        kwargs["generation_config"].num_beams = 1
        kwargs["generation_config"].num_return_sequences = 1
        captured["prompt_len"] = int(kwargs["inputs"]["input_ids"].shape[-1])
        return kwargs

    try:
        results = engine._infer(requests, request_config, pre_infer_hook=hook)
    finally:
        engine.model.forward = original_forward
        if thinker is not None:
            thinker.forward = original_thinker_forward
        engine.template.generate = original_generate

    if len(results) != len(requests) or any(isinstance(r, Exception) for r in results):
        raise RuntimeError(f"Batch encoding failed: {results}")
    output = captured["output"]
    sequences = output["sequences"]
    generated = engine.template.get_generate_ids(sequences, captured["prompt_len"])
    base = _unwrap(engine.model)
    device = next(base.thinker.model.norm.parameters()).device
    base.text_hidden_fcs.to(device)
    dtype = next(base.text_hidden_fcs.parameters()).dtype
    values = []
    for index, result in enumerate(results):
        response = result.choices[0].message.content or ""
        ids = generated[index].tolist()
        ids = [token for token in ids if token != engine.template.tokenizer.pad_token_id]
        positions = [i for i, token in enumerate(ids) if token == seg_id]
        hidden = captured["post"].get(index)
        prompt = None
        if len(positions) == 1 and hidden is not None:
            projected = base.text_hidden_fcs(hidden.to(device=device, dtype=dtype)).float()
            if projected.ndim == 2:
                projected = projected[0]
            prompt = projected.detach().cpu()
        values.append({
            "response": response,
            "seg_count": len(positions),
            "completion_tokens": len(ids),
            "generated_ids": ids,
            "finish_reason": result.choices[0].finish_reason,
            "prompt": prompt,
        })
    return values


def _fingerprint(path: Path):
    path = path.resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_journal(path: Path):
    """Last record wins on retry. Recover only a truncated final line."""
    records = {}
    if not path.exists():
        return records
    with path.open("rb+") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n") or handle.read(1):
                    raise ValueError(f"Corrupt JSONL inside {path} at byte {start}")
                handle.truncate(start)
                break
            if not line.endswith(b"\n"):
                handle.write(b"\n")
            records[record["case_index"]] = record
    return records


def _append(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _decoder_state(checkpoint: Path):
    from safetensors import safe_open

    decoder = {}
    with safe_open(str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".mask_decoder." not in key:
                continue
            suffix = key.split(".mask_decoder.", 1)[1]
            # Saved wrappers can nest in either order.
            while any(suffix.startswith(p) for p in ("decoder.", "modules_to_save.default.", "original_module.")):
                suffix = suffix.split(".", 1)[1] if not suffix.startswith("modules_to_save.default.") else suffix[len("modules_to_save.default."):]
            if suffix in decoder:
                raise ValueError(f"Ambiguous saved decoder tensor: {suffix}")
            decoder[suffix] = handle.get_tensor(key)
    if not decoder:
        raise ValueError(f"No trained SAM mask decoder in {checkpoint}")
    return decoder


def _generation_manifest(args):
    settings = {
        "FPS": str(args.video_fps), "AURORA_VIDEO_FPS": str(args.video_fps),
        "FPS_MAX_FRAMES": "10" if args.family == "qwen3" else "768",
        "VIDEO_MAX_TOKEN_NUM": "128" if args.family == "qwen3" else "768",
        "IMAGE_MAX_TOKEN_NUM": "1024" if args.family == "qwen3" else "16384",
        "ENABLE_AUDIO_OUTPUT": "0", "USE_AUDIO_IN_VIDEO": "0",
    }
    for name, default in settings.items():
        os.environ.setdefault(name, default)
    os.environ["AURORA_NUM_SAM_FRAMES"] = str(args.num_frames)
    if os.environ["ENABLE_AUDIO_OUTPUT"] != "0" or os.environ["USE_AUDIO_IN_VIDEO"] != "0":
        raise ValueError("Use ENABLE_AUDIO_OUTPUT=0 and USE_AUDIO_IN_VIDEO=0; audio is provided separately")
    return {
        "schema": 2, "family": args.family, "model": str(args.model.resolve()),
        "checkpoint": _fingerprint(args.checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256(args.checkpoint / "adapter_config.json"),
        "sam_checkpoint": _fingerprint(args.sam_checkpoint),
        "dataset": {s: _sha256(args.data_dir / f"refavs_{s}.jsonl") for s in args.splits},
        "splits": args.splits, "max_samples": args.max_samples,
        "shard": args.shard, "shards": args.shards, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "num_frames": args.num_frames,
        "attn_impl": args.attn_impl, "max_memory_per_gpu": args.max_memory_per_gpu,
        "environment": {key: os.environ[key] for key in settings},
    }


def _generation_summary(args, split, model_load_sec):
    records = list(_read_journal(args.output / f"{split}.jsonl").values())
    _write_json(args.output / f"summary_{split}.generated.json", {
        "split": split, "samples": len(records), "statuses": dict(Counter(r["status"] for r in records)),
        "generation_sec": sum(r["batch_sec"] / r["batch_items"] for r in records),
        "model_load_sec": model_load_sec,
    })


def generate_stage(args):
    from safetensors.torch import save_file

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "generation_manifest.json"
    expected = _generation_manifest(args)
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output already exists: use --resume or a new --output directory")
        actual = json.loads(manifest_path.read_text())
        if actual["config"] != expected:
            raise ValueError("Resume configuration/checkpoint/dataset differs from generation_manifest.json")
        if actual.get("decoder_sha256") and actual["decoder_sha256"] != _sha256(args.output / "mask_decoder.safetensors"):
            raise ValueError("Exported decoder checksum mismatch")
    else:
        if any(args.output.iterdir()):
            raise ValueError("Use an empty output directory for a new generation run")
        actual = {"config": expected}
        _write_json(manifest_path, actual)
    pending = {}
    for split in args.splits:
        selected = _load_rows(args.data_dir, split, args.max_samples, args.shard, args.shards)
        done = _read_journal(args.output / f"{split}.jsonl")
        pending[split] = [(i, row) for i, row in selected if i not in done or (
            args.retry_errors and done[i]["status"] in {"error", "capture_missing"})]
        for i, row in selected:
            if i in done and done[i]["status"] == "ok":
                path = args.output / done[i]["prompt_embedding"]
                if not path.is_file():
                    raise FileNotFoundError(f"Missing saved embedding: {path}")
                if done[i].get("embedding_sha256") and _sha256(path) != done[i]["embedding_sha256"]:
                    raise ValueError(f"Saved embedding checksum mismatch: {path}")
    if not any(pending.values()):
        for split in args.splits:
            _generation_summary(args, split, actual.get("model_load_sec", 0.0))
        print("All selected generation records already completed", flush=True)
        return
    os.environ["AURORA_SAM_CHECKPOINT"] = str(args.sam_checkpoint.resolve())
    started = time.perf_counter()
    engine = _build_engine(args)
    from swift.infer_engine import InferRequest, RequestConfig
    base = _unwrap(engine.model)
    actual["sam_dtypes"] = {
        name: str(next(getattr(base.sam, name).parameters()).dtype).removeprefix("torch.")
        for name in ("image_encoder", "prompt_encoder", "mask_decoder")
    }
    decoder_path = args.output / "mask_decoder.safetensors"
    if not decoder_path.exists():
        save_file(_decoder_state(args.checkpoint), str(decoder_path))
    actual["decoder_sha256"] = _sha256(decoder_path)
    actual["versions"] = {"torch": torch.__version__}
    import transformers, swift
    actual["versions"].update(transformers=transformers.__version__, swift=swift.__version__)
    actual["model_load_sec"] = time.perf_counter() - started
    _write_json(manifest_path, actual)
    config = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0)
    seg_id = engine.template.tokenizer.convert_tokens_to_ids("[SEG]")
    if seg_id is None or seg_id == engine.template.tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no [SEG] token")
    for split, selected in pending.items():
        embedding_dir = args.output / "embeddings" / split
        embedding_dir.mkdir(parents=True, exist_ok=True)
        with (args.output / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
            for offset in range(0, len(selected), args.batch_size):
                chunk = selected[offset:offset + args.batch_size]
                requests = [InferRequest(
                    messages=copy.deepcopy(row["messages"][:1]),
                    videos=row.get("videos"), audios=row.get("audios"),
                ) for _, row in chunk]
                started = time.perf_counter()
                try:
                    values = _generate_batch(engine, requests, config, seg_id)
                except Exception as exc:
                    # Fail a batch atomically, preserving row association. Retry with a new run
                    # and smaller batch size for OOM, rather than silently dropping samples.
                    values = [{"status": "error", "error": f"{type(exc).__name__}: {exc}"} for _ in chunk]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elapsed = time.perf_counter() - started
                for (case_index, row), value in zip(chunk, values):
                    prompt = value.pop("prompt", None)
                    record = {
                        "split": split, "case_index": case_index, "sample_id": row.get("sample_id"),
                        "batch_sec": elapsed, "batch_items": len(chunk), **value,
                    }
                    if "error" not in record:
                        record["status"] = "no_seg" if value["seg_count"] == 0 else (
                            "multiple_seg" if value["seg_count"] != 1 else "capture_missing")
                        if prompt is not None:
                            if prompt.shape != (256,) or not bool(torch.isfinite(prompt).all()):
                                raise ValueError("Invalid projected [SEG] embedding")
                            path = embedding_dir / f"{case_index:08d}_{_safe_name(row.get('sample_id', ''), case_index)}.pt"
                            temporary = path.with_suffix(".tmp")
                            torch.save(prompt, temporary)
                            temporary.replace(path)
                            record.update(status="ok", prompt_embedding=str(path.relative_to(args.output)),
                                          embedding_sha256=_sha256(path))
                    _append(handle, record)
                    value["status"] = record["status"]
                print(f"[{split}] {offset + len(chunk)}/{len(selected)} batch_sec={elapsed:.2f} "
                      f"statuses={dict(Counter(v.get('status', 'generated') for v in values))}", flush=True)
        _generation_summary(args, split, actual["model_load_sec"])


def _frame_metrics(logits, target):
    prediction, truth = logits.sigmoid() >= 0.5, target >= 0.5
    inter = (prediction & truth).sum().item()
    union = (prediction | truth).sum().item()
    denominator = prediction.sum().item() + truth.sum().item()
    return (1.0 if union == 0 else inter / union,
            1.0 if denominator == 0 else 2 * inter / denominator)


@torch.inference_mode()
def segment_stage(args):
    from safetensors.torch import load_file
    from model.segment_anything import build_sam_vit_h
    from swift_plugin.segmentation import load_gt_masks, preprocess_sam_frames

    manifest = json.loads((args.output / "generation_manifest.json").read_text())
    config = manifest["config"]
    if args.checkpoint is not None and _fingerprint(args.checkpoint / "adapter_model.safetensors") != config["checkpoint"]:
        raise ValueError("Checkpoint differs from the stage 1 checkpoint")
    if _fingerprint(args.sam_checkpoint) != config["sam_checkpoint"]:
        raise ValueError("SAM base checkpoint differs from stage 1")
    if _sha256(args.output / "mask_decoder.safetensors") != manifest["decoder_sha256"]:
        raise ValueError("Exported decoder checksum mismatch")
    segment_config = {
        "generation_config": config, "decoder_sha256": manifest["decoder_sha256"],
        "sam_dtypes": manifest["sam_dtypes"], "save_masks": args.save_masks,
    }
    metadata_path = args.output / "segmentation_manifest.json"
    if metadata_path.exists():
        if not args.resume:
            raise ValueError("Segmentation output exists: use --resume or a new run directory")
        if json.loads(metadata_path.read_text()) != segment_config:
            raise ValueError("Segmentation resume settings differ")
    else:
        _write_json(metadata_path, segment_config)
    splits = args.splits or config["splits"]
    work = {}
    for split in splits:
        if split not in config["dataset"] or _sha256(args.data_dir / f"refavs_{split}.jsonl") != config["dataset"][split]:
            raise ValueError(f"Dataset differs from stage 1: {split}")
        records = _read_journal(args.output / f"{split}.jsonl")
        done = _read_journal(args.output / f"{split}.segmented.jsonl")
        work[split] = (records, done)
    needs_model = any(i not in done or done[i].get("generation_record_sha256") != _record_digest(records[i])
                      or (args.retry_errors and done[i]["status"] == "error")
                      for records, done in work.values() for i in records)
    if not needs_model:
        print("All available segmentation records already completed", flush=True)
    device = torch.device(args.device)
    load_sec = 0.0
    if needs_model:
        started = time.perf_counter()
        sam = build_sam_vit_h(str(args.sam_checkpoint)).to(device).eval()
        sam.mask_decoder.load_state_dict(load_file(str(args.output / "mask_decoder.safetensors")), strict=True)
        for name, dtype in manifest["sam_dtypes"].items():
            getattr(sam, name).to(dtype=getattr(torch, dtype))
        load_sec = time.perf_counter() - started
    cache = OrderedDict()
    for split, (records, done) in work.items():
        source = dict(_load_rows(args.data_dir, split, -1, 0, 1))
        with (args.output / f"{split}.segmented.jsonl").open("a", encoding="utf-8") as handle:
            for case_index, record in sorted(records.items()):
                digest = _record_digest(record)
                if (case_index in done and done[case_index].get("generation_record_sha256") == digest
                        and not (args.retry_errors and done[case_index]["status"] == "error")):
                    continue
                result = {**record, "generation_status": record["status"], "generation_record_sha256": digest}
                started = time.perf_counter()
                try:
                    row = source[case_index]
                    if row.get("sample_id") != record["sample_id"]:
                        raise ValueError("Sample ID/index mismatch")
                    if record["status"] != "ok":
                        result.update(status="skipped", iou=0.0, dice=0.0)
                    else:
                        if not row.get("mask_paths"):
                            raise ValueError("Missing ground-truth mask_paths")
                        prompt_path = args.output / record["prompt_embedding"]
                        if record.get("embedding_sha256") and _sha256(prompt_path) != record["embedding_sha256"]:
                            raise ValueError("Saved embedding checksum mismatch")
                        prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
                        if prompt.shape != (256,) or not bool(torch.isfinite(prompt).all()):
                            raise ValueError("Expected finite [256] prompt embedding")
                        frames = tuple(row.get("sam_frame_paths") or [])
                        result["image_cache_hit"] = frames in cache
                        if frames in cache:
                            features, resize, original = cache.pop(frames)
                        else:
                            pixels, resize, original = preprocess_sam_frames(frames, config["num_frames"])
                            features = sam.image_encoder(pixels.to(device=device, dtype=next(sam.image_encoder.parameters()).dtype))
                            features = features.to(dtype=next(sam.mask_decoder.parameters()).dtype)
                        if args.cache_size > 0:
                            cache[frames] = (features, resize, original)
                            while len(cache) > args.cache_size:
                                cache.popitem(last=False)
                        sparse, dense = sam.prompt_encoder(
                            points=None, boxes=None, masks=None,
                            text_embeds=prompt.to(device=device, dtype=next(sam.prompt_encoder.parameters()).dtype).view(1, 1, 256),
                        )
                        low, _ = sam.mask_decoder(
                            features, image_pe=sam.prompt_encoder.get_dense_pe().to(features.dtype),
                            sparse_prompt_embeddings=sparse.to(features.dtype), dense_prompt_embeddings=dense.to(features.dtype),
                            multimask_output=False,
                        )
                        # Match the existing REF-AVS evaluator: masks share the clip's first
                        # frame size, metrics are averaged over frames then over samples.
                        prediction = sam.postprocess_masks(low, input_size=tuple(resize[0]), original_size=tuple(original[0]))[:, 0]
                        target = load_gt_masks(row.get("mask_paths") or [], original[0], config["num_frames"]).to(device)
                        metrics = [_frame_metrics(p, t) for p, t in zip(prediction, target)]
                        result.update(status="ok", iou=sum(v[0] for v in metrics) / len(metrics),
                                      dice=sum(v[1] for v in metrics) / len(metrics), frames_scored=len(metrics))
                        if args.save_masks:
                            from PIL import Image
                            mask_dir = args.output / "masks" / split / f"{case_index:08d}"
                            mask_dir.mkdir(parents=True, exist_ok=True)
                            masks = ((prediction.sigmoid() >= 0.5).to(torch.uint8) * 255).cpu().numpy()
                            for frame_index, mask in enumerate(masks):
                                Image.fromarray(mask).save(mask_dir / f"{frame_index:02d}.png")
                            result["mask_dir"] = str(mask_dir.relative_to(args.output))
                except Exception as exc:
                    result.update(status="error", iou=0.0, dice=0.0, error=f"{type(exc).__name__}: {exc}")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                result["segmentation_sec"] = time.perf_counter() - started
                _append(handle, result)
                done[case_index] = result
                print(f"[{split}] segmented={len(done)}/{len(records)} status={result['status']} "
                      f"sec={result['segmentation_sec']:.2f}", flush=True)
        scored = [r for r in done.values() if r["status"] == "ok"]
        selected_count = len(_load_rows(args.data_dir, split, config["max_samples"], config["shard"], config["shards"]))
        summary = {
            "split": split, "selected_samples": selected_count, "generation_records": len(records),
            "samples": len(done), "scored": len(scored), "statuses": dict(Counter(r["status"] for r in done.values())),
            "generation_statuses": dict(Counter(r["status"] for r in records.values())),
            "mean_iou_scored": sum(r["iou"] for r in scored) / max(len(scored), 1),
            "mean_dice_scored": sum(r["dice"] for r in scored) / max(len(scored), 1),
            "mean_iou_all": sum(r["iou"] for r in scored) / max(selected_count, 1),
            "mean_dice_all": sum(r["dice"] for r in scored) / max(selected_count, 1),
            "segmentation_sec": sum(r["segmentation_sec"] for r in done.values()),
            "image_cache_hits": sum(r.get("image_cache_hit", False) for r in done.values()),
            "sam_load_sec": load_sec, "complete": len(records) == selected_count and len(done) == selected_count,
        }
        _write_json(args.output / f"summary_{split}.segmented.json", summary)
        print(json.dumps(summary), flush=True)


def summarize_stage(args):
    """Aggregate metrics without relocating embeddings or copying journals."""
    manifests = [json.loads((p / "generation_manifest.json").read_text()) for p in args.inputs]
    configs = [m["config"] for m in manifests]
    reference = {k: v for k, v in configs[0].items() if k != "shard"}
    if any({k: v for k, v in c.items() if k != "shard"} != reference for c in configs):
        raise ValueError("Shard runs have different checkpoints, datasets or generation settings")
    if any(m["decoder_sha256"] != manifests[0]["decoder_sha256"] or m["sam_dtypes"] != manifests[0]["sam_dtypes"] for m in manifests):
        raise ValueError("Shard runs have different decoders or SAM precision")
    shards = [c["shard"] for c in configs]
    if sorted(shards) != list(range(reference["shards"])):
        raise ValueError("Provide every shard exactly once")
    args.output.mkdir(parents=True, exist_ok=True)
    for split in reference["splits"]:
        summaries = [json.loads((p / f"summary_{split}.segmented.json").read_text()) for p in args.inputs]
        total = sum(s["selected_samples"] for s in summaries)
        scored = sum(s["scored"] for s in summaries)
        statuses = Counter()
        generation_statuses = Counter()
        for s in summaries:
            statuses.update(s["statuses"])
            generation_statuses.update(s.get("generation_statuses", {}))
        summary = {
            "split": split, "selected_samples": total, "scored": scored,
            "statuses": dict(statuses), "complete": all(s["complete"] for s in summaries),
            "generation_statuses": dict(generation_statuses),
            "inputs": [str(p.resolve()) for p in args.inputs],
            "segmentation_sec_sum": sum(s["segmentation_sec"] for s in summaries),
        }
        for metric in ("iou", "dice"):
            metric_sum = sum(s[f"mean_{metric}_scored"] * s["scored"] for s in summaries)
            summary[f"mean_{metric}_scored"] = metric_sum / max(scored, 1)
            summary[f"mean_{metric}_all"] = metric_sum / max(total, 1)
        _write_json(args.output / f"summary_{split}.segmented.json", summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    def common(q):
        q.add_argument("--data-dir", type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--sam-checkpoint", type=Path, default=Path(
            "models/sam_vit_h_4b8939.pth"))
        q.add_argument("--resume", action="store_true")
        q.add_argument("--retry-errors", action="store_true", help="With --resume, retry failed records")
    generate = sub.add_parser("generate", help="LLM inference and projected [SEG] embedding capture")
    common(generate)
    generate.add_argument("--family", choices=["qwen2_5", "qwen3"], required=True)
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--splits", nargs="+", default=["test_s", "test_u"])
    generate.add_argument("--max-samples", type=int, default=-1)
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument("--shard", type=int, default=0)
    generate.add_argument("--shards", type=int, default=1)
    generate.add_argument("--max-new-tokens", type=int, help="Default Qwen3=256, Qwen2.5=512")
    generate.add_argument("--max-memory-per-gpu", default="52GiB")
    generate.add_argument("--attn-impl", default="sdpa")
    generate.add_argument("--video-fps", type=float, help="Default Qwen3=1, Qwen2.5=2; match training")
    generate.add_argument("--num-frames", type=int, default=10)
    segment = sub.add_parser("segment", help="SAM-only decoding, cached image features and metrics")
    common(segment)
    segment.add_argument("--checkpoint", type=Path, help="Optional verification; trained decoder is exported by stage 1")
    segment.add_argument("--splits", nargs="+")
    segment.add_argument("--device", default="cuda:0")
    segment.add_argument("--cache-size", type=int, default=32, help="LRU clips in GPU memory; 0 disables caching")
    segment.add_argument("--save-masks", action="store_true")
    summarize = sub.add_parser("summarize", help="Aggregate all shard summaries with weighted metrics")
    summarize.add_argument("--inputs", type=Path, nargs="+", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "retry_errors", False) and not args.resume:
        parser.error("--retry-errors requires --resume")
    if args.command == "generate":
        if args.max_new_tokens is None:
            args.max_new_tokens = 256 if args.family == "qwen3" else 512
        if args.batch_size < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
            parser.error("Require batch-size>=1 and 0<=shard<shards")
        if args.max_samples == 0 or args.max_samples < -1 or args.max_new_tokens < 1 or args.num_frames < 1:
            parser.error("Invalid sample/token/frame count")
        if args.video_fps is None:
            args.video_fps = 1.0 if args.family == "qwen3" else 2.0
        if args.video_fps <= 0:
            parser.error("video-fps must be positive")
        generate_stage(args)
    elif args.command == "segment":
        if args.cache_size < 0:
            parser.error("cache-size must be >=0")
        segment_stage(args)
    else:
        summarize_stage(args)


if __name__ == "__main__":
    main()
