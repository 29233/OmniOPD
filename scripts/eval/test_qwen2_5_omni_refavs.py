#!/usr/bin/env python3
"""Evaluate an AURORA Qwen2.5-Omni checkpoint on REF-AVS test splits.

The language generation uses ms-swift's official PtEngine.  When enabled,
the generated response is then encoded through the AURORA training forward to
obtain SAM masks and frame-level IoU/Dice scores.  This keeps deployment and
evaluation on the same model/template path as SFT while avoiding vLLM-specific
assumptions about the custom SAM head.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompt_style_header(row: dict) -> str:
    messages = row.get("messages") or []
    if not messages or messages[0].get("role") != "user":
        raise ValueError("Missing REF-AVS user prompt")
    content = str(messages[0].get("content") or "")
    if content.count("<video>") != 1 or content.count("<audio>") != 1:
        raise ValueError("Expected exactly one video and audio placeholder")
    if "<video><audio>" not in content:
        raise ValueError("REF-AVS prompt changed the training media ordering")
    return content.split("<video><audio>", 1)[0]


def _check_task_prompt_style(train_source: Path, test_sources: dict[str, Path]) -> dict:
    """Reject old-3B raw/new-7B structured CoT prompt swaps before model load."""
    if not train_source.is_file():
        raise FileNotFoundError(f"Training prompt source unavailable: {train_source}")
    with train_source.open(encoding="utf-8") as handle:
        train = next((json.loads(line) for line in handle if line.strip()), None)
    if train is None:
        raise ValueError(f"Empty training source: {train_source}")
    header = _prompt_style_header(train)
    counts = {}
    for split, path in test_sources.items():
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        mismatches = [i for i, row in enumerate(rows)
                      if _prompt_style_header(row) != header]
        if mismatches:
            raise ValueError(
                f"{split}: {len(mismatches)} rows use a different task prompt "
                f"than the selected checkpoint's training source; first index={mismatches[0]}. "
                "Do not evaluate a raw-prompt 3B checkpoint on structured-CoT test input."
            )
        counts[split] = len(rows)
    return {
        "source": str(train_source.resolve()),
        "header_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "header_chars": len(header), "checked_test_rows": counts,
    }


def _add_import_paths() -> None:
    # Prepend repo root (swift_plugin, scripts, model) and vendored third_party/sam2.
    # Do NOT add extra swift roots:
    # historically this injected MS_SWIFT_ROOT=third_party/ms-swift
    # (a swift 3.x checkout) AHEAD of the editable-installed ms-swift 4.5.2 in
    # third_party/, mixing 3.x package headers with 4.5.2 submodules in one
    # process. That mix caused "'BaseModelOutputWithPooling' object has no
    # attribute 'to'" (3.x omni code + transformers 5.16.1) on every sample,
    # and later "cannot import name 'HfConfigFactory'" (3.x swift.utils +
    # 4.5.2 metrics). ms-swift now comes from the editable install; MS_SWIFT_ROOT
    # is honored only if explicitly set for a deliberate legacy run.
    root = _repo_root()
    for candidate in (root, root / "third_party/sam2"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    swift_root = os.environ.get("MS_SWIFT_ROOT", "")
    if swift_root and swift_root not in sys.path:
        print(f"[warn] MS_SWIFT_ROOT={swift_root} is prepended; this may shadow "
              f"the editable ms-swift 4.5.2 with a legacy checkout.", flush=True)
        sys.path.insert(0, str(swift_root))


def _configure_projector_environment(
    checkpoint: Path,
    environ: MutableMapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Propagate checkpoint projector metadata and validate SAM/SAM2 backend selection."""
    env = os.environ if environ is None else environ
    projector_config_path = Path(checkpoint) / "aurora_projector_config.json"
    if not projector_config_path.is_file():
        return {}
    projector_config = json.loads(projector_config_path.read_text(encoding="utf-8"))
    architecture_environment = {
        "AURORA_PROJECTOR_HIDDEN_SIZE": str(projector_config["projector_hidden_size"]),
        "AURORA_PROJECTOR_LAYER_NORM": "1" if projector_config.get("layer_norm") else "0",
        "AURORA_VIDEO_FPS": str(projector_config["model_timeline_fps"]),
        "AURORA_NUM_SAM_FRAMES": str(projector_config["sam_frames"]),
    }
    for key, value in architecture_environment.items():
        configured = env.get(key)
        if configured is not None and configured != value:
            raise ValueError(
                f"Checkpoint architecture mismatch: {key}={configured!r}, expected {value!r} "
                f"from {projector_config_path}"
            )
        env[key] = value
    backend = str(projector_config.get("backend") or "").strip().lower()
    if backend == "sam2_video_memory":
        configured_backend = (env.get("AURORA_SEG_BACKEND") or "").strip()
        if configured_backend and configured_backend.lower() != "sam2":
            raise ValueError(
                f"Checkpoint architecture mismatch: AURORA_SEG_BACKEND={configured_backend!r}, "
                f"expected 'sam2' for backend='sam2_video_memory' from {projector_config_path}"
            )
        env["AURORA_SEG_BACKEND"] = "sam2"
        sam2_config = projector_config.get("sam2_config")
        if sam2_config and not (env.get("AURORA_SAM2_CONFIG") or "").strip():
            env["AURORA_SAM2_CONFIG"] = str(sam2_config)
    elif backend == "sam":
        configured_backend = (env.get("AURORA_SEG_BACKEND") or "").strip()
        if configured_backend and configured_backend.lower() != "sam":
            raise ValueError(
                f"Checkpoint architecture mismatch: AURORA_SEG_BACKEND={configured_backend!r}, "
                f"expected 'sam' from {projector_config_path}"
            )
        env["AURORA_SEG_BACKEND"] = "sam"
    return projector_config


def _resolve_sam2_checkpoint(environ: MutableMapping[str, str] | None = None) -> Path | None:
    """Ensure AURORA_SAM2_CHECKPOINT points to an existing file when using SAM2."""
    env = os.environ if environ is None else environ
    if (env.get("AURORA_SEG_BACKEND") or "sam").strip().lower() != "sam2":
        return None
    sam2_ckpt = (env.get("AURORA_SAM2_CHECKPOINT") or "").strip()
    if not sam2_ckpt:
        sam2_ckpt = (env.get("SAM2_CHECKPOINT") or "").strip()
        if sam2_ckpt:
            env["AURORA_SAM2_CHECKPOINT"] = sam2_ckpt
    if not sam2_ckpt:
        default_sam2 = _repo_root() / "models/sam2_hiera_large.pt"
        if default_sam2.is_file():
            env["AURORA_SAM2_CHECKPOINT"] = str(default_sam2)
            return default_sam2
        raise FileNotFoundError(
            "AURORA_SEG_BACKEND=sam2 requires a SAM2 base checkpoint, but neither "
            "AURORA_SAM2_CHECKPOINT nor SAM2_CHECKPOINT is set (and default "
            f"{default_sam2} does not exist). Set AURORA_SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt."
        )
    ckpt_path = Path(sam2_ckpt)
    if not ckpt_path.is_file():
        repo_relative = _repo_root() / sam2_ckpt
        if not ckpt_path.is_absolute() and repo_relative.is_file():
            env["AURORA_SAM2_CHECKPOINT"] = str(repo_relative)
            return repo_relative
        raise FileNotFoundError(
            f"AURORA_SAM2_CHECKPOINT points to a missing file: {sam2_ckpt!r}. "
            "Set AURORA_SAM2_CHECKPOINT (or SAM2_CHECKPOINT) to an existing SAM2 checkpoint file."
        )
    return ckpt_path


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _unwrap_model(model):
    current = model
    for _ in range(8):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            next_model = current.get_base_model()
            if next_model is current:
                break
            current = next_model
            continue
        break
    return current


def _frame_metrics(logits: torch.Tensor, target: torch.Tensor):
    prediction = logits.sigmoid() >= 0.5
    target = target >= 0.5
    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()
    pred_sum = prediction.sum().item()
    target_sum = target.sum().item()
    iou = 1.0 if union == 0 else intersection / union
    denominator = pred_sum + target_sum
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    return iou, dice


def _safe_sample_id(value: Any, case_index: int) -> str:
    """Convert a dataset id into a path component without allowing traversal."""
    text = str(value).strip() if value is not None else ""
    safe = "".join(
        char if (char.isascii() and (char.isalnum() or char in "._-")) else "_"
        for char in text
    )
    safe = safe.strip(".")
    return safe or f"case-{case_index:06d}"


def _mask_sample_dir_name(sample_id: Any, case_index: int) -> str:
    """Use the stable row index because REF-AVS contains duplicate sample ids."""
    return f"{case_index:06d}__{_safe_sample_id(sample_id, case_index)}"


def _save_masks(predictions: Iterable[torch.Tensor], mask_dir: Path, sample_id: Any, case_index: int) -> List[str]:
    """Save thresholded per-frame logits as portable grayscale PNG files."""
    from PIL import Image

    sample_dir = mask_dir / _mask_sample_dir_name(sample_id, case_index)
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for frame_index, prediction in enumerate(predictions):
        mask = (prediction.detach().float().sigmoid() >= 0.5).cpu().numpy()
        # ``aurora_last_pred_masks`` is normally [frames, height, width].
        # Accept a singleton channel/batch dimension from older checkpoints.
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2-D frame mask, got shape {mask.shape}")
        path = sample_dir / f"frame_{frame_index:03d}.png"
        Image.fromarray((mask.astype("uint8") * 255), mode="L").save(path)
        paths.append(str(path))
    return paths


def _load_rows(
    path: Path,
    max_samples: int,
    rank: int,
    world_size: int,
    degraded_audio_name: str = "",
    sample_ids: set[str] | None = None,
) -> List[Dict[str, Any]]:
    all_rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if sample_ids:
        rows = []
        for source_index, row in enumerate(all_rows):
            if str(row.get("sample_id", "")) in sample_ids:
                # Preserve the full-manifest index for targeted case joins.
                row["case_index"] = source_index
                rows.append(row)
    else:
        rows = all_rows
    if max_samples > 0:
        rows = rows[:max_samples]
    if degraded_audio_name:
        # Zero-shot audio-robustness mode: point every row's audio at the
        # degraded copy. Both the generation request and the [SEG] SAM scoring
        # read row["audios"], so the whole evaluation hears the degraded
        # audio. A missing file is a hard error — silently scoring a mixture
        # of clean/degraded rows would corrupt the comparison.
        missing: List[str] = []
        for row in rows:
            audios = row.get("audios") or []
            if not audios:
                continue
            degraded = Path(audios[0]).parent / degraded_audio_name
            if not degraded.is_file():
                missing.append(str(degraded))
                continue
            row["audios"] = [str(degraded)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} degraded audio files missing (first: {missing[0]}); "
                "run scripts/data/degrade_refavs_audio.py first"
            )
    if world_size > 1:
        rows = rows[rank::world_size]
    return rows


def _dist_barrier(world_size: int, backend: str) -> None:
    """Synchronize ranks without coupling evaluation to model collectives."""
    if world_size <= 1:
        return
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    dist.barrier()


def _extract_referring(row: Dict[str, Any]) -> str:
    """Return the source referring expression preserved in a dataset row."""
    for key in ("referring", "referring_expression", "exp"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    messages = row.get("messages") or []
    user_content = next(
        (str(message.get("content", "")) for message in messages
         if message.get("role") == "user"),
        "",
    )
    marker = "The reference is:"
    if marker in user_content:
        value = user_content.split(marker, 1)[1]
        value = value.split(" Please segment", 1)[0]
        return value.strip()
    return ""


def _extract_category(row: Dict[str, Any]) -> str:
    """Recover the annotated target category when the converted row omitted it.

    REF-AVS sample ids follow ``<video>_<category>_<instance>``.  The original
    dataset conversion uses the same ``uid.split('_')[-2]`` convention.  A
    literal ``null`` category in test_n means the split intentionally does not
    expose the target class and must not receive privileged information.
    """
    for key in ("class_name", "category", "target_category"):
        value = row.get(key)
        if value is not None and str(value).strip():
            category = str(value).strip()
            if category.lower() not in {"null", "none", "unknown"}:
                return category
    sample_id = str(row.get("sample_id") or "").strip()
    parts = sample_id.split("_")
    if len(parts) >= 2:
        category = parts[-2].strip()
        if category and category.lower() not in {"null", "none", "unknown"}:
            return category
    return ""


def _privileged_row(row: Dict[str, Any], enabled: bool) -> tuple[Dict[str, Any], str]:
    """Return a copy with the ground-truth category appended to the user turn."""
    copied = copy.deepcopy(row)
    category = _extract_category(row)
    if not enabled or not category:
        return copied, category
    messages = copied.get("messages") or []
    user_index = next(
        (index for index, message in enumerate(messages)
         if message.get("role") == "user"),
        None,
    )
    if user_index is None:
        raise ValueError("Privileged evaluation requires a user message")
    content = str(messages[user_index].get("content", ""))
    insertion = f"Hint: The target object is: {category}."
    if insertion not in content:
        content = content.rstrip() + "\n" + insertion
    messages[user_index]["content"] = content
    copied["messages"] = messages
    copied["privileged_category"] = category
    return copied, category


def _aggregate_split(
    output_dir: Path,
    split: str,
    world_size: int,
    source_path: Path | None = None,
    max_samples: int = -1,
) -> Dict[str, Any]:
    """Merge rank summaries and case rollouts after all ranks finish a split."""
    rank_summaries = []
    case_records = []
    source_rows: List[Dict[str, Any]] = []
    if source_path is not None and source_path.is_file():
        with source_path.open(encoding="utf-8") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
        if max_samples > 0:
            source_rows = source_rows[:max_samples]
    # Selected manifests may carry the original full-dataset index so a
    # targeted run can be joined back to the full evaluation without padding
    # the manifest with thousands of unrelated samples.
    source_by_case_index = {
        int(row["case_index"]): row
        for fallback_index, row in enumerate(source_rows)
        if row.get("case_index") is not None
    }
    for rank in range(world_size):
        if world_size == 1:
            summary_path = output_dir / f"summary_{split}.json"
            records_path = output_dir / f"{split}.jsonl"
        else:
            summary_path = output_dir / f"summary_{split}.rank{rank}.json"
            records_path = output_dir / f"{split}.rank{rank}.jsonl"
        if not summary_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(
                f"Missing rank output for {split}: {summary_path} or {records_path}"
            )
        rank_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        rank_records = []
        with records_path.open(encoding="utf-8") as handle:
            for local_index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                # Backfill fields for rank files produced by the previous
                # evaluator version, so existing probes can be aggregated
                # without rerunning inference.
                record.setdefault("rank", rank)
                record.setdefault("case_index", rank + local_index * world_size)
                record.setdefault("rollout", record.get("response"))
                record.setdefault("status", "error" if record.get("error") else "ok")
                source_index = int(record["case_index"])
                source_row = source_by_case_index.get(source_index)
                if source_row is None and 0 <= source_index < len(source_rows):
                    source_row = source_rows[source_index]
                if source_row is not None:
                    record.setdefault("referring", _extract_referring(source_row))
                    source_messages = source_row.get("messages") or []
                    record.setdefault(
                        "prompt",
                        source_messages[0].get("content", "") if source_messages else "",
                    )
                    record.setdefault(
                        "class_name",
                        source_row.get("class_name", ""),
                    )
                    record.setdefault("privileged_category", _extract_category(source_row))
                rank_records.append(record)
                case_records.append(record)
        # Normalize old rank files in place so both the per-rank and merged
        # JSONL outputs expose the same rollout/referring schema.
        with records_path.open("w", encoding="utf-8") as handle:
            for record in rank_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    case_records.sort(key=lambda item: (item.get("case_index", -1), item.get("rank", -1)))
    merged_path = output_dir / f"{split}.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for record in case_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    totals = {
        "samples": sum(item.get("samples", 0) for item in rank_summaries),
        "errors": sum(item.get("errors", 0) for item in rank_summaries),
        "seg_emitted": sum(item.get("seg_emitted", 0) for item in rank_summaries),
        "seg_scored": sum(item.get("seg_scored", 0) for item in rank_summaries),
        "iou_sum": sum(item.get("iou_sum", 0.0) for item in rank_summaries),
        "dice_sum": sum(item.get("dice_sum", 0.0) for item in rank_summaries),
        "privileged_category_applied": sum(
            item.get("privileged_category_applied", 0) for item in rank_summaries
        ),
        "seg_features_saved": sum(item.get("seg_features_saved", 0) for item in rank_summaries),
    }
    aggregate = {
        "split": split,
        "world_size": world_size,
        "rank_summaries": [
            str(output_dir / (f"summary_{split}.json" if world_size == 1 else f"summary_{split}.rank{rank}.json"))
            for rank in range(world_size)
        ],
        "case_output": str(merged_path),
        **totals,
        "seg_rate": totals["seg_emitted"] / max(totals["samples"], 1),
        "privileged_category": any(
            item.get("privileged_category", False) for item in rank_summaries
        ),
        # IoU/Dice are averaged over samples for which the model emitted one
        # valid [SEG] token and SAM produced a mask.
        "mean_iou_scored": totals["iou_sum"] / max(totals["seg_scored"], 1),
        "mean_dice_scored": totals["dice_sum"] / max(totals["seg_scored"], 1),
        "rank_count": len(rank_summaries),
    }
    summary_path = output_dir / f"summary_{split}.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    return aggregate


def _sam_scores(
    model,
    template,
    row: Dict[str, Any],
    response: str,
    device: torch.device,
    mask_dir: Path | None = None,
    case_index: int = 0,
    image_cache: Dict[str, Any] | None = None,
    first_frame_only: bool = False,
    capture_seg_features: bool = False,
):
    full_row = copy.deepcopy(row)
    full_row["messages"] = row["messages"][:1] + [{"role": "assistant", "content": response}]
    # Generation checks the prompt against max_length, then appends up to
    # max_new_tokens. Scoring that accepted response must not reject/truncate
    # prompt+response merely because their combined length exceeds the prompt
    # limit. Preserve every generated token (especially SEG) during replay.
    previous_max_length = template.max_length
    template.max_length = None
    template.set_mode("train")
    try:
        encoded = template.encode(full_row)
        encoded.pop("_extra_kwargs", None)
        batch = _to_device(template.data_collator([encoded]), device)
        template.register_post_encode_hook([model])
        base = _unwrap_model(model)
        if first_frame_only:
            # Keep the LLM's complete video/prompt/response context, but ask
            # the SAM branch to decode only frame zero. This is used for exact
            # cached-rollout first-frame ablations without rerunning generation.
            batch["sam_pixel_values"] = batch["sam_pixel_values"][:, :1]
            batch["gt_masks"] = batch["gt_masks"][:, :1]
        frame_paths = tuple(str(path) for path in (row.get("sam_frame_paths") or []))
        frame_key = ("first", frame_paths[0]) if first_frame_only else ("all", *frame_paths)
        # The original SAM backend can cache independent per-frame image
        # embeddings. SAM2 cannot use that cache: its decoder consumes Hiera
        # features while updating temporal memory inside sam2_head.forward().
        # Let the SAM2 forward own that complete video path.
        can_cache_image_embeddings = image_cache is not None and hasattr(base, "sam")
        if can_cache_image_embeddings and image_cache.get("key") == frame_key:
            batch["sam_image_embeddings"] = image_cache["value"]
        elif can_cache_image_embeddings:
            images = batch["sam_pixel_values"]
            image_dtype = next(base.sam.image_encoder.parameters()).dtype
            encoder_context = torch.autocast("cuda", enabled=False) if images.is_cuda else nullcontext()
            with encoder_context, torch.no_grad():
                embeddings = base.sam.image_encoder(
                    images.to(dtype=image_dtype).flatten(0, 1)
                )
            batch["sam_image_embeddings"] = embeddings.view(
                images.shape[0], images.shape[1], *embeddings.shape[1:]
            )
            image_cache.clear()
            image_cache.update(key=frame_key, value=batch["sam_image_embeddings"])
        with torch.no_grad():
            model(**batch)
        predictions = getattr(base, "aurora_last_pred_masks", None)
        targets = batch.get("gt_masks")
        if not predictions or targets is None:
            raise RuntimeError("The generated [SEG] response did not produce AURORA masks")
        ious: List[float] = []
        dices: List[float] = []
        for prediction, target in zip(predictions, targets):
            target = target.to(device=prediction.device, dtype=prediction.dtype)
            for frame_prediction, frame_target in zip(prediction, target):
                iou, dice = _frame_metrics(frame_prediction, frame_target)
                ious.append(iou)
                dices.append(dice)
        scores = {
            "iou": sum(ious) / max(len(ious), 1),
            "dice": sum(dices) / max(len(dices), 1),
            "frames_scored": len(ious),
            "frame_ious": ious,
            "frame_dices": dices,
        }
        if capture_seg_features:
            hidden_feature = getattr(base, "aurora_last_seg_hidden", None)
            prompt_feature = getattr(base, "aurora_last_seg_prompts", None)
            if hidden_feature is None or prompt_feature is None:
                raise RuntimeError("SAM2 forward did not expose SEG features")
            if hidden_feature.shape[0] != 1 or prompt_feature.shape[0] != 1:
                raise RuntimeError(
                    f"Expected one SEG feature, got hidden={tuple(hidden_feature.shape)} "
                    f"projected={tuple(prompt_feature.shape)}"
                )
            scores["_seg_hidden_feature"] = hidden_feature[0].float().cpu()
            scores["_seg_projected_feature"] = prompt_feature[0].float().cpu()
        if mask_dir is not None:
            scores["mask_dir"] = str(mask_dir / _mask_sample_dir_name(row.get("sample_id"), case_index))
            scores["mask_paths"] = _save_masks(
                predictions[0], mask_dir, row.get("sample_id"), case_index
            )
        return scores
    finally:
        template.max_length = previous_max_length
        template.set_mode("transformers")  # "pt" is deprecated in swift 4.x


def _build_engine(args):
    # Importing the registration file is required before PtEngine resolves the
    # custom model/template names.
    import swift_plugin.aurora_qwen2_5_omni  # noqa: F401
    from safetensors.torch import load_file, save_file
    # ms-swift 4.x moved everything out of swift.llm: PtEngine is now
    # TransformersEngine (same class, renamed) in swift.infer_engine. The
    # old swift.llm path only exists in swift 3.x — running this script
    # there pairs a 3.x template stack with transformers 5.16.1 and crashes
    # with "'BaseModelOutputWithPooling' object has no attribute 'to'" on
    # every sample (3.x assumes get_audio_features returns a tensor).
    from swift.infer_engine import TransformersEngine as PtEngine
    from swift.tuners import Swift

    device = torch.device(args.device)
    engine = PtEngine(
        args.model,
        torch_dtype=getattr(torch, getattr(args, 'torch_dtype', 'float16')),
        model_type="aurora_qwen2_5_omni",
        new_special_tokens=["[SEG]"],
        attn_impl=args.attn_impl,
        device_map={"": str(device)},
        download_model=False,
        max_batch_size=1,
    )
    # PEFT treats a broad ``modules_to_save=['embed_tokens']`` entry as
    # matching both Omni thinker and talker embeddings.  SFT only updates the
    # thinker embedding, so older ms-swift/PEFT versions fail if the untouched
    # talker key is absent.  Add that base tensor to a temporary adapter copy;
    # the user's checkpoint remains unchanged and all other loading is native.
    adapter_dir = Path(args.checkpoint)
    state_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not state_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint must contain adapter_model.safetensors and adapter_config.json: {adapter_dir}")
    # Restore and validate the trained sidecar BEFORE PEFT wraps
    # modules_to_save. The adapter can also contain active SAM2 weights:
    # restoring the sidecar made no difference to the single audited smoke
    # mask, so that smoke does NOT establish that the older path was random.
    # Afterwards these modules are ModulesToSaveWrapper objects and their
    # state_dict keys no longer match the unwrapped sidecar.
    from swift_plugin.aurora_qwen2_5_omni import _restore_trained_sam_weights
    if getattr(engine.model, "sam2_head", None) is not None:
        sidecar = adapter_dir / "sam2_video_decoder.safetensors"
        if not sidecar.is_file():
            raise FileNotFoundError(f"Trained SAM2 checkpoint sidecar missing: {sidecar}")
    _restore_trained_sam_weights(engine.model, str(adapter_dir))
    state = load_file(str(state_path), device="cpu")
    talker_key = "base_model.model.talker.model.embed_tokens.weight"
    # Qwen2.5-Omni model variants differ across Transformers/ms-swift
    # versions: some expose a talker module while others load only the
    # thinker used by this evaluation path.  Only synthesize the compatibility
    # tensor when the loaded model actually has a talker; otherwise the
    # missing talker key is irrelevant and the adapter can be loaded directly.
    talker_weight = None
    talker_module = getattr(engine.model, "talker", None)
    if talker_key not in state and talker_module is not None:
        talker_model = getattr(talker_module, "model", talker_module)
        talker_embed = getattr(talker_model, "embed_tokens", None)
        if talker_embed is not None:
            talker_weight = talker_embed.weight.detach().cpu().contiguous()
    if talker_weight is not None:
        state[talker_key] = talker_weight
        temp_dir = tempfile.TemporaryDirectory(prefix="aurora_adapter_")
        compat_dir = Path(temp_dir.name)
        shutil.copy2(config_path, compat_dir / "adapter_config.json")
        save_file(state, compat_dir / "adapter_model.safetensors", metadata={"format": "pt"})
        engine.model = Swift.from_pretrained(engine.model, str(compat_dir))
        # Keep the temporary directory alive for the lifetime of the engine.
        engine._aurora_adapter_tmp = temp_dir
    else:
        engine.model = Swift.from_pretrained(engine.model, str(adapter_dir))
    # PEFT may materialize ``modules_to_save`` on CPU even when the base Omni
    # model was loaded with a single-device map.  Keep all custom AURORA
    # modules colocated for the post-generation SAM forward.
    engine.model.to(device)
    # ms-swift 4.x: the engine already built the template in its constructor
    # (engine.template, bound to the wrapped Processor). Rebuilding one from
    # the raw HF processor fails on 4.x (get_template expects a
    # swift.model.Processor with .model_info).
    template = engine.template
    # Explicit settings from the selected training run take precedence over
    # inference template defaults. Other launchers retain their existing defaults.
    for key in ('max_pixels', 'max_length'):
        value = getattr(args, key, None)
        if value is not None:
            setattr(template, key, value)
    # The standalone script does not run through Trainer, so bind the loaded
    # model explicitly.  The multimodal collator needs it to compute Omni
    # position ids instead of constructing a meta-device dummy model.
    template.model = engine.model
    template.set_mode("transformers")  # "pt" is deprecated in swift 4.x
    return engine, template, device


def _evaluate_split(args, engine, template, device, split: str, output_dir: Path, rank: int, world_size: int):
    from swift.infer_engine import InferRequest, RequestConfig

    path = Path(args.data_dir) / f"refavs_{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing test split: {path}")
    rows = _load_rows(
        path,
        args.max_samples,
        rank,
        world_size,
        args.degraded_audio_name,
        set(args.sample_ids or []),
    )
    split_mask_dir = None
    if args.run_sam and args.save_masks:
        split_mask_dir = Path(args.mask_dir) / split
    output_name = f"{split}.rank{rank}.jsonl" if world_size > 1 else f"{split}.jsonl"
    output_path = output_dir / output_name
    feature_name = f"seg_features_{split}.rank{rank}.safetensors"
    feature_path = output_dir / feature_name
    feature_case_indices: List[int] = []
    hidden_features: List[torch.Tensor] = []
    projected_features: List[torch.Tensor] = []
    seg_id = template.tokenizer.convert_tokens_to_ids("[SEG]")
    totals = {
        "samples": 0,
        "errors": 0,
        "seg_emitted": 0,
        "seg_scored": 0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
        "privileged_category_applied": 0,
    }
    started = time.time()
    sam_image_cache: Dict[str, Any] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            # The source index is stable across ranks and lets rank 0 restore
            # the original dataset order in the merged rollout file.
            # Preserve an explicit original index in targeted manifests;
            # regular full manifests continue to use rank-interleaved order.
            case_index = int(row.get("case_index", rank + (index - 1) * world_size))
            eval_row, privileged_category = _privileged_row(row, args.privileged_category)
            if args.prompt_suffix:
                messages = copy.deepcopy(eval_row.get("messages") or [])
                if not messages or messages[0].get("role") != "user":
                    raise ValueError("Oracle prompt suffix requires a leading user message")
                messages[0]["content"] = (
                    str(messages[0].get("content") or "").rstrip()
                    + "\n\n"
                    + args.prompt_suffix.strip()
                )
                eval_row["messages"] = messages
            if args.privileged_category and privileged_category:
                totals["privileged_category_applied"] += 1
            request = InferRequest(
                messages=copy.deepcopy(eval_row["messages"][:1]),
                videos=eval_row.get("videos"),
                audios=eval_row.get("audios"),
            )
            totals["samples"] += 1
            record = {
                "split": split, "case_index": case_index, "rank": rank,
                "sample_id": row.get("sample_id"), "referring": _extract_referring(row),
                "clip_id": row.get("clip_id"), "privilege_type": row.get("privilege_type"),
                "privilege_source_clip": row.get("privilege_source_clip"),
                "prompt": (eval_row.get("messages") or [{}])[0].get("content", ""),
                "class_name": _extract_category(row), "privileged_category": privileged_category,
                "privileged_category_applied": bool(args.privileged_category and privileged_category),
                "rollout": None, "response": None,
            }
            try:
                result = engine.infer(
                    [request],
                    RequestConfig(
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed + case_index,
                    ),
                    use_tqdm=False,
                )[0]
                response = result.choices[0].message.content or ""
                generated_ids = template.tokenizer(response, add_special_tokens=False)["input_ids"]
                seg_count = sum(token == seg_id for token in generated_ids)
                record: Dict[str, Any] = {
                    "split": split,
                    "case_index": case_index,
                    "rank": rank,
                    "sample_id": row.get("sample_id"),
                    "referring": _extract_referring(row),
                    "clip_id": row.get("clip_id"),
                    "privilege_type": row.get("privilege_type"),
                    "privilege_source_clip": row.get("privilege_source_clip"),
                    "prompt": (eval_row.get("messages") or [{}])[0].get("content", ""),
                    "class_name": _extract_category(row),
                    "privileged_category": privileged_category,
                    "privileged_category_applied": bool(args.privileged_category and privileged_category),
                    "rollout": response,
                    "response": response,
                    "labels": row.get("messages", [{}, {}])[-1].get("content", ""),
                    "seg_count": seg_count,
                    "response_token_count": len(generated_ids),
                    "generation_temperature": args.temperature,
                    "generation_top_p": args.top_p,
                    "generation_seed": args.seed + case_index,
                    "status": "ok",
                }
                if seg_count == 1:
                    totals["seg_emitted"] += 1
                    if args.run_sam:
                        scores = _sam_scores(
                            engine.model,
                            template,
                            eval_row,
                            response,
                            device,
                            mask_dir=split_mask_dir,
                            case_index=case_index,
                            image_cache=sam_image_cache,
                            capture_seg_features=args.save_seg_features,
                        )
                        if args.save_seg_features:
                            hidden_feature = scores.pop("_seg_hidden_feature")
                            projected_feature = scores.pop("_seg_projected_feature")
                            feature_index = len(feature_case_indices)
                            feature_case_indices.append(case_index)
                            hidden_features.append(hidden_feature)
                            projected_features.append(projected_feature)
                            record.update({
                                "seg_feature_file": str(feature_path.resolve()),
                                "seg_feature_index": feature_index,
                                "seg_hidden_size": int(hidden_feature.numel()),
                                "seg_projected_size": int(projected_feature.numel()),
                            })
                        record.update(scores)
                        totals["seg_scored"] += 1
                        totals["iou_sum"] += scores["iou"]
                        totals["dice_sum"] += scores["dice"]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                # Keep completed rollouts durable during long multi-GPU runs.
                # Without an explicit flush, the default file buffer can keep
                # many successfully evaluated samples invisible on disk.
                handle.flush()
            except Exception as exc:
                totals["errors"] += 1
                # Keep a successfully generated response if SAM/replay fails.
                record.update(status="error", error=repr(exc))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if args.fail_fast:
                    raise
            if index % args.log_every == 0 or index == len(rows):
                # Running metrics in the progress line so the eval is
                # monitorable live (final numbers still go to summary_*.json).
                n = max(totals["samples"], 1)
                print(
                    f"[{split} rank={rank}] {index}/{len(rows)} "
                    f"err={totals['errors']} seg%={totals['seg_emitted'] / n:.1%} "
                    f"mIoU(scored)={totals['iou_sum'] / max(totals['seg_scored'], 1):.4f} "
                    f"n_scored={totals['seg_scored']}",
                    flush=True)
    if args.save_seg_features:
        from safetensors.torch import save_file

        if hidden_features:
            save_file({
                "case_indices": torch.tensor(feature_case_indices, dtype=torch.int64),
                "seg_hidden": torch.stack(hidden_features).to(torch.float16).contiguous(),
                "seg_projected": torch.stack(projected_features).to(torch.float16).contiguous(),
            }, str(feature_path), metadata={"split": split, "rank": str(rank)})
        else:
            save_file({
                "case_indices": torch.empty((0,), dtype=torch.int64),
                "seg_hidden": torch.empty((0, 0), dtype=torch.float16),
                "seg_projected": torch.empty((0, 0), dtype=torch.float16),
            }, str(feature_path), metadata={"split": split, "rank": str(rank)})
    summary = {
        "split": split,
        "rank": rank,
        "world_size": world_size,
        **totals,
        "seg_rate": totals["seg_emitted"] / max(totals["samples"], 1),
        "privileged_category": args.privileged_category,
        "privileged_category_applied": totals["privileged_category_applied"],
        "mean_iou_scored": totals["iou_sum"] / max(totals["seg_scored"], 1),
        "mean_dice_scored": totals["dice_sum"] / max(totals["seg_scored"], 1),
        "runtime_sec": time.time() - started,
        "output": str(output_path),
        "seg_features_saved": len(feature_case_indices),
        "seg_feature_file": str(feature_path.resolve()) if args.save_seg_features else None,
    }
    summary_path = output_dir / (f"summary_{split}.rank{rank}.json" if world_size > 1 else f"summary_{split}.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    _dist_barrier(world_size, args.dist_backend)
    if rank == 0:
        _aggregate_split(
            output_dir,
            split,
            world_size,
            source_path=path,
            max_samples=args.max_samples,
        )
    _dist_barrier(world_size, args.dist_backend)
    return summary


def main() -> None:
    _add_import_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "models/Qwen2___5-Omni-3B"))
    parser.add_argument("--checkpoint", required=True, help="ms-swift checkpoint directory containing adapter_model.safetensors")
    parser.add_argument("--data-dir", default=str(_repo_root() / "outputs/data/refavs"))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Evaluation directory; defaults to outputs/eval/<model>/<checkpoint-name>",
    )
    parser.add_argument("--splits", nargs="+", default=["test_s", "test_u", "test_n"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        default=[],
        help="Evaluate only rows with these sample_id values; repeat for multiple cases.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature; zero preserves the historical greedy evaluation protocol.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus-sampling threshold used only when temperature is nonzero.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base generation seed; the stable case_index is added per sample.",
    )
    parser.add_argument(
        "--privileged-category",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the ground-truth target category to each user prompt when available.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Append a fixed diagnostic instruction to every user prompt. "
             "For controlled oracle/routing experiments only.",
    )
    parser.add_argument("--device", default=f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max-pixels", type=int, default=None, help="Match the training template max_pixels")
    parser.add_argument("--max-length", type=int, default=None, help="Match the training template max_length")
    parser.add_argument("--dataset-timing-audit", type=Path,
                        help="Audited dataset FPS JSON, required when checkpoint has no training manifest")
    parser.add_argument("--train-prompt-source", type=Path,
                        help="Override a relocated checkpoint's saved dataset path for prompt-style parity checks")
    parser.add_argument("--match-training-settings", action="store_true",
                        help="Require matching checkpoint args.json and dual_launch.json before loading the model")
    parser.add_argument("--fps-policy", choices=['training', 'dataset'], default='training',
                        help="dataset explicitly uses audited data FPS and reports historical training FPS differences")
    parser.add_argument("--dist-backend", default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--run-sam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Directory for predicted mask PNGs; defaults to <output-dir>/masks",
    )
    parser.add_argument(
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save thresholded SAM masks as PNG files when --run-sam is enabled",
    )
    parser.add_argument(
        "--save-seg-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the last-layer SEG hidden state and projected decoder prompt per scored sample",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--degraded-audio-name", default="",
        help="Zero-shot robustness check: replace each row's audios[0] with "
             "the noise-degraded copy of this name (e.g. audio_snr10db.wav "
             "from scripts/data/degrade_refavs_audio.py) living in the same "
             "directory. Empty (default) = original audio. Missing degraded "
             "files fail the run.")
    args = parser.parse_args()
    projector_config = _configure_projector_environment(Path(args.checkpoint))
    if projector_config:
        print(f"[checkpoint-architecture] {json.dumps(projector_config, sort_keys=True)}", flush=True)
    if args.match_training_settings:
        from scripts.eval.omni_eval_settings import validate_training_settings
        checked = validate_training_settings(
            args.checkpoint, os.environ, torch_dtype=args.torch_dtype,
            max_pixels=args.max_pixels, max_length=args.max_length,
            max_new_tokens=args.max_new_tokens, attn_impl=args.attn_impl, fps_policy=args.fps_policy)
        print('[training-settings] ' + json.dumps(checked), flush=True)
    default_data_dir = _repo_root() / "outputs/data/refavs"
    if args.output_dir is None:
        checkpoint_name = _safe_sample_id(Path(args.checkpoint).name, 0)
        model_text = f"{args.model} {args.checkpoint}".lower()
        model_name = "qwen2_5_omni_7b" if "7b" in model_text else "qwen2_5_omni_3b"
        args.output_dir = str(_repo_root() / "outputs/eval" / model_name / checkpoint_name)
    if (os.environ.get("AURORA_SEG_BACKEND") or "sam").strip().lower() == "sam2":
        _resolve_sam2_checkpoint()
    else:
        default_sam = _repo_root() / "models/sam_vit_h_4b8939.pth"
        os.environ.setdefault(
            "AURORA_SAM_CHECKPOINT",
            str(default_sam) if default_sam.is_file() else "models/sam_vit_h_4b8939.pth",
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_dir is None:
        args.mask_dir = str(output_dir / "masks")
    # Fail before loading 7B weights on known train/eval preprocessing
    # mismatches. The SFT checkpoint has no dual_launch.json, so validate
    # directly against its saved args, projector and parent run manifest.
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        saved_args = json.loads((checkpoint_path / "args.json").read_text())
        source_manifest = next(
            (p for p in (checkpoint_path / "training_manifest.json",
                         checkpoint_path.parent / "training_manifest.json",
                         checkpoint_path.parent.parent / "training_manifest.json") if p.is_file()),
            None,
        )
        training = json.loads(source_manifest.read_text()) if source_manifest else {}
        if not training and args.dataset_timing_audit is None:
            raise ValueError("No saved training manifest: supply --dataset-timing-audit to verify media timing")
        audit = json.loads(args.dataset_timing_audit.read_text()) if args.dataset_timing_audit else {}
        if audit and (audit.get("unique_clips_checked", 0) < 1 or audit.get("nominal_fps", 0) <= 0):
            raise ValueError("Incomplete dataset timing audit")
        expected = {
            "max_pixels": args.max_pixels, "max_length": args.max_length,
            "torch_dtype": args.torch_dtype, "attn_impl": args.attn_impl,
        }
        for key, actual in expected.items():
            if actual is None or saved_args.get(key) != actual:
                raise ValueError(f"Checkpoint/eval mismatch: {key} train={saved_args.get(key)!r}, eval={actual!r}")
        if training:
            for key, actual in (("max_pixels", args.max_pixels), ("max_length", args.max_length),
                                ("attn_impl", args.attn_impl)):
                if key in training and training[key] != actual:
                    raise ValueError(f"Run manifest/eval mismatch: {key}")
        projector_path = checkpoint_path / "aurora_projector_config.json"
        projector = json.loads(projector_path.read_text()) if projector_path.is_file() else {}
        if projector:
            for key in ("FPS", "AURORA_VIDEO_FPS"):
                if key not in os.environ or float(os.environ[key]) != float(projector["model_timeline_fps"]):
                    raise ValueError(f"Checkpoint/eval FPS mismatch: {key}")
            if int(os.environ["AURORA_NUM_SAM_FRAMES"]) != int(projector["sam_frames"]):
                raise ValueError("Checkpoint/eval SAM frame count mismatch")
        nominal_fps = training.get("dataset_nominal_fps", audit.get("nominal_fps"))
        if audit and training.get("dataset_nominal_fps") is not None and float(audit["nominal_fps"]) != float(nominal_fps):
            raise ValueError("Training manifest/dataset audit FPS mismatch")
        data_sources = {split: Path(args.data_dir) / f"refavs_{split}.jsonl" for split in args.splits}
        datasets = saved_args.get("dataset") or []
        if isinstance(datasets, str):
            datasets = [datasets]
        if not datasets and args.train_prompt_source is None:
            raise ValueError("Checkpoint args lack a recorded training prompt source")
        prompt_style = _check_task_prompt_style(
            args.train_prompt_source or Path(datasets[0]), data_sources
        )
        def sha256(path):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            return digest.hexdigest()
        manifest = {
            "checkpoint": str(checkpoint_path.resolve()),
            "training_args": str((checkpoint_path / "args.json").resolve()),
            "training_manifest": str(source_manifest.resolve()) if source_manifest else None,
            "dataset_timing_audit": str(args.dataset_timing_audit.resolve()) if audit else None,
            "projector_config": str(projector_path.resolve()) if projector else None,
            "data_sources_sha256": {split: sha256(path) for split, path in data_sources.items()},
            "training_prompt_style": prompt_style,
            "splits": args.splits, "max_samples": args.max_samples,
            "full_official_split": args.max_samples == -1 and set(args.splits) <= {"test_s", "test_u"},
            "language_only": not args.run_sam, "save_masks": args.run_sam and args.save_masks,
            "mask_directory": str(Path(args.mask_dir).resolve()) if args.run_sam and args.save_masks else None,
            "mask_saving_exception": "language-only diagnostic" if not args.run_sam else None,
            "per_frame_metrics": args.run_sam,
            "model_timeline_fps": projector.get("model_timeline_fps"),
            "dataset_nominal_fps": nominal_fps,
            "exact_dataset_frame_timestamps_verified": False,
            "media_limits": {k: os.environ.get(k) for k in
                             ("FPS_MAX_FRAMES", "VIDEO_MAX_TOKEN_NUM", "IMAGE_MAX_TOKEN_NUM")},
            "training_settings_matched": expected,
            "sam_frames": projector.get("sam_frames"),
            "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                         "max_new_tokens": args.max_new_tokens},
            "inference_differences": [
                "Free greedy generation versus supervised teacher-forced SFT"
            ] if args.temperature == 0 else ["Free sampled generation versus supervised teacher-forced SFT"],
            "opd_gain_claim": False,
        }
        (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    engine, template, device = _build_engine(args)
    engine.model.eval()
    summaries = [_evaluate_split(args, engine, template, device, split, output_dir, rank, world_size) for split in args.splits]
    if rank == 0:
        aggregate_summaries = [
            json.loads((output_dir / f"summary_{split}.json").read_text(encoding="utf-8"))
            for split in args.splits
        ]
        (output_dir / "summary.json").write_text(
            json.dumps(aggregate_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    _dist_barrier(world_size, args.dist_backend)
    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
