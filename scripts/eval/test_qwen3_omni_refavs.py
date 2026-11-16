#!/usr/bin/env python3
"""Evaluate Qwen3-Omni AURORA joint SFT on REF-AVS.

Reports generation, [SEG], and teacher-forced SAM mask IoU/Dice metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tempfile
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_adapter_for_eval(checkpoint: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Make a PEFT-compatible view of checkpoints saved by ms-swift training.

    The training checkpoint stores a regex string in ``target_modules``.
    Current PEFT treats a string as a collection of exact module suffixes in
    this loading path, so it can reject that regex before reading any LoRA
    weights.  Derive the exact module paths from the saved ``lora_A`` keys and
    disable PEFT's broad ``modules_to_save`` matching; those saved tensors are
    restored explicitly after the adapter is attached.
    """
    config_path = checkpoint / "adapter_config.json"
    state_path = checkpoint / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, str):
        return checkpoint, None
    from safetensors import safe_open

    lora_modules: set[str] = set()
    with safe_open(str(state_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            marker = ".lora_A.weight"
            if not key.endswith(marker) or not key.startswith("base_model.model."):
                continue
            lora_modules.add(key[len("base_model.model.") : -len(marker)])
    if not lora_modules:
        return checkpoint, None
    expert_modules = {module for module in lora_modules if ".experts." in module}
    config["target_modules"] = sorted(module for module in lora_modules if ".experts." not in module)
    config["modules_to_save"] = []
    # PEFT represents Qwen3 MoE expert parameters with ParamWrapper, whose
    # inference loader rejects non-zero LoRA dropout.  Dropout is inactive in
    # this eval path, so normalize it only in the temporary config.
    config["lora_dropout"] = 0.0
    temp_dir = tempfile.TemporaryDirectory(prefix="aurora_qwen3_adapter_")
    compat_dir = Path(temp_dir.name)
    # Transformers 5 packs Qwen3-Omni routed experts into 3-D parameters,
    # while the ms-swift checkpoint stores one LoRA pair per expert.  Convert
    # the checkpoint into PEFT's target_parameters representation so every
    # trained expert adapter is retained during inference.
    if expert_modules:
        import re
        from safetensors.torch import save_file

        pattern = re.compile(
            r"^base_model\.model\.thinker\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(gate_proj|up_proj|down_proj)\.lora_([AB])\.weight$"
        )
        layers: dict[int, dict[tuple[int, str, str], torch.Tensor]] = {}
        converted_state: dict[str, torch.Tensor] = {}
        with safe_open(str(state_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                match = pattern.match(key)
                if match is None:
                    converted_state[key] = handle.get_tensor(key)
                    continue
                layer, expert, branch, side = match.groups()
                layers.setdefault(int(layer), {})[(int(expert), branch, side)] = handle.get_tensor(key)

        if not layers:
            raise RuntimeError("Found expert target modules but no per-expert LoRA tensors to convert")
        for layer, tensors in layers.items():
            experts = sorted({expert for expert, _, _ in tensors})
            branches = {branch for _, branch, _ in tensors}
            if not {"gate_proj", "up_proj", "down_proj"}.issubset(branches):
                raise RuntimeError(f"Incomplete Qwen3 expert LoRA branches in layer {layer}: {sorted(branches)}")
            gate_a = torch.stack([tensors[(expert, "gate_proj", "A")] for expert in experts])
            up_a = torch.stack([tensors[(expert, "up_proj", "A")] for expert in experts])
            down_a = torch.stack([tensors[(expert, "down_proj", "A")] for expert in experts])
            gate_b = torch.stack([tensors[(expert, "gate_proj", "B")] for expert in experts])
            up_b = torch.stack([tensors[(expert, "up_proj", "B")] for expert in experts])
            down_b = torch.stack([tensors[(expert, "down_proj", "B")] for expert in experts])
            num_experts, rank, hidden = gate_a.shape
            intermediate = gate_b.shape[1]
            if experts != list(range(num_experts)):
                raise RuntimeError(f"Non-contiguous expert ids in layer {layer}: {experts[:5]}...")

            # gate_up_proj delta: each expert keeps independent gate/up ranks.
            gate_up_a = torch.cat([gate_a, up_a], dim=1).reshape(num_experts * 2 * rank, hidden)
            gate_up_b = torch.zeros(
                (2 * intermediate, num_experts * 2 * rank), dtype=gate_b.dtype
            )
            for expert in range(num_experts):
                start = expert * 2 * rank
                gate_up_b[:intermediate, start:start + rank] = gate_b[expert]
                gate_up_b[intermediate:, start + rank:start + 2 * rank] = up_b[expert]

            # down_proj delta: flatten expert rows/columns in the order PEFT's
            # ParamWrapper expects for a 3-D packed parameter.
            down_a_packed = down_a.reshape(num_experts * rank, down_a.shape[-1])
            down_b_packed = down_b.permute(1, 0, 2).reshape(down_b.shape[1], num_experts * rank)
            prefix = f"base_model.model.thinker.model.layers.{layer}.mlp.experts"
            converted_state[f"{prefix}.base_layer.lora_A.weight"] = gate_up_a
            converted_state[f"{prefix}.base_layer.lora_B.weight"] = gate_up_b
            converted_state[f"{prefix}.lora_A.weight"] = down_a_packed
            converted_state[f"{prefix}.lora_B.weight"] = down_b_packed

        target_parameters = []
        for layer in sorted(layers):
            target_parameters.extend([
                f"thinker.model.layers.{layer}.mlp.experts.gate_up_proj",
                f"thinker.model.layers.{layer}.mlp.experts.down_proj",
            ])
        config["target_parameters"] = target_parameters
        config["rank_pattern"] = {"gate_up_proj": int(config["r"]) * 2}
        config["alpha_pattern"] = {"gate_up_proj": int(config["lora_alpha"]) * 2}
        save_file(converted_state, str(compat_dir / "adapter_model.safetensors"))
        print(
            f"[adapter] packed {len(layers)} Qwen3 expert layers and retained {len(converted_state)} tensors",
            flush=True,
        )
    else:
        # Avoid copying the multi-gigabyte tensor file when no packed experts
        # are present; Swift/PEFT accepts a local symlink.
        (compat_dir / "adapter_model.safetensors").symlink_to(state_path.resolve())
    (compat_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        f"[adapter] normalized {len(lora_modules)} LoRA target modules for {checkpoint} -> {compat_dir}",
        flush=True,
    )
    return compat_dir, temp_dir


def _restore_saved_modules(model, checkpoint: Path) -> None:
    """Restore PEFT ``modules_to_save`` tensors skipped by compatibility loading."""
    from safetensors import safe_open

    base = _unwrap_model(model)
    parameters = dict(base.named_parameters())
    restored = 0
    state_path = checkpoint / "adapter_model.safetensors"
    with safe_open(str(state_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_A." in key or ".lora_B." in key:
                continue
            prefix = "base_model.model."
            if not key.startswith(prefix):
                continue
            target_name = key[len(prefix) :]
            parameter = parameters.get(target_name)
            if parameter is None:
                continue
            tensor = handle.get_tensor(key)
            with torch.no_grad():
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            restored += 1
    print(f"[adapter] restored {restored} saved module tensors", flush=True)


def _load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows[:max_samples] if max_samples > 0 else rows


def _category(row: dict[str, Any]) -> str:
    for key in ("class_name", "category", "target_category"):
        value = str(row.get(key) or "").strip()
        if value and value.lower() not in {"null", "none", "unknown"}:
            return value
    parts = str(row.get("sample_id") or "").split("_")
    return parts[-2].strip() if len(parts) >= 2 else ""


def _category_mentioned(response: str, category: str) -> bool:
    if not category:
        return False
    response_norm = re.sub(r"[^a-z0-9]+", " ", response.lower()).strip()
    category_norm = re.sub(r"[^a-z0-9]+", " ", category.lower()).strip()
    return bool(category_norm and re.search(rf"\b{re.escape(category_norm)}\b", response_norm))


def _summary(split: str, records: list[dict[str, Any]], runtime_sec: float) -> dict[str, Any]:
    samples = len(records)
    errors = sum(record["status"] == "error" for record in records)
    ok = [record for record in records if record["status"] == "ok"]
    seg_emitted = sum(record.get("seg_count", 0) > 0 for record in ok)
    seg_once = sum(record.get("seg_count", 0) == 1 for record in ok)
    categories = [record for record in ok if record.get("target_category")]
    category_mentioned = sum(record.get("category_mentioned", False) for record in categories)
    mask_records = [(record.get("mask_metrics") or {}) for record in ok]
    scored_masks = [metrics for metrics in mask_records if "iou" in metrics]
    return {
        "split": split,
        "samples": samples,
        "errors": errors,
        "ok": len(ok),
        "seg_emitted": seg_emitted,
        "seg_once": seg_once,
        "seg_rate": seg_emitted / max(len(ok), 1),
        "seg_once_rate": seg_once / max(len(ok), 1),
        "category_samples": len(categories),
        "category_mentioned": category_mentioned,
        "category_mention_rate": category_mentioned / max(len(categories), 1),
        "seg_scored": sum(metrics.get("frames_scored", 0) for metrics in scored_masks),
        "mean_iou": (
            sum(metrics.get("iou", 0.0) for metrics in scored_masks) / max(len(scored_masks), 1)
        ),
        "mean_dice": (
            sum(metrics.get("dice", 0.0) for metrics in scored_masks) / max(len(scored_masks), 1)
        ),
        "mean_completion_tokens": (
            sum(record.get("completion_tokens", 0) for record in ok) / max(len(ok), 1)
        ),
        "runtime_sec": runtime_sec,
        "mask_metrics_available": True,
    }


def _unwrap_model(model):
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


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _frame_metrics(logits, target):
    pred = logits.sigmoid() >= 0.5
    target = target >= 0.5
    intersection = (pred & target).sum().item()
    union = (pred | target).sum().item()
    pred_sum = pred.sum().item()
    target_sum = target.sum().item()
    return (
        1.0 if union == 0 else intersection / union,
        1.0 if pred_sum + target_sum == 0 else 2.0 * intersection / (pred_sum + target_sum),
    )


def _aurora_mask_scores(engine, row, response):
    """Teacher-force one generated answer through the joint SAM branch."""
    from swift_plugin.segmentation import load_gt_masks, preprocess_sam_frames

    template = engine.template
    base = _unwrap_model(engine.model)
    device = next(base.thinker.model.norm.parameters()).device
    # Modules attached after device_map=auto are initially on CPU.  Keep the
    # projector, SAM, and hidden state on the same final-thinker device.
    base.text_hidden_fcs.to(device)
    base.sam.to(device)
    full_row = copy.deepcopy(row)
    full_row["messages"] = row["messages"][:1] + [{"role": "assistant", "content": response}]
    template.set_mode("train")
    try:
        encoded = template.encode(full_row)
        encoded.pop("_extra_kwargs", None)
        batch = template.data_collator([encoded])
        template.register_post_encode_hook([engine.model])
        # Some ms-swift data collators retain only native model fields.  The
        # AURORA supervision tensors are already prepared by ``encode``;
        # explicitly carry them through for this one teacher-force call.
        for key in ("sam_pixel_values", "gt_masks", "sam_resize_sizes", "sam_original_sizes"):
            if key not in batch and key in encoded:
                batch[key] = encoded[key]
        if "sam_pixel_values" in batch and torch.is_tensor(batch["sam_pixel_values"]):
            if batch["sam_pixel_values"].ndim == 4:
                batch["sam_pixel_values"] = batch["sam_pixel_values"].unsqueeze(0)
        if "gt_masks" in batch and torch.is_tensor(batch["gt_masks"]):
            batch["gt_masks"] = [batch["gt_masks"]]
        for key in ("sam_resize_sizes", "sam_original_sizes"):
            if key in batch and batch[key] and isinstance(batch[key][0], tuple):
                batch[key] = [batch[key]]
        if batch.get("sam_pixel_values") is None:
            frame_paths = row.get("sam_frame_paths") or row.get("videos", [[]])[0]
            mask_paths = row.get("mask_paths")
            pixels, resize_sizes, original_sizes = preprocess_sam_frames(
                frame_paths, int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
            )
            batch["sam_pixel_values"] = pixels.unsqueeze(0)
            batch["gt_masks"] = [load_gt_masks(mask_paths, original_sizes[0], pixels.shape[0])]
            batch["sam_resize_sizes"] = [resize_sizes]
            batch["sam_original_sizes"] = [original_sizes]
        if "labels" not in batch:
            # The generated answer is already in input_ids.  A sparse label
            # tensor is sufficient for the [SEG] hidden-state branch and
            # avoids depending on collator-specific label retention.
            labels = batch["input_ids"].clone()
            seg_id = engine.template.tokenizer.convert_tokens_to_ids("[SEG]")
            labels[labels != seg_id] = -100
            batch["labels"] = labels
        if os.environ.get("AURORA_DEBUG", "0") == "1":
            print(
                "[AURORA] teacher batch: "
                f"keys={sorted(batch.keys())} labels={batch.get('labels') is not None} "
                f"sam_pixel_values={batch.get('sam_pixel_values') is not None} "
                f"gt_masks={batch.get('gt_masks') is not None}",
                flush=True,
            )
        if batch.get("video_second_per_grid") is None and batch.get("video_grid_thw") is not None:
            grid = batch["video_grid_thw"]
            num_videos = int(grid.shape[0]) if torch.is_tensor(grid) else len(grid)
            video_processor = getattr(template.processor, "video_processor", None)
            temporal_patch_size = getattr(video_processor, "temporal_patch_size", 2)
            value = float(temporal_patch_size) / float(os.environ.get("AURORA_VIDEO_FPS", "1.0"))
            batch["video_second_per_grid"] = [torch.tensor(value) for _ in range(num_videos)]
        batch = _to_device(batch, device)
        if isinstance(batch.get("labels"), list):
            batch["labels"] = torch.stack(batch["labels"])
        if torch.is_tensor(batch.get("labels")):
            batch["labels"] = batch["labels"].to(device)
        with torch.no_grad():
            engine.model(**batch)
        predictions = getattr(base, "aurora_last_pred_masks", None)
        targets = batch.get("gt_masks")
        if not predictions or targets is None:
            raise RuntimeError(
                "joint forward produced no SAM masks: "
                f"batch_keys={sorted(batch.keys())} "
                f"labels={batch.get('labels') is not None} "
                f"sam_pixel_values={batch.get('sam_pixel_values') is not None} "
                f"targets={targets is not None}"
            )
        ious, dices = [], []
        for prediction, target in zip(predictions, targets):
            target = target.to(device=prediction.device, dtype=prediction.dtype)
            for frame_prediction, frame_target in zip(prediction, target):
                iou, dice = _frame_metrics(frame_prediction, frame_target)
                ious.append(iou)
                dices.append(dice)
        return {
            "iou": sum(ious) / max(len(ious), 1),
            "dice": sum(dices) / max(len(dices), 1),
            "frames_scored": len(ious),
        }
    finally:
        template.set_mode("transformers")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=_root() / "outputs/data/refavs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test_s", "test_u"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-impl", default="flash_attn")
    parser.add_argument("--max-memory-per-gpu", default="52GiB")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    for required in (args.model, args.checkpoint, args.data_dir):
        if not required.exists():
            parser.error(f"required path does not exist: {required}")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.checkpoint / name).is_file():
            parser.error(f"checkpoint is incomplete: missing {args.checkpoint / name}")

    os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "0")
    os.environ.setdefault("USE_AUDIO_IN_VIDEO", "0")
    os.environ.setdefault("FPS", "1.0")
    os.environ.setdefault("FPS_MAX_FRAMES", "10")
    os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "128")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
    os.environ.setdefault("AURORA_SAM_CHECKPOINT", "models/sam_vit_h_4b8939.pth")
    os.environ.setdefault("AURORA_NUM_SAM_FRAMES", "10")
    os.environ.setdefault("AURORA_VIDEO_FPS", "1.0")

    import swift_plugin.aurora_qwen3_omni  # noqa: F401
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 1:
        raise RuntimeError("Qwen3-Omni-30B evaluation requires at least one CUDA GPU")
    max_memory = {index: args.max_memory_per_gpu for index in range(visible_gpus)}
    print(
        f"[load] model={args.model} checkpoint={args.checkpoint} "
        f"visible_gpus={visible_gpus} max_memory={max_memory}",
        flush=True,
    )
    adapter_path, adapter_tmp = _prepare_adapter_for_eval(args.checkpoint)
    engine = TransformersEngine(
        str(args.model),
        adapters=[str(adapter_path)],
        torch_dtype=torch.bfloat16,
        model_type="aurora_qwen3_omni",
        template_type="aurora_qwen3_omni",
        new_special_tokens=["[SEG]"],
        attn_impl=args.attn_impl,
        device_map="auto",
        max_memory=max_memory,
        download_model=False,
        max_batch_size=1,
    )
    if adapter_tmp is not None:
        engine._aurora_adapter_tmp = adapter_tmp
        _restore_saved_modules(engine.model, args.checkpoint)
    engine.model.eval()
    tokenizer = engine.template.tokenizer
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
    if seg_id is None or seg_id == tokenizer.unk_token_id:
        raise RuntimeError("[SEG] was not registered as a special token")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in args.splits:
        source = args.data_dir / f"refavs_{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = _load_rows(source, args.max_samples)
        records: list[dict[str, Any]] = []
        output = args.output_dir / f"{split}.jsonl"
        started = time.time()
        with output.open("w", encoding="utf-8") as handle:
            for case_index, row in enumerate(rows):
                category = _category(row)
                try:
                    result = engine.infer(
                        [InferRequest(
                            messages=copy.deepcopy(row["messages"][:1]),
                            videos=row.get("videos"),
                            audios=row.get("audios"),
                        )],
                        RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0),
                        use_tqdm=False,
                    )[0]
                    response = result.choices[0].message.content or ""
                    token_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                    mask_metrics = None
                    if response.count("[SEG]") == 1:
                        try:
                            mask_metrics = _aurora_mask_scores(engine, row, response)
                        except Exception as mask_exc:
                            mask_metrics = {
                                "error": repr(mask_exc),
                                "traceback": traceback.format_exc(),
                            }
                    record = {
                        "split": split,
                        "case_index": case_index,
                        "sample_id": row.get("sample_id"),
                        "target_category": category,
                        "prompt": row["messages"][0].get("content", ""),
                        "rollout": response,
                        "reference": row["messages"][-1].get("content", ""),
                        "seg_count": sum(token_id == seg_id for token_id in token_ids),
                        "completion_tokens": len(token_ids),
                        "category_mentioned": _category_mentioned(response, category),
                        "mask_metrics": mask_metrics,
                        "status": "ok",
                    }
                except Exception as exc:
                    record = {
                        "split": split,
                        "case_index": case_index,
                        "sample_id": row.get("sample_id"),
                        "target_category": category,
                        "status": "error",
                        "error": repr(exc),
                    }
                    if args.fail_fast:
                        raise
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if (case_index + 1) % args.log_every == 0 or case_index + 1 == len(rows):
                    current = _summary(split, records, time.time() - started)
                    print(
                        f"[{split}] {case_index + 1}/{len(rows)} err={current['errors']} "
                        f"seg%={current['seg_rate']:.1%} "
                        f"category%={current['category_mention_rate']:.1%}",
                        flush=True,
                    )
        summary = _summary(split, records, time.time() - started)
        summary["output"] = str(output)
        (args.output_dir / f"summary_{split}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summaries.append(summary)

    provenance = {
        "model": str(args.model.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(args.checkpoint.name.rsplit("-", 1)[-1]),
        "max_new_tokens": args.max_new_tokens,
        "max_samples": args.max_samples,
        "visible_gpus": visible_gpus,
        "mask_metrics_available": True,
        "sam_checkpoint": os.environ.get("AURORA_SAM_CHECKPOINT"),
        "note": "Qwen3-Omni AURORA joint checkpoint with trained text_hidden_fcs and SAM mask_decoder.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps({"provenance": provenance, "splits": summaries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"provenance": provenance, "splits": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
