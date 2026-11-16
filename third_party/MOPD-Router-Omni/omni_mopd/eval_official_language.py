"""Official test_s/test_u *language/class* diagnostic for a merged native Thinker.

SAM2/masks are deliberately not evaluated here; do not present language
accuracy as segmentation IoU or a matched OPD gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from omni_mopd.data import OmniEncoder, audit_timing, video_preprocessing_for_run


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", required=True, choices=["test_s", "test_u"])
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--baseline-sft", action="store_true",
                   help="Evaluate the audited SFT4809 export with the identical native inference path")
    a = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo")
    root = Path(os.environ.get("AURORA_ROOT", str(Path(__file__).resolve().parents[3])))
    train = Path(os.environ.get("AURORA_MOPD_TRAIN_DIR", str(root / "outputs/train/clean_omni_mopd_8gpu")))
    sft_model = Path(os.environ.get("AURORA_SFT_EXPORT", str(root / "outputs/models/mopd_omni_reasoning/sft_export")))
    if a.baseline_sft:
        if a.checkpoint.resolve() != sft_model.resolve():
            raise ValueError("--baseline-sft requires the exact audited SFT Thinker export")
        export = json.loads((a.checkpoint / "omni_export_manifest.json").read_text())
        expected_adapter = Path(os.environ.get("AURORA_SFT_INITIALIZER", str(root / "outputs/train/sam2_merged_sft_3epoch_8gpu/checkpoint-4809")))
        if not export["complete"] or Path(export["adapter"]).resolve() != expected_adapter.resolve():
            raise ValueError("Unverified SFT initializer export")
        source_step = 0
        source_checkpoint = export["adapter"]
        trainable_tensors_changed = 0
        source_settings = expected_adapter.parent / "training_manifest.json"
        video_preprocessing = "swift_training_parity"
    else:
        snapshot = json.loads((a.checkpoint / "fsdp_snapshot_manifest.json").read_text())
        if not snapshot["complete"] or snapshot["source_step"] not in {500, 600, 700, 800, 900, 1000}:
            raise ValueError("Unverified/missing merged FSDP snapshot")
        source_checkpoint = Path(snapshot["source_checkpoint"])
        if source_checkpoint.parent.parent != train:
            raise ValueError("Export does not derive from the selected real training run")
        if not snapshot["all_frozen_tensors_exact_after_dtype_cast"]:
            raise ValueError("Frozen weights are not initializer-matched")
        if Path(snapshot["initializer"]).resolve() != sft_model.resolve():
            raise ValueError("Wrong language/SEG initializer")
        source_step = snapshot["source_step"]
        trainable_tensors_changed = snapshot["trainable_language_tensors_changed"]
        source_settings = train / "global_step_500/actor/huggingface/config.json"
        video_preprocessing = video_preprocessing_for_run(train / "omni_training_manifest.json")
    from transformers import (
        Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from swift_plugin.sam2_grpo_rewards import (
        class_exact_score, extract_target_class, class_name_from_sample_id,
    )

    audited = audit_timing(root / "outputs/artifacts/refavs_fps_audit_20261025/audit.json")
    expected_source = root / "outputs/data/refavs_official_template_matched_20261102" / f"refavs_{a.split}.jsonl"
    if a.source.resolve() != expected_source.resolve():
        raise ValueError("Only official test_s/test_u manifests are allowed")
    expected_rows = {"test_s": 2288, "test_u": 1656}[a.split]
    rows = [json.loads(line) for line in a.source.open() if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError("Official split changed")
    if a.max_samples > 0:
        rows = rows[:a.max_samples]
    if rank == 0:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("Refuse overwriting existing evaluation output")
        a.output.mkdir(parents=True, exist_ok=False)
        sha_source = sha(a.source)
        prior = root / "outputs/eval/merged_sft_official_20261102_merged_v4_full/epoch3" / a.split / "evaluation_manifest.json"
        expected = json.loads(prior.read_text())
        if expected["source_sha256"] != sha_source:
            raise ValueError("Official test manifest differs from audited SFT source")
        settings = {
            "schema": 1, "split": a.split, "samples": len(rows),
            "full_official_split": len(rows) == expected_rows,
            "checkpoint": str(a.checkpoint.resolve()),
            "checkpoint_step": source_step,
            "checkpoint_source": str(source_checkpoint),
            "trainable_tensors_changed": trainable_tensors_changed,
            "baseline_sft": a.baseline_sft,
            "training_settings_source": str(source_settings),
            "video_preprocessing": video_preprocessing,
            "video_preprocessing_source": str(
                source_settings if a.baseline_sft else train / "omni_training_manifest.json"),
            "training_plan": None if a.baseline_sft else str(
                root / "outputs/automation/clean_omni_mopd_20261105_v1/plan.json"),
            "source": str(a.source.resolve()), "source_sha256": sha_source,
            "dataset_audit": audited,
            "settings": {
                "world_size": world, "model_timeline_fps": 1.0,
                "sam_training": False, "sam_evaluation": False, "save_masks": False,
                "mask_saving_exception": "Explicit language-only diagnostic, no SAM segmentation forward",
                "audio": f"original {a.split} waveform", "stored_frames": 10,
                "video_preprocessing": video_preprocessing,
                "max_pixels": 200704, "fps_max_frames": 768,
                "video_max_token_num": 768, "image_max_token_num": 16384,
                "max_prompt_length": 4864, "max_length": 5120,
                "training_response_length": 256, "evaluation_max_new_tokens": 256,
                "dtype": "bfloat16", "attention": "sdpa",
                "template": "omni_mopd.data.render_prompt", "new_special_tokens": ["[SEG]"],
                "decoding": "greedy",
            },
            "intended_inference_differences": [
                "greedy free-generation rather than sampled top_p=1 training rollout",
                "class diagnostic only; frozen trained projector/SAM2 not reattached",
                "no Teacher/optimizer or KL routing during evaluation",
            ] + (["BF16-merged native SFT export versus original FP16 SFT-trained LoRA"]
                 if a.baseline_sft else []),
            "no_opd_matched_control": False, "opd_gain_claim": False,
        }
        (a.output / "evaluation_manifest.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
        )
    if world > 1:
        dist.barrier()
    processor = Qwen2_5OmniProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        a.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    if processor.tokenizer.convert_tokens_to_ids("[SEG]") >= model.get_input_embeddings().num_embeddings:
        raise ValueError("SEG token ID out of trained vocabulary")
    encoder = OmniEncoder(processor, model.config, fps=1., max_pixels=200704,
                          video_preprocessing=video_preprocessing)
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad = processor.tokenizer.pad_token_id
    scores = {"samples": 0, "class_correct": 0, "seg_emitted": 0, "errors": 0}
    started = time.monotonic()
    output_file = a.output / f"{a.split}.rank{rank}.jsonl"
    with output_file.open("w", buffering=1) as f:
        for index in range(rank, len(rows), world):
            row = rows[index]
            with torch.inference_mode():
                batch, positions, _ = encoder.encode(row)
                if batch["input_ids"].shape[1] > 4864:
                    raise ValueError("Prompt exceeds trained max length; refuse truncation")
                batch = {
                    k: v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
                    for k, v in batch.items()
                }
                generated = model.generate(
                    **batch, max_new_tokens=256, do_sample=False, use_audio_in_video=False,
                    eos_token_id=eos, pad_token_id=pad,
                )
            output_ids = generated[0, batch["input_ids"].shape[1]:].detach().cpu().tolist()
            response = processor.tokenizer.decode(
                output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            category = str(row.get("class_name") or "").strip() or class_name_from_sample_id(row["sample_id"])
            class_correct = bool(class_exact_score(response, category, row["sample_id"]))
            seg = response.count("[SEG]")
            record = {
                "split": a.split, "case_index": index, "sample_id": row["sample_id"],
                "modality": row["template_match"]["template_reference"],
                "class_name": category, "response": response,
                "predicted_class": extract_target_class(response),
                "class_correct": class_correct, "response_token_count": len(output_ids),
                "seg_count": seg, "seg_emitted": seg == 1,
                "frame_ious": None, "frame_dices": None, "mask_paths": [],
                "scope": "language-only; SAM2 never run",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            scores["samples"] += 1
            scores["class_correct"] += int(class_correct)
            scores["seg_emitted"] += int(seg == 1)
            if scores["samples"] % 50 == 0:
                print(f"[{a.split} rank{rank}] {scores['samples']}/{len(rows[rank::world])} "
                      f"class={scores['class_correct']/scores['samples']:.3f} "
                      f"SEG={scores['seg_emitted']/scores['samples']:.3f}", flush=True)
    scores["runtime_sec"] = time.monotonic() - started
    (a.output / f"summary_{a.split}.rank{rank}.json").write_text(json.dumps(scores, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    if rank == 0:
        summary = {
            "split": a.split, "checkpoint_step": source_step,
            "full_official_split": len(rows) == expected_rows,
            "language_only": True, "mask_iou": None,
            **{k: sum(json.loads((a.output / f"summary_{a.split}.rank{r}.json").read_text())[k]
                       for r in range(world))
               for k in ("samples", "class_correct", "seg_emitted", "errors")},
        }
        if summary["samples"] != len(rows) or summary["errors"]:
            raise RuntimeError("Incomplete official language diagnostic")
        summary["class_accuracy"] = summary["class_correct"] / summary["samples"]
        summary["seg_rate"] = summary["seg_emitted"] / summary["samples"]
        merged = []
        for r in range(world):
            merged.extend(json.loads(line) for line in (
                a.output / f"{a.split}.rank{r}.jsonl"
            ).open() if line.strip())
        merged.sort(key=lambda record: record["case_index"])
        if [r["case_index"] for r in merged] != list(range(len(rows))):
            raise RuntimeError("Missing/duplicated official examples")
        (a.output / f"{a.split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in merged)
        )
        (a.output / f"summary_{a.split}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
