"""Official test_s/test_u *language/class* diagnostic for a merged native Thinker.

SAM2/masks are deliberately not evaluated here; do not present language
accuracy as segmentation IoU or a matched OPD gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _add_import_paths() -> None:
    mopd_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (mopd_root, repo_root, repo_root / "third_party/sam2"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_add_import_paths()

from omni_mopd.data import OmniEncoder, audit_timing, video_preprocessing_for_run

OFFICIAL_SPLIT_COUNTS = {"test_s": 2288, "test_u": 1656}
OFFICIAL_SPLIT_SHA256 = {
    "test_s": "a75ff4b056284d3728e4d392929a8f8e9d1ab338339fb41159b168197da46eb",
    "test_u": "ee342befc29b83f8a448a175687f42788b1e9f685bab09609351b028d6dccf25",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_timing_audit(timing_path: Path) -> dict[str, Any]:
    """Validate dataset timing audit JSON (original full audit or omni_eval_settings output)."""
    timing_path = Path(timing_path)
    if not timing_path.is_file():
        raise FileNotFoundError(f"Dataset timing audit not found: {timing_path}")
    audit = json.loads(timing_path.read_text(encoding="utf-8"))
    if "distribution" in audit:
        return audit_timing(timing_path)
    if float(audit.get("nominal_fps", 0)) != 1.0 or int(audit.get("unique_clips_checked", 0)) < 1:
        raise ValueError(f"Unrecognized media timing audit: {timing_path}")
    return {
        "nominal_frame_density_fps": 1.0,
        "model_timeline_fps": 1.0,
        "stored_frames": 10,
        "exact_sampling_instants_verified": False,
        "audit_source": str(timing_path.resolve()),
        "unique_clips_checked": int(audit["unique_clips_checked"]),
    }


def _extract_manifest_source_sha(manifest: dict[str, Any], split: str) -> str | None:
    if manifest.get("source_sha256"):
        return str(manifest["source_sha256"])
    by_split = manifest.get("data_sources_sha256")
    if isinstance(by_split, dict) and by_split.get(split):
        return str(by_split[split])
    return None


def verify_official_split_source(
    source: Path,
    split: str,
    *,
    root: Path,
    expected_source: Path | None = None,
    reference_manifest: Path | None = None,
    media_manifest: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an official test_s/test_u JSONL and verify split count + provenance hashes."""
    if split not in OFFICIAL_SPLIT_COUNTS:
        raise ValueError(f"Unsupported official split: {split}")
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Official split file not found: {source}")
    if expected_source is None and os.environ.get("AURORA_OFFICIAL_DATA_DIR"):
        expected_source = Path(os.environ["AURORA_OFFICIAL_DATA_DIR"]) / f"refavs_{split}.jsonl"
    if expected_source is not None and source.resolve() != Path(expected_source).resolve():
        raise ValueError(
            f"Source path {source.resolve()} does not match expected official split {Path(expected_source).resolve()}"
        )
    expected_rows = OFFICIAL_SPLIT_COUNTS[split]
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"Official {split} row count mismatch: expected {expected_rows}, got {len(rows)}")
    for row in rows:
        if len((row.get("videos") or [[]])[0]) != 10 or len(row.get("audios") or []) != 1:
            raise ValueError(f"Unexpected frame/audio count in {source} row {row.get('sample_id')!r}")

    sha_source = sha(source)
    explicit_ref = reference_manifest or (
        Path(os.environ["AURORA_REFERENCE_EVAL_MANIFEST"])
        if os.environ.get("AURORA_REFERENCE_EVAL_MANIFEST")
        else None
    )
    default_prior = root / "outputs/eval/merged_sft_official_20261102_merged_v4_full/epoch3" / split / "evaluation_manifest.json"
    prior_path = explicit_ref if explicit_ref is not None else (default_prior if default_prior.is_file() else None)
    if explicit_ref is not None and not Path(explicit_ref).is_file():
        raise FileNotFoundError(f"Reference evaluation manifest not found: {explicit_ref}")
    prior_sha: str | None = None
    if prior_path is not None:
        prior_sha = _extract_manifest_source_sha(
            json.loads(Path(prior_path).read_text(encoding="utf-8")), split
        )

    if media_manifest is not None:
        media_manifest_path = Path(media_manifest)
        if not media_manifest_path.is_file():
            raise FileNotFoundError(f"Relocation manifest not found: {media_manifest_path}")
    else:
        candidate = source.with_suffix(".media_manifest.json")
        media_manifest_path = candidate if candidate.is_file() else None

    official_sha = OFFICIAL_SPLIT_SHA256[split]
    if media_manifest_path is not None:
        reloc = json.loads(media_manifest_path.read_text(encoding="utf-8"))
        if reloc.get("purpose") != "REF-AVS media path relocation only" or not reloc.get("label_and_message_contents_unchanged"):
            raise ValueError(f"Unverified relocation manifest: {media_manifest_path}")
        if reloc.get("check_files") is not True:
            raise ValueError(f"Relocation manifest must have check_files=true: {media_manifest_path}")
        if int(reloc.get("rows", -1)) != expected_rows:
            raise ValueError(f"Relocation manifest row count mismatch in {media_manifest_path}")
        if reloc.get("output_sha256") != sha_source:
            raise ValueError(f"Relocated JSONL SHA256 differs from {media_manifest_path}")
        orig_sha = reloc.get("source_sha256")
        allowed_orig = {official_sha}
        if prior_sha is not None:
            allowed_orig.add(prior_sha)
        if orig_sha not in allowed_orig:
            raise ValueError(f"Relocated JSONL original source_sha256={orig_sha!r} is not an audited official {split} split")
        provenance = {
            "source_sha256": sha_source,
            "original_source_sha256": orig_sha,
            "media_manifest": str(media_manifest_path.resolve()),
            "reference_manifest": str(Path(prior_path).resolve()) if prior_path else None,
        }
        return rows, provenance

    expected_sha = prior_sha if prior_sha is not None else official_sha
    if sha_source != expected_sha:
        raise ValueError(
            f"Official {split} manifest differs from audited source (got {sha_source}, expected {expected_sha}). "
            "If media paths were relocated with scripts/data/relocate_media_paths.py, keep its "
            ".media_manifest.json sidecar alongside the JSONL or pass --reference-manifest."
        )
    provenance = {
        "source_sha256": sha_source,
        "original_source_sha256": sha_source,
        "media_manifest": None,
        "reference_manifest": str(Path(prior_path).resolve()) if prior_path else None,
    }
    return rows, provenance


def verify_fsdp_snapshot(
    checkpoint: Path,
    train_dir: Path,
    *,
    sft_export: Path | None = None,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Validate a merged FSDP snapshot manifest against its training run."""
    snapshot = json.loads((Path(checkpoint) / "fsdp_snapshot_manifest.json").read_text(encoding="utf-8"))
    source_step = snapshot.get("source_step")
    if (
        not snapshot.get("complete")
        or not isinstance(source_step, int)
        or isinstance(source_step, bool)
        or source_step <= 0
    ):
        raise ValueError("Unverified/missing merged FSDP snapshot")
    if expected_step is not None and source_step != expected_step:
        raise ValueError(f"Snapshot step mismatch: got {source_step}, expected {expected_step}")
    source_checkpoint = Path(snapshot["source_checkpoint"])
    if (
        source_checkpoint.resolve().parent.parent != Path(train_dir).resolve()
        or source_checkpoint.parent.name != f"global_step_{source_step}"
    ):
        raise ValueError("Export does not derive from the selected training run")
    if not snapshot.get("all_frozen_tensors_exact_after_dtype_cast"):
        raise ValueError("Frozen weights are not initializer-matched")
    if sft_export is not None and Path(snapshot["initializer"]).resolve() != Path(sft_export).resolve():
        raise ValueError("Wrong language/SEG initializer")
    source_settings = Path(train_dir) / f"global_step_{source_step}/actor/huggingface/config.json"
    if not source_settings.is_file():
        raise FileNotFoundError(f"Missing step config for FSDP snapshot: {source_settings}")
    return {
        "snapshot": snapshot,
        "source_step": source_step,
        "source_checkpoint": source_checkpoint,
        "source_settings": source_settings,
        "video_preprocessing": video_preprocessing_for_run(Path(train_dir) / "omni_training_manifest.json"),
    }


def main():
    root = Path(os.environ.get("AURORA_ROOT") or str(Path(__file__).resolve().parents[3]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", required=True, choices=["test_s", "test_u"])
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--baseline-sft", action="store_true",
                   help="Evaluate the audited SFT export with the identical native inference path")
    p.add_argument("--train-dir", type=Path,
                   default=Path(os.environ.get("AURORA_MOPD_TRAIN_DIR") or str(root / "outputs/train/clean_omni_mopd_8gpu")))
    p.add_argument("--sft-export", type=Path,
                   default=Path(os.environ.get("AURORA_SFT_EXPORT") or str(root / "outputs/models/mopd_omni_reasoning/sft_export")))
    p.add_argument("--sft-initializer", type=Path,
                   default=Path(os.environ.get("AURORA_SFT_INITIALIZER") or str(root / "outputs/train/sam2_merged_sft_3epoch_8gpu/checkpoint-4809")))
    p.add_argument("--timing-audit", type=Path,
                   default=Path(os.environ.get("OMNI_TIMING_AUDIT") or str(root / "outputs/artifacts/refavs_fps_audit_20261025/audit.json")))
    p.add_argument("--expected-source", type=Path, default=None)
    p.add_argument("--reference-manifest", type=Path, default=None)
    p.add_argument("--media-manifest", type=Path, default=None)
    p.add_argument("--expected-step", type=int, default=None)
    a = p.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo")
    train = a.train_dir
    sft_model = a.sft_export
    if a.baseline_sft:
        if a.checkpoint.resolve() != sft_model.resolve():
            raise ValueError("--baseline-sft requires the exact audited SFT Thinker export")
        export = json.loads((a.checkpoint / "omni_export_manifest.json").read_text())
        expected_adapter = a.sft_initializer
        if not export["complete"] or Path(export["adapter"]).resolve() != expected_adapter.resolve():
            raise ValueError("Unverified SFT initializer export")
        source_step = 0
        source_checkpoint = export["adapter"]
        trainable_tensors_changed = 0
        source_settings = expected_adapter.parent / "training_manifest.json"
        video_preprocessing = "swift_training_parity"
    else:
        verified = verify_fsdp_snapshot(
            a.checkpoint, train, sft_export=sft_model, expected_step=a.expected_step
        )
        snapshot = verified["snapshot"]
        source_checkpoint = verified["source_checkpoint"]
        source_step = verified["source_step"]
        trainable_tensors_changed = snapshot["trainable_language_tensors_changed"]
        source_settings = verified["source_settings"]
        video_preprocessing = verified["video_preprocessing"]
    from transformers import (
        Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from swift_plugin.sam2_grpo_rewards import (
        class_exact_score, extract_target_class, class_name_from_sample_id,
    )

    audited = verify_timing_audit(a.timing_audit)
    rows, source_provenance = verify_official_split_source(
        a.source,
        a.split,
        root=root,
        expected_source=a.expected_source,
        reference_manifest=a.reference_manifest,
        media_manifest=a.media_manifest,
    )
    expected_rows = OFFICIAL_SPLIT_COUNTS[a.split]
    if a.max_samples > 0:
        rows = rows[:a.max_samples]
    if rank == 0:
        if a.output.exists() and any(a.output.iterdir()):
            raise ValueError("Refuse overwriting existing evaluation output")
        a.output.mkdir(parents=True, exist_ok=False)
        plan_path = os.environ.get("AURORA_MOPD_PLAN")
        if not plan_path:
            candidate_plan = root / "outputs/automation/clean_omni_mopd_20261105_v1/plan.json"
            if candidate_plan.is_file():
                plan_path = str(candidate_plan)
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
            "training_plan": None if a.baseline_sft else plan_path,
            "source": str(a.source.resolve()),
            **source_provenance,
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
