#!/usr/bin/env python3
"""Diagnostic ONLY: decode native Thinker under historical and corrected media.

The historical MOPD checkpoint saw capped video frames during training. It
must not be evaluated under corrected media and called a matched test gain.
No segmentation branch is loaded or scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/MOPD-Router-Omni"))
from omni_mopd.data import (  # noqa: E402
    LEGACY_VIDEO_ENCODER_SHA256, OmniEncoder, video_preprocessing_for_run,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.max_new_tokens < 1 or args.max_new_tokens > 256:
        raise ValueError("Diagnostic max_new_tokens must be within trained response limit")
    os.environ.update(FPS="1.0", AURORA_VIDEO_FPS="1.0",
                      FPS_MAX_FRAMES="768", VIDEO_MAX_TOKEN_NUM="768",
                      IMAGE_MAX_TOKEN_NUM="16384", ENABLE_AUDIO_OUTPUT="0",
                      USE_AUDIO_IN_VIDEO="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    with args.source.open() as stream:
        row = next((json.loads(line) for i, line in enumerate(stream) if i == args.case_index), None)
    if row is None:
        raise IndexError(args.case_index)
    snapshot = json.loads((args.checkpoint / "fsdp_snapshot_manifest.json").read_text())
    if not snapshot["complete"] or snapshot["source_step"] != 500:
        raise ValueError("Not the verified historical native MOPD step500 export")
    train_manifest = Path(snapshot["source_checkpoint"]).parents[1] / "omni_training_manifest.json"
    prior = json.loads(train_manifest.read_text())
    expected_legacy_sha = LEGACY_VIDEO_ENCODER_SHA256
    if prior["omni_adapter_source_sha256"]["omni_mopd/data.py"] != expected_legacy_sha:
        raise ValueError("Historical training did not use the audited legacy encoder")
    if video_preprocessing_for_run(train_manifest) != "historical_native_capped":
        raise ValueError("Expected historical native video preparation for step500")
    device = torch.device("cuda:0")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    variants = {}
    for label, video_preprocessing in (
        ("historical_training_media", "historical_native_capped"),
        ("corrected_sft_matched_media", "swift_training_parity"),
    ):
        encoder = OmniEncoder(
            processor, model.config, fps=1.0, max_pixels=200704,
            video_preprocessing=video_preprocessing,
        )
        batch, positions, _ = encoder.encode(row)
        prompt_len = batch["input_ids"].shape[1]
        batch = {k: v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
                 for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
                **batch, max_new_tokens=args.max_new_tokens, do_sample=False,
                use_audio_in_video=False, eos_token_id=eos,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        ids = generated[0, prompt_len:].tolist()
        text = processor.tokenizer.decode(ids, skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
        variants[label] = {
            "prompt_tokens": prompt_len, "video_grid": batch["video_grid_thw"].tolist(),
            "video_second_per_grid": batch["video_second_per_grid"].tolist(),
            "video_preprocessing": video_preprocessing,
            "response_tokens": len(ids), "response": text, "seg_count": text.count("[SEG]"),
            "max_response_tokens": args.max_new_tokens,
        }
    result = {
        "scope": "one-sample language-only diagnostic; no SAM masks/no model-benefit claim",
        "sample_id": row["sample_id"], "split": "test_s",
        "checkpoint": str(args.checkpoint.resolve()),
        "training_manifest": str(train_manifest.resolve()),
        "historical_encoder_sha256": expected_legacy_sha,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "inference_differences": [
            "Greedy versus sampled historical training rollout",
            "Corrected input variant is NOT the step500 checkpoint's training preprocessing",
            "No segmentation/projector/SAM2 was loaded",
        ],
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
