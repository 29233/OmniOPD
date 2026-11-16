"""Small SFT-native vs Swift-SFT mask sanity check; NOT a gain evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from omni_mopd.data import OmniEncoder
from omni_mopd.eval_official_sam2 import (
    INITIALIZER, ROOT, frame_metric, projector_and_sam,
)
from omni_mopd.modeling import extend_positions
from swift_plugin.sam2_segmentation import preprocess_sam2_frames
from swift_plugin.segmentation import load_gt_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1 or args.count > 16:
        raise ValueError("Sanity check restricted to <=16 official test_s examples")
    model_path = ROOT / "outputs/models/mopd_omni_reasoning/sft4809_v2"
    if not (model_path / "omni_export_manifest.json").exists():
        raise FileNotFoundError("Missing trained-SFT native export")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    projector, sam = projector_and_sam(device)
    encoder = OmniEncoder(processor, model.config, fps=1., max_pixels=200704)
    source = ROOT / "outputs/data/refavs_official_template_matched_20261102/refavs_test_s.jsonl"
    baseline = ROOT / "outputs/eval/mask_primary_sft_control_20261104_v1/test_s/test_s.jsonl"
    rows = [json.loads(l) for _, l in zip(range(args.count), source.open())]
    reference = [json.loads(l) for _, l in zip(range(args.count), baseline.open())]
    results = []
    seg_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for index, row in enumerate(rows):
        with torch.inference_mode():
            encoded, positions, _ = encoder.encode(row)
            encoded = {
                k: v.to(device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(device)
                for k, v in encoded.items()
            }
            ids = model.generate(
                **encoded, max_new_tokens=256, do_sample=False, use_audio_in_video=False,
                eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id,
            )
            response_ids = ids[:, encoded["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(response_ids[0], skip_special_tokens=False)
            if int(response_ids.eq(seg_id).sum()) != 1:
                raise RuntimeError(f"Unexpected SFT-native SEG count case{index}: {response[-120:]}")
            full = torch.cat([encoded["input_ids"], response_ids], -1)
            batch = dict(encoded, input_ids=full, attention_mask=torch.ones_like(full),
                         position_ids=extend_positions(
                             positions.transpose(0, 1).to(device), response_ids.shape[1]
                         ).transpose(0, 1))
            hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1]
            prompt = projector(hidden[full.eq(seg_id)].to(next(projector.parameters()).dtype))[0]
            frames, _, sizes = preprocess_sam2_frames(row["sam_frame_paths"], 10)
            gt = load_gt_masks(row["mask_paths"], sizes[0], 10).to(device)
            logits = sam(frames.to(device), prompt, tuple(gt.shape[-2:]))
            frame = [frame_metric(logits[j], gt[j])[0] for j in range(10)]
            score = sum(frame) / 10
        results.append({
            "case_index": index, "sample_id": row["sample_id"],
            "native_iou": score, "saved_swift_iou": reference[index]["iou"],
            "native_response": response,
            "saved_swift_response": reference[index]["response"],
            "response_exact_equal": response.replace("<|im_end|>", "").strip() ==
                                    reference[index]["response"].strip(),
            "native_frame_ious": frame,
            "saved_swift_frame_ious": reference[index].get("frame_ious"),
        })
        print(index, round(score, 4), round(reference[index]["iou"], 4),
              results[-1]["response_exact_equal"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "scope": "SFT native+frozen trained SAM2 vs prior official Swift SFT; first "
                 f"{args.count} test_s cases only; not gain evidence",
        "native_model": str(model_path), "swift_checkpoint": str(INITIALIZER),
        "media": "same ten stored frames and original WAV",
        "intended_inference_differences": [
            "native HF mRoPE instead of historical Swift pre-forward-hook",
            "BF16-merged SFT language weights vs unmerged Swift SFT adapter",
            "response re-generated by native HF, not replaying Swift response",
        ],
        "items": results,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
