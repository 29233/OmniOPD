#!/usr/bin/env python3
"""CPU audit: Omni train/inference prefix IDs, media timing and task prompt.

This loads a processor/tokenizer only. It does not load model weights or
measure model quality; run actual test_s/test_u inference separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open


def read_case(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i == index:
                return json.loads(line)
    raise IndexError(f"No case {index} in {path}")


def task_header(row: dict) -> str:
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    if user.count("<video>") != 1 or user.count("<audio>") != 1:
        raise ValueError("Expected one separate video and audio placeholder")
    return user.split("<video><audio>", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--train-source", required=True, type=Path)
    p.add_argument("--test-source", required=True, type=Path)
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--native-thinker", type=Path,
                   help="Also compare native MOPD's processor tokens/media to ms-swift's template")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    saved = json.loads((args.checkpoint / "args.json").read_text())
    projector = json.loads((args.checkpoint / "aurora_projector_config.json").read_text())
    if saved["model_type"] != "aurora_qwen2_5_omni" or saved["template"] != "aurora_qwen2_5_omni":
        raise ValueError("Checkpoint did not train with the Aurora Omni template")
    fps = float(projector["model_timeline_fps"])
    for key, value in {
        "FPS": fps, "AURORA_VIDEO_FPS": fps, "FPS_MAX_FRAMES": 768,
        "VIDEO_MAX_TOKEN_NUM": 768, "IMAGE_MAX_TOKEN_NUM": 16384,
        "ENABLE_AUDIO_OUTPUT": 0, "USE_AUDIO_IN_VIDEO": 0,
    }.items():
        os.environ[str(key)] = str(value)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from swift.model import get_processor
    from swift.template import get_template
    import swift_plugin.aurora_qwen2_5_omni  # noqa: F401 - register model/template

    test = read_case(args.test_source, args.case_index)
    train = read_case(args.train_source, 0)
    styles_match = task_header(train) == task_header(test)
    processor = get_processor(str(args.model), model_type=saved["model_type"],
                              new_special_tokens=saved["new_special_tokens"], download_model=False)
    template = get_template(processor, template_type=saved["template"],
                            max_length=saved["max_length"], max_pixels=saved["max_pixels"])
    row = {k: copy.deepcopy(test[k]) for k in ("messages", "videos", "audios")}
    if len(row["videos"]) != 1 or len(row["videos"][0]) != projector["sam_frames"] or len(row["audios"]) != 1:
        raise ValueError("Unexpected test media frame/audio count")
    template.set_mode("train")
    forced = template.encode(copy.deepcopy(row))
    labels = forced["labels"]
    answer_begin = next(i for i, label in enumerate(labels) if label != -100)
    template.set_mode("transformers")
    prompt_row = {**copy.deepcopy(row), "messages": row["messages"][:1]}
    prompt = template.encode(prompt_row)
    prompt_ids = prompt["input_ids"]
    seg_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")
    with safe_open(str(args.checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu") as weights:
        embed_shape = weights.get_slice("base_model.model.thinker.model.embed_tokens.weight").get_shape()
        head_shape = weights.get_slice("base_model.model.thinker.lm_head.weight").get_shape()
    seconds_per_grid = [float(value) for value in prompt["video_second_per_grid"]]
    temporal_patch = processor.video_processor.temporal_patch_size
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train_source": str(args.train_source.resolve()),
        "test_source": str(args.test_source.resolve()),
        "case_index": args.case_index, "sample_id": test["sample_id"],
        "train_test_task_header_same": styles_match,
        "train_task_header_sha256": hashlib.sha256(task_header(train).encode()).hexdigest(),
        "test_task_header_sha256": hashlib.sha256(task_header(test).encode()).hexdigest(),
        "model_timeline_fps": fps, "temporal_patch_size": temporal_patch,
        "video_second_per_grid": seconds_per_grid,
        "video_grid_thw": prompt["video_grid_thw"].tolist(),
        "stored_video_frames": len(row["videos"][0]),
        "audio_input_features_shape": list(prompt["input_features"].shape),
        "input_prompt_tokens": len(prompt_ids), "train_prefix_tokens": answer_begin,
        "train_inference_prefix_token_ids_exact": forced["input_ids"][:answer_begin] == prompt_ids,
        "train_answer_labeled_tokens": sum(label != -100 for label in labels),
        "seg_id": seg_id, "seg_label_count": sum(label == seg_id for label in labels),
        "saved_embedding_shape": embed_shape, "saved_lm_head_shape": head_shape,
        "remaining_trained_sequence_budget": saved["max_length"] - len(prompt_ids),
        "template_class": type(template).__name__,
    }
    if args.native_thinker is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                               / "third_party/MOPD-Router-Omni"))
        from transformers import AutoConfig
        from omni_mopd.data import OmniEncoder

        native, positions, _ = OmniEncoder(
            processor, AutoConfig.from_pretrained(args.native_thinker, local_files_only=True),
            fps=fps, max_pixels=saved["max_pixels"],
        ).encode(test)
        report["native"] = {
            "thinker_config": str(args.native_thinker.resolve()),
            "prompt_tokens": native["input_ids"].shape[1],
            "same_prompt_ids": native["input_ids"][0].tolist() == prompt_ids,
            "same_video_grid": native["video_grid_thw"].tolist() == prompt["video_grid_thw"].tolist(),
            "same_video_values_after_cast": bool(
                (native["pixel_values_videos"].to(prompt["pixel_values_videos"].dtype)
                 == prompt["pixel_values_videos"]).all()
            ),
            "same_audio_values_after_cast": bool(
                (native["input_features"].to(prompt["input_features"].dtype)
                 == prompt["input_features"]).all()
            ),
            "position_ids_shape": list(positions.shape),
        }
    report["valid"] = (
        styles_match and report["train_inference_prefix_token_ids_exact"]
        and all(abs(v - temporal_patch / fps) < 1e-6 for v in seconds_per_grid)
        and seg_id < embed_shape[0] == head_shape[0]
        and report["seg_label_count"] == 1
        and report["remaining_trained_sequence_budget"] > 0
        and (args.native_thinker is None or all(
            report["native"][key] for key in (
                "same_prompt_ids", "same_video_grid",
                "same_video_values_after_cast", "same_audio_values_after_cast"
            )
        ))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["valid"]:
        raise SystemExit("Template contract mismatch; see audit JSON")


if __name__ == "__main__":
    main()
