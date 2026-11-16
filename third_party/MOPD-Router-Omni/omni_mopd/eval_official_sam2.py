"""Official REF-AVS mask evaluation: trained SAM2 + projector, frozen during native OPD.

Generate with native Thinker (canonical mRoPE). Teacher-force the exact
generated token IDs for SEG hidden states; never use the ground-truth answer.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from omni_mopd.data import OmniEncoder, audit_timing
from omni_mopd.eval_official_language import sha
from omni_mopd.modeling import extend_positions
from swift_plugin.sam2_segmentation import AuroraSAM2VideoHead, preprocess_sam2_frames
from swift_plugin.segmentation import load_gt_masks
from swift_plugin.sam2_grpo_rewards import (
    class_exact_score, class_name_from_sample_id, extract_target_class,
)

ROOT = Path(os.environ.get("AURORA_ROOT", str(Path(__file__).resolve().parents[3])))
INITIALIZER = Path(os.environ.get("AURORA_SFT_INITIALIZER", str(ROOT / "outputs/train/sam2_merged_sft_3epoch_8gpu/checkpoint-4809")))
BASE_SAM2 = Path(os.environ.get("AURORA_SAM2_CHECKPOINT", str(ROOT / "models/sam2_hiera_large.pt")))
MODEL_TRAIN = Path(os.environ.get("AURORA_MOPD_TRAIN_DIR", str(ROOT / "outputs/train/clean_omni_mopd_8gpu")))

def _mask_sample_dir_name(sample_id, index):
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_"
                   for c in str(sample_id)).strip(".")
    return f"{index:06d}__{safe or 'unknown'}"


def projector_and_sam(device):
    config = json.loads((INITIALIZER / "aurora_projector_config.json").read_text())
    if {k: config[k] for k in ("backend", "sam_frames", "model_timeline_fps", "input_hidden_size",
                               "projector_hidden_size", "output_hidden_size")} != {
        "backend": "sam2_video_memory", "sam_frames": 10, "model_timeline_fps": 1.,
        "input_hidden_size": 3584, "projector_hidden_size": 1024, "output_hidden_size": 256,
    }:
        raise ValueError("Trained projector contract differs from selected checkpoint")
    projector = nn.Sequential(
        nn.Linear(3584, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 256),
        nn.Dropout(0.0),
    )
    with safe_open(str(INITIALIZER / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        state = {k.split(".text_hidden_fcs.", 1)[1]: f.get_tensor(k)
                 for k in f.keys() if ".text_hidden_fcs." in k}
    projector.load_state_dict(state, strict=True)
    projector.to(device).eval()
    sam = AuroraSAM2VideoHead(str(BASE_SAM2), config["sam2_config"])
    saved = load_file(INITIALIZER / "sam2_video_decoder.safetensors")
    if len(saved) != 285:
        raise ValueError("Wrong trained SAM2 sidecar tensor count")
    sam.load_trainable_state_dict(saved)
    sam.to(device).eval()
    return projector, sam


def frame_metric(pred, target):
    pred = pred.float().sigmoid().ge(.5)
    truth = target.ge(.5)
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    pixels = int(pred.sum() + truth.sum())
    return (intersection / union if union else 1.,
            2 * intersection / pixels if pixels else 1., pred)


def run(args):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo")
    snapshot = json.loads((args.checkpoint / "fsdp_snapshot_manifest.json").read_text())
    if not snapshot["complete"] or snapshot["source_step"] != 500:
        raise ValueError("Unverified/mismatched native checkpoint")
    if Path(snapshot["source_checkpoint"]).parent.parent.resolve() != MODEL_TRAIN.resolve():
        raise ValueError("Snapshot is not this run's native FSDP checkpoint")
    if not snapshot["all_frozen_tensors_exact_after_dtype_cast"]:
        raise ValueError("Frozen initializer weights differ")
    timing = audit_timing(ROOT / "outputs/artifacts/refavs_fps_audit_20261025/audit.json")
    saved_config = json.loads((MODEL_TRAIN / "global_step_500/actor/huggingface/config.json").read_text())
    if saved_config.get("model_type") != "qwen2_5_omni_thinker" or saved_config.get("attn_implementation") != "sdpa":
        raise ValueError("Trained model type/SDPA differs")
    source = ROOT / "outputs/data/refavs_official_template_matched_20261102" / f"refavs_{args.split}.jsonl"
    rows = [json.loads(line) for line in source.open() if line.strip()]
    full = {"test_s": 2288, "test_u": 1656}[args.split]
    if len(rows) != full:
        raise ValueError("Changed official split")
    previous = json.loads((
        ROOT / "outputs/eval/merged_sft_official_20261102_merged_v4_full/epoch3"
        / args.split / "evaluation_manifest.json"
    ).read_text())
    if previous["source_sha256"] != sha(source):
        raise ValueError("Official source SHA differs from initializer's evaluated split")
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Refuse overwriting existing mask evaluation")
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": 500,
            "source_checkpoint": snapshot["source_checkpoint"],
            "split": args.split, "samples": len(rows),
            "full_official_split": len(rows) == full,
            "source": str(source), "source_sha256": sha(source),
            "training_config_source": str(
                MODEL_TRAIN / "global_step_500/actor/huggingface/config.json"),
            "training_launcher": str(ROOT / "third_party/MOPD-Router-Omni/run_omni.sh"),
            "initializer_args": str(INITIALIZER / "args.json"),
            "trained_mask_components": str(INITIALIZER),
            "trained_mask_base": str(BASE_SAM2),
            "model_timings": timing,
            "settings": {
                "world_size": world, "max_pixels": 200704, "max_length": 5120,
                "max_prompt_length": 4864, "response_length": 256,
                "dtype": "bfloat16", "attention": "sdpa",
                "fps_max_frames": 768, "video_max_token_num": 768,
                "image_max_token_num": 16384, "model_timeline_fps": 1.,
                "sam_frames": 10, "template": "omni_mopd.data.render_prompt",
                "audio": "original WAV", "decoding": "greedy",
                "save_masks": True, "per_frame_metrics": True,
                "mask_directory": str((args.output / "masks").resolve()),
            },
            "inference_differences": [
                "greedy decoding versus stochastic temperature1/top_p1 native training rollout",
                "frozen trained SFT4809 SAM2/projector reattached only for inference",
            ],
            "no_opd_matched_control": False, "opd_gain_claim": False,
        }
        (args.output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).to(device).eval()
    projector, sam = projector_and_sam(device)
    seg_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    encoder = OmniEncoder(processor, model.config, fps=1., max_pixels=200704)
    record_path = args.output / f"{args.split}.rank{rank}.jsonl"
    count = class_count = seg_count = 0
    sums = [0., 0.]
    with record_path.open("w", buffering=1) as f:
        for index in range(rank, len(rows), world):
            row = rows[index]
            with torch.inference_mode():
                encoded, positions, _ = encoder.encode(row)
                if encoded["input_ids"].shape[1] > 4864:
                    raise ValueError("Prompt exceeds trained 4864-token cap")
                encoded = {k: v.to(device, dtype=torch.bfloat16)
                           if v.is_floating_point() else v.to(device)
                           for k, v in encoded.items()}
                generated = model.generate(
                    **encoded, max_new_tokens=256, do_sample=False, use_audio_in_video=False,
                    eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id,
                )
                response_ids = generated[:, encoded["input_ids"].shape[1]:]
                response = processor.tokenizer.decode(
                    response_ids[0].tolist(), skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                category = str(row.get("class_name") or "").strip() or class_name_from_sample_id(row["sample_id"])
                cls = bool(class_exact_score(response, category, row["sample_id"]))
                nseg = int(response_ids.eq(seg_id).sum())
                per_iou, per_dice, saved_masks = [0.] * 10, [0.] * 10, []
                if nseg == 1:
                    full_ids = torch.cat([encoded["input_ids"], response_ids], dim=-1)
                    forward = dict(encoded)
                    forward.update(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                                   position_ids=extend_positions(
                                       positions.transpose(0, 1).to(device),
                                       response_ids.shape[1],
                                   ).transpose(0, 1))
                    output = model(**forward, output_hidden_states=True, use_cache=False)
                    hidden = output.hidden_states[-1][full_ids.eq(seg_id)]
                    if hidden.shape != (1, 3584):
                        raise ValueError(f"Expected one 3584D native SEG state, got {hidden.shape}")
                    prompt = projector(hidden.to(next(projector.parameters()).dtype))[0]
                    images, _, sizes = preprocess_sam2_frames(row["sam_frame_paths"], 10)
                    truth = load_gt_masks(row["mask_paths"], sizes[0], 10).to(device)
                    logits = sam(images.to(device), prompt, tuple(truth.shape[-2:]))
                    folder = args.output / "masks" / _mask_sample_dir_name(row["sample_id"], index)
                    folder.mkdir(parents=True, exist_ok=True)
                    for frame in range(10):
                        iou, dice, mask = frame_metric(logits[frame], truth[frame])
                        per_iou[frame], per_dice[frame] = iou, dice
                        mask_path = folder / f"frame_{frame:03d}.png"
                        Image.fromarray(mask.cpu().numpy().astype("uint8") * 255).save(mask_path)
                        saved_masks.append(str(mask_path))
                    del output
                iou = sum(per_iou) / 10
                dice = sum(per_dice) / 10
                count += 1
                class_count += int(cls)
                seg_count += int(nseg == 1)
                sums[0] += iou
                sums[1] += dice
                record = {
                    "split": args.split, "case_index": index, "sample_id": row["sample_id"],
                    "modality": row["template_match"]["template_reference"],
                    "class_name": category, "predicted_class": extract_target_class(response),
                    "class_correct": cls, "response": response, "seg_count": nseg,
                    "response_token_count": response_ids.shape[1], "iou": iou, "dice": dice,
                    "frame_ious": per_iou, "frame_dices": per_dice,
                    "mask_paths": saved_masks, "frames_scored": len(saved_masks),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if count % 50 == 0:
                print(f"[{args.split} rank{rank}] {count}/{len(rows[rank::world])} "
                      f"mIoU={sums[0]/count:.4f} SEG={seg_count/count:.3f}", flush=True)
    (args.output / f"summary_{args.split}.rank{rank}.json").write_text(json.dumps({
        "samples": count, "seg_emitted": seg_count, "class_correct": class_count,
        "iou_sum": sums[0], "dice_sum": sums[1], "errors": 0,
    }, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    if rank == 0:
        rank_summaries = [json.loads(
            (args.output / f"summary_{args.split}.rank{i}.json").read_text()
        ) for i in range(world)]
        merged = [json.loads(line) for i in range(world)
                  for line in (args.output / f"{args.split}.rank{i}.jsonl").open() if line.strip()]
        merged.sort(key=lambda x: x["case_index"])
        if [record["case_index"] for record in merged] != list(range(len(rows))):
            raise RuntimeError("Missing/duplicated official test records")
        (args.output / f"{args.split}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in merged)
        )
        summary = {k: sum(item[k] for item in rank_summaries) for k in rank_summaries[0]}
        if summary["samples"] != len(rows) or summary["errors"]:
            raise RuntimeError("Incomplete full-split segmentation evaluation")
        summary.update(
            split=args.split, checkpoint_step=500,
            full_official_split=len(rows) == full,
            mean_iou_all=summary["iou_sum"] / summary["samples"],
            mean_dice_all=summary["dice_sum"] / summary["samples"],
            seg_rate=summary["seg_emitted"] / summary["samples"],
            class_accuracy=summary["class_correct"] / summary["samples"],
        )
        (args.output / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("test_s", "test_u"))
    p.add_argument("--max-samples", type=int, default=-1)
    run(p.parse_args())
