"""Official REF-AVS mask evaluation: trained SAM2 + projector, frozen during native OPD.

Generate with native Thinker (canonical mRoPE). Teacher-force the exact
generated token IDs for SEG hidden states; never use the ground-truth answer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn


def _add_import_paths() -> None:
    mopd_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (mopd_root, repo_root, repo_root / "third_party/sam2"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_add_import_paths()

from omni_mopd.data import OmniEncoder
from omni_mopd.eval_official_language import (
    OFFICIAL_SPLIT_COUNTS,
    verify_fsdp_snapshot,
    verify_official_split_source,
    verify_timing_audit,
)
from omni_mopd.modeling import extend_positions
from swift_plugin.sam2_segmentation import AuroraSAM2VideoHead, preprocess_sam2_frames
from swift_plugin.segmentation import load_gt_masks
from swift_plugin.sam2_grpo_rewards import (
    class_exact_score, class_name_from_sample_id, extract_target_class,
)

ROOT = Path(os.environ.get("AURORA_ROOT") or str(Path(__file__).resolve().parents[3]))
INITIALIZER = Path(os.environ.get("AURORA_SFT_INITIALIZER") or str(ROOT / "outputs/train/sam2_merged_sft_3epoch_8gpu/checkpoint-4809"))
BASE_SAM2 = Path(
    os.environ.get("AURORA_SAM2_CHECKPOINT")
    or os.environ.get("SAM2_CHECKPOINT")
    or str(ROOT / "models/sam2_hiera_large.pt")
)
MODEL_TRAIN = Path(os.environ.get("AURORA_MOPD_TRAIN_DIR") or str(ROOT / "outputs/train/clean_omni_mopd_8gpu"))

def _mask_sample_dir_name(sample_id, index):
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_"
                   for c in str(sample_id)).strip(".")
    return f"{index:06d}__{safe or 'unknown'}"


def projector_and_sam(
    device,
    initializer: Path = INITIALIZER,
    base_sam2: Path = BASE_SAM2,
):
    initializer = Path(initializer)
    base_sam2 = Path(base_sam2)
    config = json.loads((initializer / "aurora_projector_config.json").read_text())
    required = {
        "backend": "sam2_video_memory",
        "sam_frames": 10,
        "model_timeline_fps": 1.0,
        "output_hidden_size": 256,
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Trained projector contract differs from selected checkpoint")
    input_hidden_size = int(config["input_hidden_size"])
    projector_hidden_size = int(config["projector_hidden_size"])
    output_hidden_size = int(config["output_hidden_size"])
    projector = nn.Sequential(
        nn.Linear(input_hidden_size, projector_hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(projector_hidden_size, output_hidden_size),
        nn.Dropout(0.0),
    )
    with safe_open(str(initializer / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        state = {k.split(".text_hidden_fcs.", 1)[1]: f.get_tensor(k)
                 for k in f.keys() if ".text_hidden_fcs." in k}
    projector.load_state_dict(state, strict=True)
    projector.to(device).eval()
    sam = AuroraSAM2VideoHead(str(base_sam2), config["sam2_config"])
    saved = load_file(initializer / "sam2_video_decoder.safetensors")
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
    train_dir = getattr(args, "train_dir", None) or MODEL_TRAIN
    initializer = getattr(args, "sft_initializer", None) or INITIALIZER
    base_sam2 = getattr(args, "sam2_checkpoint", None) or BASE_SAM2
    timing_audit = getattr(args, "timing_audit", None) or Path(
        os.environ.get("OMNI_TIMING_AUDIT") or str(ROOT / "outputs/artifacts/refavs_fps_audit_20261025/audit.json")
    )
    verified = verify_fsdp_snapshot(
        args.checkpoint,
        train_dir,
        expected_step=getattr(args, "expected_step", None),
    )
    snapshot = verified["snapshot"]
    source_step = verified["source_step"]
    video_preprocessing = verified["video_preprocessing"]
    timing = verify_timing_audit(timing_audit)
    saved_config = json.loads(verified["source_settings"].read_text())
    if saved_config.get("model_type") != "qwen2_5_omni_thinker" or saved_config.get("attn_implementation") != "sdpa":
        raise ValueError("Trained model type/SDPA differs")
    default_data_dir = Path(os.environ.get("AURORA_OFFICIAL_DATA_DIR") or str(ROOT / "outputs/data/refavs_official_template_matched_20261102"))
    source = getattr(args, "source", None) or (default_data_dir / f"refavs_{args.split}.jsonl")
    rows, source_provenance = verify_official_split_source(
        source,
        args.split,
        root=ROOT,
        expected_source=getattr(args, "expected_source", None),
        reference_manifest=getattr(args, "reference_manifest", None),
        media_manifest=getattr(args, "media_manifest", None),
    )
    full = OFFICIAL_SPLIT_COUNTS[args.split]
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Refuse overwriting existing mask evaluation")
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": source_step,
            "source_checkpoint": snapshot["source_checkpoint"],
            "split": args.split, "samples": len(rows),
            "full_official_split": len(rows) == full,
            "source": str(Path(source).resolve()),
            **source_provenance,
            "training_config_source": str(verified["source_settings"]),
            "training_launcher": str(Path(__file__).resolve().parents[1] / "run_omni.sh"),
            "initializer_args": str(initializer / "args.json"),
            "trained_mask_components": str(initializer),
            "trained_mask_base": str(base_sam2),
            "model_timings": timing,
            "video_preprocessing_source": str(train_dir / "omni_training_manifest.json"),
            "settings": {
                "world_size": world, "max_pixels": 200704, "max_length": 5120,
                "max_prompt_length": 4864, "response_length": 256,
                "dtype": "bfloat16", "attention": "sdpa",
                "fps_max_frames": 768, "video_max_token_num": 768,
                "image_max_token_num": 16384, "model_timeline_fps": 1.,
                "sam_frames": 10, "template": "omni_mopd.data.render_prompt",
                "video_preprocessing": video_preprocessing,
                "audio": "original WAV", "decoding": "greedy",
                "save_masks": True, "per_frame_metrics": True,
                "mask_directory": str((args.output / "masks").resolve()),
            },
            "inference_differences": [
                "greedy decoding versus stochastic temperature1/top_p1 native training rollout",
                "frozen trained SAM2/projector reattached only for inference",
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
    projector, sam = projector_and_sam(device, initializer=initializer, base_sam2=base_sam2)
    input_hidden_size = int(projector[0].in_features)
    seg_id = processor.tokenizer.convert_tokens_to_ids("[SEG]")
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    encoder = OmniEncoder(processor, model.config, fps=1., max_pixels=200704,
                          video_preprocessing=video_preprocessing)
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
                    if hidden.shape != (1, input_hidden_size):
                        raise ValueError(f"Expected one {input_hidden_size}D native SEG state, got {hidden.shape}")
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
            split=args.split, checkpoint_step=source_step,
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("test_s", "test_u"))
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--train-dir", type=Path, default=MODEL_TRAIN)
    p.add_argument("--sft-initializer", type=Path, default=INITIALIZER)
    p.add_argument("--sam2-checkpoint", type=Path, default=BASE_SAM2)
    p.add_argument("--timing-audit", type=Path, default=None)
    p.add_argument("--expected-source", type=Path, default=None)
    p.add_argument("--reference-manifest", type=Path, default=None)
    p.add_argument("--media-manifest", type=Path, default=None)
    p.add_argument("--expected-step", type=int, default=None)
    run(p.parse_args())
