"""REF-AVS media -> native Omni tensors, using the stock Ray dataset interface."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from omni_mopd.modeling import rope_positions


def audit_timing(path):
    audit = json.loads(Path(path).read_text())
    if audit.get("nominal_fps") != 1 or audit.get("unique_clips_checked") != 3566:
        raise ValueError("Unrecognized media timing audit")
    if not all(
        d["frames"] == d["used_frames"] == 10 and d["clip_seconds"] == 10
        for d in audit["distribution"]
    ):
        raise ValueError("Stored media timing has changed")
    return {
        "nominal_frame_density_fps": 1.0,
        "model_timeline_fps": 1.0,
        "stored_frames": 10,
        "exact_sampling_instants_verified": False,
        "audit_source": str(Path(path).resolve()),
    }


def prompt_messages(row):
    messages = []
    for message in row["messages"]:
        if message["role"] == "assistant":
            break
        messages.append(copy.deepcopy(message))
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected prompt-only user conversation")
    if sum(m["content"].count("<video>") for m in messages) != 1:
        raise ValueError("Exactly one stored-frame video required")
    if sum(m["content"].count("<audio>") for m in messages) != 1:
        raise ValueError("Exactly one original WAV required")
    return messages


def render_prompt(messages):
    # Match the initializer's explicit video + separate audio user template.
    text = ""
    for message in messages:
        content = message["content"].replace(
            "<video>", "<|vision_bos|><|VIDEO|><|vision_eos|>"
        ).replace("<audio>", "<|audio_bos|><|AUDIO|><|audio_eos|>")
        text += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


class OmniEncoder:
    def __init__(self, processor, model_config, *, fps, max_pixels):
        if fps != 1.0:
            raise ValueError("Selected REF-AVS initializer/data contract requires 1 FPS")
        self.processor, self.model_config = processor, model_config
        self.fps, self.max_pixels = fps, max_pixels

    def encode(self, row):
        import librosa
        from qwen_omni_utils import fetch_video, vision_process

        messages = prompt_messages(row)
        frames = row["videos"][0]
        if len(frames) != 10 or frames != row["sam_frame_paths"]:
            raise ValueError("Expected exactly the audited ten stored frames, in order")
        if len(row["audios"]) != 1:
            raise ValueError("Expected one separate original audio input")
        for path in [*frames, row["audios"][0]]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        # Values come from the checked initializer and timing audit, not a
        # default vLLM/torchrun launcher. No SAM runtime dependency is needed.
        for name, value in {
            "FPS": self.fps, "FPS_MAX_FRAMES": 768,
            "VIDEO_MAX_TOKEN_NUM": 768, "IMAGE_MAX_TOKEN_NUM": 16384,
        }.items():
            setattr(vision_process, name, value)
        video = fetch_video({
            "video": frames, "sample_fps": self.fps, "max_pixels": self.max_pixels,
        }).to(torch.uint8)
        sr = self.processor.feature_extractor.sampling_rate
        audio, _ = librosa.load(row["audios"][0], sr=sr)
        result = dict(self.processor(
            text=[render_prompt(messages)], videos=[video], audio=[audio],
            sampling_rate=sr, fps=self.fps, do_resize=False,
            use_audio_in_video=False, return_tensors="pt",
        ))
        result["video_second_per_grid"] = torch.tensor([
            self.processor.video_processor.temporal_patch_size / self.fps
        ])
        for name, value in result.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Non-tensor processor input would be lost in actor collation: {name}")
        positions = rope_positions(self.model_config, result)
        return result, positions, messages


class OmniReasoningDataset(Dataset):
    """Raw JSONL loader; all rows retained; overlength errors are never hidden."""
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        from transformers import AutoConfig

        if processor is None or "Omni" not in type(processor).__name__:
            raise ValueError("A real Omni processor is required; refusing text-only fallback")
        self.tokenizer, self.config = tokenizer, config
        self.max_prompt_length = int(config.max_prompt_length)
        self.rows = []
        paths = [data_files] if isinstance(data_files, str) else list(data_files)
        for path in paths:
            with Path(path).open() as f:
                self.rows.extend(json.loads(line) for line in f if line.strip())
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.media_protocol = audit_timing(config.omni_timing_audit)
        model_config = AutoConfig.from_pretrained(config.omni_model_path, local_files_only=True)
        if getattr(model_config, "model_type", "") != "qwen2_5_omni_thinker":
            raise ValueError("Export a standalone native Omni Thinker before training")
        self.encoder = OmniEncoder(
            processor, model_config, fps=self.media_protocol["model_timeline_fps"],
            max_pixels=int(config.omni_max_pixels),
        )
        if not self.rows:
            raise ValueError("Empty training/debug dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded, positions, messages = self.encoder.encode(row)
        ids, mask = encoded.pop("input_ids"), encoded.pop("attention_mask")
        n = ids.shape[1]
        if n > self.max_prompt_length:
            raise ValueError(f"{row['sample_id']}: {n} prompt tokens > {self.max_prompt_length}; no truncation")
        pad = self.max_prompt_length - n
        return {
            "input_ids": torch.nn.functional.pad(ids[0], (pad, 0), value=self.tokenizer.pad_token_id),
            "attention_mask": torch.nn.functional.pad(mask[0], (pad, 0), value=0),
            "position_ids": torch.nn.functional.pad(positions[:, 0], (pad, 0), value=1),
            "multi_modal_inputs": encoded,
            "raw_prompt_ids": ids[0].tolist(),
            "raw_prompt": messages,
            "data_source": "refavs_language_only",
            "reward_model": {"style": "rule", "ground_truth": row["class_name"]},
            "extra_info": {"index": index, "sample_id": row["sample_id"],
                           "modality": row["template_match"]["template_reference"],
                           "language_only": True, "segmentation_decoded": False},
        }
