"""Explicit RANDOM tiny Omni fixture for native Ray/FSDP integration, NOT a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    from transformers import AutoConfig, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from omni_mopd.test_migration import tiny_thinker

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Real Omni processor/config, no base model weights loaded")
    p.add_argument("--source", required=True, help="Real REF-AVS training JSONL")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    processor = Qwen2_5OmniProcessor.from_pretrained(a.base, local_files_only=True)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[SEG]"]})
    base_config = AutoConfig.from_pretrained(a.base, local_files_only=True).thinker_config
    config = tiny_thinker().config
    config.text_config.vocab_size = base_config.text_config.vocab_size
    config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    config.text_config.eos_token_id = processor.tokenizer.eos_token_id
    config.vision_config.patch_size = 14
    config.vision_config.window_size = 112
    config.audio_config.num_mel_bins = 128
    config.audio_config.max_source_positions = 1500
    config.audio_config.n_window = 100
    for key in (
        "image_token_id", "video_token_id", "audio_token_id",
        "vision_start_token_id", "vision_end_token_id", "audio_start_token_id",
        "audio_end_token_id", "user_token_id",
    ):
        if hasattr(base_config, key):
            setattr(config, key, getattr(base_config, key))
    for i, name in enumerate(("student", "base", "teacher1", "teacher2", "teacher3")):
        torch.manual_seed(99 + i)
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        folder = out / name
        model.save_pretrained(folder)
        processor.save_pretrained(folder)
        (folder / "omni_export_manifest.json").write_text(json.dumps({
            "complete": True, "format": "merged-native-thinker-v1",
            "segmentation_exported": False, "base": str(Path(a.base).resolve()),
            "adapter": None, "scope": "RANDOM_TINY_INTEGRATION_FIXTURE_NOT_A_TRAINED_MODEL",
        }, indent=2))
    rows = []
    with Path(a.source).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == 2:
                break
    (out / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(out)


if __name__ == "__main__":
    main()
