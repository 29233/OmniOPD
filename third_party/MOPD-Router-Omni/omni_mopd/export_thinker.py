"""Export a merged native Thinker; discard segmentation weights explicitly."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def language_adapter_key(name):
    prefix = "base_model.model.thinker."
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix):]
    return suffix.startswith("model.layers.") or suffix.startswith(
        ("model.embed_tokens.", "lm_head.")
    )


def export(base, adapter, out, device):
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    base, out = Path(base).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # A completion marker is written last. Failed/partial exports are never
    # accepted by preflight as independent teacher checkpoints.
    processor = Qwen2_5OmniProcessor.from_pretrained(base, local_files_only=True)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["[SEG]"]})
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        base, enable_audio_output=False, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, device_map=device,
    )
    provenance = {"base": str(base), "adapter": None, "segmentation_exported": False}
    if adapter:
        adapter = Path(adapter).resolve()
        config = json.loads((adapter / "adapter_config.json").read_text())
        if Path(config["base_model_name_or_path"]).resolve() != base:
            raise ValueError("Adapter base does not match selected native Omni checkpoint")
        with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
            keys = list(f.keys())
            selected = {key: f.get_tensor(key) for key in keys if language_adapter_key(key)}
        embeddings = [v for k, v in selected.items() if k.endswith("model.embed_tokens.weight")]
        if len(embeddings) != 1 or not any("lora_A" in key for key in selected):
            raise ValueError("Expected trained language LoRA and unique trained SEG embedding")
        if processor.tokenizer.convert_tokens_to_ids("[SEG]") >= embeddings[0].shape[0]:
            raise ValueError("SEG ID lies outside trained vocabulary")
        model.thinker.resize_token_embeddings(embeddings[0].shape[0], mean_resizing=False)
        config["modules_to_save"] = ["embed_tokens", "lm_head"]
        with tempfile.TemporaryDirectory(prefix="omni-language-adapter-") as tmp:
            tmp = Path(tmp)
            (tmp / "adapter_config.json").write_text(json.dumps(config))
            save_file(selected, str(tmp / "adapter_model.safetensors"))
            peft = PeftModel.from_pretrained(model, tmp, is_trainable=False)
            # Confirm *actual loaded* embedding/head, not just adapter args.
            actual = peft.get_base_model()
            verified_lora = 0
            for name, param in peft.named_parameters():
                normalized = name.replace(".lora_A.default.", ".lora_A.").replace(
                    ".lora_B.default.", ".lora_B."
                )
                if normalized in selected and ".lora_" in normalized:
                    torch.testing.assert_close(
                        param.detach().cpu(), selected[normalized].to(param.dtype),
                        atol=0, rtol=0,
                    )
                    verified_lora += 1
            if verified_lora != sum(".lora_" in name for name in selected):
                raise ValueError("Not every selected language LoRA tensor was loaded")
            torch.testing.assert_close(
                actual.thinker.model.embed_tokens.weight.detach().cpu(),
                embeddings[0].to(actual.thinker.model.embed_tokens.weight.dtype), atol=0, rtol=0,
            )
            heads = [v for k, v in selected.items() if k.endswith("thinker.lm_head.weight")]
            if len(heads) != 1:
                raise ValueError("Missing trained language head")
            torch.testing.assert_close(
                actual.thinker.lm_head.weight.detach().cpu(),
                heads[0].to(actual.thinker.lm_head.weight.dtype), atol=0, rtol=0,
            )
            model = peft.merge_and_unload()
        provenance.update(
            adapter=str(adapter), language_tensor_count=len(selected),
            excluded_segmentation_tensor_count=len(keys) - len(selected),
            verified_loaded_lora_tensors=verified_lora,
            adapter_config_sha256=hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest(),
        )
    thinker = model.thinker
    if len(processor.tokenizer) > thinker.get_input_embeddings().num_embeddings:
        raise ValueError("Export requires a base vocabulary already covering the SEG token ID")
    if any("sam" in name or "text_hidden_fcs" in name for name, _ in thinker.named_parameters()):
        raise ValueError("Unexpected segmentation weights")
    thinker.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(out)
    provenance.update(
        model_type=thinker.config.model_type, dtype="bfloat16",
        frozen_segmentation_source=str(adapter) if adapter else None,
        format="merged-native-thinker-v1", complete=True,
    )
    (out / "omni_export_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", help="Existing SFT/GRPO adapter; only language tensors are imported")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    print(json.dumps(export(a.base, a.adapter, a.out, a.device), indent=2))


if __name__ == "__main__":
    main()
