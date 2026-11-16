"""Save the trainable SAM decoder as a portable checkpoint sidecar."""

from __future__ import annotations

import os
import json

from safetensors.torch import load_file, save_file
from swift.callbacks import TrainerCallback, callbacks_map


def _base_model(model):
    current = model
    for _ in range(4):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        break
    return current


def _normalize_sam2_adapter_paths(checkpoint_dir: str) -> None:
    """Remove runtime compatibility-proxy duplication from PEFT save keys."""
    path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if not os.path.isfile(path):
        return
    state = load_file(path, device="cpu")
    normalized = {}
    changed = False
    for key, value in state.items():
        new_key = key.replace(
            ".sam_mask_decoder.decoder.decoder.", ".sam_mask_decoder.decoder."
        ).replace(
            ".memory_attention.attention.attention.", ".memory_attention.attention."
        )
        if new_key in normalized:
            raise RuntimeError(f"SAM2 adapter key normalization collision: {new_key}")
        normalized[new_key] = value.contiguous()
        changed = changed or new_key != key
    if changed:
        temporary = path + ".normalized.tmp"
        save_file(normalized, temporary, metadata={"format": "pt"})
        os.replace(temporary, path)


class AuroraSAMCheckpointCallback(TrainerCallback):
    """Write the trainable SAM mask decoder next to each ms-swift checkpoint."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)

    def on_save(self, args, state, control, model=None, **kwargs):
        # Transformers' on_save event does not consistently include the model.
        model = model or self.trainer.model
        if model is None or not getattr(state, "is_world_process_zero", True):
            return control
        base = _base_model(model)
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        sam2_head = getattr(base, "sam2_head", None)
        if sam2_head is not None:
            sam2_state = sam2_head.trainable_state_dict()
            save_file(
                sam2_state,
                os.path.join(checkpoint_dir, "sam2_video_decoder.safetensors"),
                metadata={"format": "pt", "backend": "sam2_video_memory"},
            )
            projector_config = getattr(base, "aurora_projector_config", None)
            if projector_config:
                with open(os.path.join(checkpoint_dir, "aurora_projector_config.json"), "w", encoding="utf-8") as handle:
                    json.dump(projector_config, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            _normalize_sam2_adapter_paths(checkpoint_dir)
            return control
        sam = getattr(base, "sam", None)
        decoder = getattr(sam, "mask_decoder", None)
        if decoder is None:
            return control

        # PEFT wraps modules_to_save while the default copy is the trainable one.
        trained = getattr(decoder, "modules_to_save", None)
        if trained is not None and "default" in trained:
            decoder = trained["default"]
        state_dict = {key: value.detach().cpu().contiguous() for key, value in decoder.state_dict().items()}
        save_file(state_dict, os.path.join(checkpoint_dir, "sam_mask_decoder.safetensors"), metadata={"format": "pt"})
        projector_config = getattr(base, "aurora_projector_config", None)
        if projector_config:
            with open(os.path.join(checkpoint_dir, "aurora_projector_config.json"), "w", encoding="utf-8") as handle:
                json.dump(projector_config, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return control


callbacks_map["aurora_sam_checkpoint"] = AuroraSAMCheckpointCallback
