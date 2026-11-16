"""Qwen3-Omni AURORA integration.

The native Qwen3-Omni loader does not know about the AURORA SAM branch.  This
module keeps the native multimodal processor and adds the same supervised
``[SEG] -> text_hidden_fcs -> SAM`` path used by the Qwen2.5 integration.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import Any, Dict, List

import torch

from swift_plugin.segmentation import (
    attach_aurora_segmentation,
    load_gt_masks,
    preprocess_sam_frames,
)

try:
    from swift.model.model_meta import Model, ModelGroup, ModelMeta
    from swift.model.model_arch import ModelArch
    from swift.model.register import register_model
    from swift.model.models.qwen import Qwen3OmniLoader
    from swift.template.register import register_template
    from swift.template.template_inputs import StdTemplateInputs
    from swift.template.template_meta import TemplateMeta
    from swift.template.templates.qwen import Qwen3OmniTemplate

    _SWIFT4 = True
except ImportError:  # pragma: no cover - legacy ms-swift compatibility
    from swift.llm import Model, ModelGroup, ModelMeta, TemplateMeta, register_model, register_template
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen3_omni
    from swift.llm.template.template.qwen import Qwen3OmniTemplate
    from swift.llm.template.template_inputs import StdTemplateInputs

    _SWIFT4 = False


MODEL_TYPE = "aurora_qwen3_omni"
TEMPLATE_TYPE = "aurora_qwen3_omni"


def _restore_sam_weights(model, adapters) -> None:
    """Restore modules added after the base model was loaded.

    PEFT loads the adapter before the custom SAM modules are attached in some
    ms-swift versions.  Explicitly restoring these tensors avoids silently
    replacing a resumed projector/decoder with fresh random weights.
    """
    if not adapters:
        return
    adapter_dir = adapters[-1] if isinstance(adapters, (list, tuple)) else adapters
    path = os.path.join(str(adapter_dir), "adapter_model.safetensors")
    if not os.path.isfile(path):
        return
    from safetensors.torch import load_file

    state = load_file(path)
    projector = {}
    for key, value in state.items():
        marker = ".text_hidden_fcs."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        for prefix in ("modules_to_save.default.", "original_module."):
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix):]
                break
        projector[suffix] = value
    if projector:
        model.text_hidden_fcs.load_state_dict(projector, strict=True)
    decoder = {}
    for key, value in state.items():
        marker = ".mask_decoder."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        for prefix in ("modules_to_save.default.", "original_module."):
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix):]
                break
        # MaskDecoderSaveProxy owns the original decoder under ``decoder``.
        if suffix.startswith("decoder."):
            suffix = suffix[len("decoder."):]
        decoder[suffix] = value
    if decoder:
        model.sam.mask_decoder.decoder.load_state_dict(decoder, strict=True)


def _attach(model, processor, adapters=None):
    checkpoint = os.environ.get("AURORA_SAM_CHECKPOINT", "")
    attach_aurora_segmentation(model, processor, checkpoint)
    _restore_sam_weights(model, adapters)
    return model


if _SWIFT4:

    class AuroraQwen3OmniLoader(Qwen3OmniLoader):
        def __init__(self, *args, **kwargs):
            self._aurora_adapters = kwargs.pop("adapters", None)
            super().__init__(*args, **kwargs)

        def load(self):
            model, processor = super().load()
            return _attach(model, processor, self._aurora_adapters), processor

    _LOADER = AuroraQwen3OmniLoader
    _META_KWARGS = {"loader": _LOADER, "template": TEMPLATE_TYPE}
else:  # pragma: no cover

    def _legacy_loader(model_dir, *args, **kwargs):
        model, processor = get_model_tokenizer_qwen3_omni(model_dir, *args, **kwargs)
        return _attach(model, processor), processor

    _META_KWARGS = {"get_function": _legacy_loader, "template": TEMPLATE_TYPE}


register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([Model(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            model_path=os.environ.get("QWEN3_OMNI_30B_PATH", "models/Qwen3-Omni-30B-A3B-Instruct"),
        )])],
        model_arch=ModelArch.qwen3_omni,
        is_multimodal=True,
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
        requires=["transformers>=4.57", "soundfile", "qwen_omni_utils", "decord"],
        tags=["vision", "video", "audio"],
        **_META_KWARGS,
    )
)


class AuroraQwen3OmniTemplate(Qwen3OmniTemplate):
    """Native Qwen3 encoding plus SAM supervision tensors."""

    @property
    def _VIDEO_FPS(self) -> float:
        return float(os.environ.get("AURORA_VIDEO_FPS", "1.0"))

    def _encode_truncated(self, inputs: StdTemplateInputs):
        encoded = super()._encode_truncated(inputs)
        if inputs.videos and self._VIDEO_FPS != 2.0:
            spg = encoded.get("video_second_per_grid")
            if spg:
                tps = getattr(getattr(self.processor, "video_processor", None), "temporal_patch_size", 2)
                encoded["video_second_per_grid"] = [torch.tensor(float(tps) / self._VIDEO_FPS) for _ in spg]
        return encoded

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        extras = inputs.extra_kwargs
        if not self.is_training:
            for key in ("sam_frame_paths", "mask_paths", "class_name", "sample_id"):
                extras.pop(key, None)
            return encoded
        frame_paths = extras.get("sam_frame_paths")
        mask_paths = extras.get("mask_paths")
        if frame_paths is None and mask_paths is None:
            return encoded
        if not frame_paths or not mask_paths:
            raise ValueError("sam_frame_paths and mask_paths must be provided together")
        num_frames = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
        pixels, resize_sizes, original_sizes = preprocess_sam_frames(frame_paths, num_frames)
        encoded.update(
            sam_pixel_values=pixels,
            gt_masks=load_gt_masks(mask_paths, original_sizes[0], num_frames),
            sam_resize_sizes=resize_sizes,
            sam_original_sizes=original_sizes,
        )
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = super()._post_encode(model, inputs)
        if result.get("video_second_per_grid") is None and result.get("video_grid_thw") is not None:
            video_grid_thw = result["video_grid_thw"]
            num_videos = int(video_grid_thw.shape[0]) if torch.is_tensor(video_grid_thw) else len(video_grid_thw)
            temporal_patch_size = getattr(
                getattr(self.processor, "video_processor", None), "temporal_patch_size", 2
            )
            value = float(temporal_patch_size) / self._VIDEO_FPS
            result["video_second_per_grid"] = [torch.tensor(value) for _ in range(num_videos)]
        for key in ("sam_pixel_values", "gt_masks", "sam_resize_sizes", "sam_original_sizes"):
            if key in inputs:
                result[key] = inputs[key]
        return result

    def _get_position_ids(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Parent Qwen3 code pops this value before calling get_rope_index.  A
        # few ms-swift versions omit it from the post-encode dictionary, so
        # restore it at the last point where video_grid_thw is still present.
        if inputs.get("video_second_per_grid") is None and inputs.get("video_grid_thw") is not None:
            video_grid_thw = inputs["video_grid_thw"]
            num_videos = int(video_grid_thw.shape[0]) if torch.is_tensor(video_grid_thw) else len(video_grid_thw)
            temporal_patch_size = getattr(
                getattr(self.processor, "video_processor", None), "temporal_patch_size", 2
            )
            value = float(temporal_patch_size) / self._VIDEO_FPS
            inputs["video_second_per_grid"] = [torch.tensor(value) for _ in range(num_videos)]
        return super()._get_position_ids(inputs)

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super()._data_collator_mm_data(batch)
        # Qwen3's position-id calculation needs this per-video timeline
        # metadata when we re-encode a generated answer for mask scoring.
        video_second_per_grid = self.gather_list(batch, "video_second_per_grid")
        if video_second_per_grid:
            result["video_second_per_grid"] = video_second_per_grid
        elif result.get("video_grid_thw") is not None:
            # The native processor emits this metadata while encoding the
            # multimodal prompt, but some ms-swift versions drop it during
            # the post-encode hook.  Qwen3 needs one timeline scalar per
            # video when audio is present, so reconstruct the exact processor
            # value from the configured FPS instead of letting the model see
            # None (which fails inside get_rope_index).
            video_grid_thw = result["video_grid_thw"]
            num_videos = int(video_grid_thw.shape[0]) if torch.is_tensor(video_grid_thw) else len(video_grid_thw)
            temporal_patch_size = getattr(
                getattr(self.processor, "video_processor", None), "temporal_patch_size", 2
            )
            value = float(temporal_patch_size) / self._VIDEO_FPS
            result["video_second_per_grid"] = [torch.tensor(value) for _ in range(num_videos)]
        samples = [item for item in batch if item.get("sam_pixel_values") is not None]
        if not samples:
            return result
        if len(samples) != len(batch):
            raise ValueError("Do not mix segmentation and non-segmentation examples in one batch")
        result["sam_pixel_values"] = torch.stack([item["sam_pixel_values"] for item in samples])
        result["gt_masks"] = [item["gt_masks"] for item in samples]
        result["sam_resize_sizes"] = [item["sam_resize_sizes"] for item in samples]
        result["sam_original_sizes"] = [item["sam_original_sizes"] for item in samples]
        return result


register_template(
    TemplateMeta(
        TEMPLATE_TYPE,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
        system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
        default_system=None,
        stop_words=["<|endoftext|>"],
        agent_template="hermes",
        template_cls=AuroraQwen3OmniTemplate,
    )
)
