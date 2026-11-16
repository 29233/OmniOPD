from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from swift_plugin.segmentation import load_gt_masks, preprocess_sam_frames
from swift_plugin.segmentation import attach_aurora_segmentation

# ---------------------------------------------------------------------------
# ms-swift version compatibility layer.
#
# ms-swift <= 3.x exposes everything under ``swift.llm`` and registers models
# via a ``get_function`` loader callback. ms-swift >= 4.x flattened the
# package (``swift.model`` / ``swift.template`` / ``swift.dataset``) and
# switched registration to ``ModelLoader`` subclasses. Support both layouts
# from a single plugin file so the same code runs in the base env (swift
# 3.10.1, used by the existing SFT/eval scripts) and the swift46 env (swift
# 4.5.2, used for OPSD/GKD training).
# ---------------------------------------------------------------------------
try:  # ms-swift >= 4.x
    from swift.model.model_meta import Model, ModelGroup, ModelMeta
    from swift.model.model_arch import ModelArch
    from swift.model.register import register_model
    from swift.model.models.qwen import Qwen2_5OmniLoader
    from swift.template.register import register_template
    from swift.template.template_meta import TemplateMeta
    from swift.template.templates.qwen import Qwen2_5OmniTemplate
    from swift.template.template_inputs import StdTemplateInputs

    _SWIFT_MAJOR = 4
except ImportError:  # ms-swift <= 3.x
    from swift.llm import Model, ModelGroup, ModelMeta, TemplateMeta, register_model, register_template
    from swift.llm.model.model.qwen import get_model_tokenizer_qwen2_5_omni
    from swift.llm.model.model_arch import ModelArch
    from swift.llm.template.template.qwen import Qwen2_5OmniTemplate
    from swift.llm.template.template_inputs import StdTemplateInputs

    _SWIFT_MAJOR = 3


# Transformers 4.57 calls unwrap_model(..., keep_torch_compile=...), while
# the environment's Accelerate may still expose the older signature.
try:
    import accelerate
    import inspect

    if "keep_torch_compile" not in inspect.signature(accelerate.Accelerator.unwrap_model).parameters:
        _aurora_unwrap_model = accelerate.Accelerator.unwrap_model

        def _aurora_compat_unwrap_model(self, model, keep_torch_compile=False):
            return _aurora_unwrap_model(self, model)

        accelerate.Accelerator.unwrap_model = _aurora_compat_unwrap_model
except Exception:
    pass


MODEL_TYPE = "aurora_qwen2_5_omni"
TEMPLATE_TYPE = "aurora_qwen2_5_omni"


def _restore_trained_sam_weights(model, adapters):
    """Reload the Stage-1 trained SAM-side weights after attach.

    ``attach_aurora_segmentation`` builds ``text_hidden_fcs`` and the SAM
    ``mask_decoder`` wrapper with FRESH (random) parameters. When ms-swift
    loads a Stage-1/OPSD checkpoint via ``--adapters``, its
    ``modules_to_save`` restore happens BEFORE our loader attaches the SAM
    head, so the trained projector/decoder weights are silently overwritten
    by the random ones — every resumed run (OPSD, GRPO, eval) then trains or
    scores against a random SAM head (observed: mask_loss stuck at the
    ~0.63 random-head floor, IoU ~ 0). Pull those two modules' weights back
    out of the adapter safetensors here.
    """
    if not adapters:
        return
    adapter_dir = adapters[-1] if isinstance(adapters, (list, tuple)) else adapters
    state_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(state_path):
        return
    from safetensors.torch import load_file

    state = load_file(state_path)

    projector = {}
    for k, v in state.items():
        if ".text_hidden_fcs." in k:
            name = k.split(".text_hidden_fcs.")[-1]
            projector[name] = v
    if projector:
        model.text_hidden_fcs.load_state_dict(projector, strict=True)
        print(f"[aurora] restored {len(projector)} trained projector tensors "
              f"from {adapter_dir}")

    # mask_decoder on disk is the MaskDecoderSaveProxy layout
    # '...mask_decoder.decoder.<sam-decoder-key>'
    decoder = {}
    for k, v in state.items():
        if ".mask_decoder.decoder." in k:
            decoder[k.split(".mask_decoder.decoder.")[-1]] = v
    if decoder and getattr(model, "sam", None) is not None:
        model.sam.mask_decoder.decoder.load_state_dict(decoder, strict=True)
        print(f"[aurora] restored {len(decoder)} trained mask_decoder tensors "
              f"from {adapter_dir}")

    sam2_sidecar = os.path.join(adapter_dir, "sam2_video_decoder.safetensors")
    if getattr(model, "sam2_head", None) is not None and os.path.isfile(sam2_sidecar):
        sam2_state = load_file(sam2_sidecar)
        model.sam2_head.load_trainable_state_dict(sam2_state)
        print(f"[aurora] restored {len(sam2_state)} trained SAM2 video tensors from {adapter_dir}")

    # embed_tokens / lm_head are ALSO modules_to_save entries: the thinker's
    # embedding matrix (with the trained [SEG] row) must match the checkpoint
    # or the [SEG] hidden state is computed from a random embedding row and
    # the projector sees an out-of-distribution input (observed: IoU 0 ->
    # 0.2 after restoring).
    emb = state.get("base_model.model.thinker.model.embed_tokens.weight")
    lm = state.get("base_model.model.thinker.lm_head.weight")
    if emb is not None and emb.shape == model.thinker.model.embed_tokens.weight.shape:
        model.thinker.model.embed_tokens.weight.data.copy_(
            emb.to(model.thinker.model.embed_tokens.weight.device,
                   dtype=model.thinker.model.embed_tokens.weight.dtype))
        print("[aurora] restored thinker embed_tokens "
              f"({tuple(emb.shape)}) from {adapter_dir}")
    if lm is not None and lm.shape == model.thinker.lm_head.weight.shape:
        model.thinker.lm_head.weight.data.copy_(
            lm.to(model.thinker.lm_head.weight.device,
                  dtype=model.thinker.lm_head.weight.dtype))
        print("[aurora] restored thinker lm_head "
              f"({tuple(lm.shape)}) from {adapter_dir}")


def _attach_aurora(model, processor, adapters=None):
    """Attach the AURORA SAM segmentation head to a loaded Omni model."""
    if model is None:
        return
    model.aurora_processor = processor
    backend = os.environ.get("AURORA_SEG_BACKEND", "sam").lower()
    if backend == "sam2":
        from swift_plugin.sam2_segmentation import attach_aurora_sam2_segmentation

        checkpoint = os.environ.get("AURORA_SAM2_CHECKPOINT", "")
        config = os.environ.get("AURORA_SAM2_CONFIG", "configs/sam2/sam2_hiera_l.yaml")
        attach_aurora_sam2_segmentation(model, processor, checkpoint, config)
    elif backend == "sam":
        checkpoint = os.environ.get("AURORA_SAM_CHECKPOINT", "")
        attach_aurora_segmentation(model, processor, checkpoint)
    else:
        raise ValueError(f"Unsupported AURORA_SEG_BACKEND={backend!r}")
    # attach builds a RANDOM SAM head; put the trained weights back
    _restore_trained_sam_weights(model, adapters)


if _SWIFT_MAJOR >= 4:

    class AuroraQwen2_5OmniLoader(Qwen2_5OmniLoader):
        """Qwen2_5OmniLoader + AURORA SAM head attachment.

        The 4.x loader class takes over everything the old
        ``get_model_tokenizer_qwen2_5_omni`` callback did (config tweaks,
        thinker/talker wiring, processor patching); we only wrap ``load`` to
        attach the segmentation head to whatever it produces.
        """

        def __init__(self, *args, **kwargs):
            # ModelLoader.__init__ swallows unknown kwargs (like `adapters`)
            # into **kwargs and drops them; remember them for load().
            self._aurora_adapters = kwargs.pop("adapters", None)
            super().__init__(*args, **kwargs)

        def load(self):
            model, processor = super().load()
            # super().load() applied the --adapters modules_to_save weights,
            # then _attach_aurora's fresh SAM head overwrote them; pass the
            # adapter dir so the trained tensors can be restored on top.
            # NOTE: `adapters` is NOT part of ModelLoader's stored attributes
            # (it lands in __init__'s **kwargs and is dropped) nor of
            # model_kwargs — the training pipeline applies it AFTER loading
            # via tuner.from_pretrained, which is the correct order; for
            # direct get_model_processor(adapters=...) callers we fish it out
            # of the leftover kwargs captured here.
            adapters = getattr(self, "_aurora_adapters", None)
            _attach_aurora(model, processor, adapters)
            return model, processor

    _AURORA_LOADER = AuroraQwen2_5OmniLoader
else:

    def get_model_tokenizer_aurora_qwen2_5_omni(model_dir, *args, **kwargs):
        model, processor = get_model_tokenizer_qwen2_5_omni(model_dir, *args, **kwargs)
        _attach_aurora(model, processor)
        return model, processor

    _AURORA_LOADER = get_model_tokenizer_aurora_qwen2_5_omni


if _SWIFT_MAJOR >= 4:
    _MODEL_META_KWARGS = dict(
        loader=_AURORA_LOADER,
        template=TEMPLATE_TYPE,
    )
else:
    _MODEL_META_KWARGS = dict(
        get_function=_AURORA_LOADER,
        template=TEMPLATE_TYPE,
    )


register_model(
    ModelMeta(
        MODEL_TYPE,
        [ModelGroup([
            Model("Qwen/Qwen2.5-Omni-3B",
                  "Qwen/Qwen2.5-Omni-3B",
                  model_path=os.environ.get("QWEN2_5_OMNI_3B_PATH", "models/Qwen2___5-Omni-3B")),
            Model("Qwen/Qwen2.5-Omni-7B",
                  "Qwen/Qwen2.5-Omni-7B",
                  model_path=os.environ.get("QWEN2_5_OMNI_7B_PATH", "models/Qwen2___5-Omni-7B")),
        ])],
        model_arch=ModelArch.qwen2_5_omni,
        is_multimodal=True,
        architectures=["Qwen2_5OmniModel", "Qwen2_5OmniForConditionalGeneration"],
        requires=["transformers>=4.50", "soundfile", "qwen_omni_utils", "decord"],
        tags=["vision", "video", "audio"],
        additional_saved_files=["spk_dict.pt"],
        **_MODEL_META_KWARGS,
    )
)


class AuroraQwen2_5OmniTemplate(Qwen2_5OmniTemplate):
    # Frame-path lists do not carry timing metadata. The model timeline must
    # follow the selected checkpoint's training manifest, not an inference
    # made from the number of paths.
    # Read the env LAZILY (per call), not at class-definition time: the class
    # body runs whenever this module is first imported — which can happen
    # BEFORE the launcher sets its env vars if anything on the import path
    # touches the plugin early.
    @property
    def _VIDEO_FPS(self) -> float:
        return float(os.environ.get("AURORA_VIDEO_FPS", "2.0"))

    def _encode_truncated(self, inputs: StdTemplateInputs):
        """Apply the video timeline explicitly configured for this checkpoint.

        The parent's omni ``_encode`` calls ``processor(text='', audio=...,
        videos=..., ...)`` without any videos_kwargs, so the processor
        computes ``video_second_per_grid = temporal_patch_size / fps`` with
        its default ``fps=2.0``. Swift feeds that list
        into ``get_rope_index`` for the video position ids (consumed as
        ``second_per_grids[i].cpu().float()``, i.e. per-element tensors) and
        nothing else in the training path touches it, so we simply REWRITE
        the returned values here: second_per_grid = temporal_patch_size /
        AURORA_VIDEO_FPS. Wrapping the processor call instead does not work:
        instance-level ``__call__`` patching is invisible to Python's
        special-method lookup, and class-level patching is not safe across
        concurrent templates.
        """
        encoded = super()._encode_truncated(inputs)
        if not inputs.videos or self._VIDEO_FPS == 2.0:
            return encoded
        spg = encoded.get("video_second_per_grid")
        if not spg:
            return encoded
        processor = self.processor
        tps = getattr(getattr(processor, "video_processor", None), "temporal_patch_size", 2)
        correct = float(tps) / self._VIDEO_FPS
        # Per-element 0-d tensors: indexable like the original list and each
        # element supports the .cpu().float() the model applies.
        encoded["video_second_per_grid"] = [torch.tensor(correct) for _ in spg]
        return encoded

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        extras = inputs.extra_kwargs
        frame_paths = extras.get("sam_frame_paths")
        mask_paths = extras.get("mask_paths")
        # SAM supervision belongs only to the training forward. In
        # ms-swift's PtEngine generation mode (used by predict_with_generate)
        # validation rows retain these metadata fields, but encoding them
        # would perform needless image/mask preprocessing and pass extra
        # tensors through the LLM generation path.
        if not self.is_training:
            # Generation passes the encoded row's extra kwargs directly to
            # ``model.generate``. Strip AURORA-only metadata so the native
            # PtEngine path receives only model-supported arguments.
            for key in ("sam_frame_paths", "mask_paths", "class_name", "sample_id", "dataset"):
                extras.pop(key, None)
            return encoded
        if frame_paths is None and mask_paths is None:
            return encoded
        if not frame_paths or mask_paths is None:
            raise ValueError("sam_frame_paths and mask_paths must be provided together")
        num_frames = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
        if os.environ.get("AURORA_SEG_BACKEND", "sam").lower() == "sam2":
            from swift_plugin.sam2_segmentation import preprocess_sam2_frames

            sam_pixels, resize_sizes, original_sizes = preprocess_sam2_frames(frame_paths, num_frames)
        else:
            sam_pixels, resize_sizes, original_sizes = preprocess_sam_frames(frame_paths, num_frames)
        gt_masks = load_gt_masks(mask_paths, original_sizes[0], num_frames)
        encoded.update(
            sam_pixel_values=sam_pixels,
            gt_masks=gt_masks,
            sam_resize_sizes=resize_sizes,
            sam_original_sizes=original_sizes,
        )
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # The native Omni hook converts media tokens to inputs_embeds and
        # intentionally returns only model-facing keys. Preserve AURORA's
        # per-sample supervision tensors across that hook as well.
        result = super()._post_encode(model, inputs)
        for key in ("sam_pixel_values", "gt_masks", "sam_resize_sizes", "sam_original_sizes"):
            if key in inputs:
                result[key] = inputs[key]
        return result

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = super()._data_collator_mm_data(batch)
        samples = [item for item in batch if item.get("sam_pixel_values") is not None]
        if not samples:
            return result
        if len(samples) != len(batch):
            raise ValueError("Do not mix segmentation and non-segmentation examples in one batch")
        shapes = {tuple(item["gt_masks"].shape[-2:]) for item in samples}
        if len(shapes) != 1:
            raise ValueError("Variable mask sizes require per_device_train_batch_size=1")
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
        # The original AURORA Qwen2.5-Omni Stage-1 message builder has no
        # system turn. Keep system support available for explicit callers,
        # but do not inject ms-swift's default system text implicitly.
        default_system=None,
        stop_words=["<|endoftext|>"],
        agent_template="hermes",
        template_cls=AuroraQwen2_5OmniTemplate,
    )
)
