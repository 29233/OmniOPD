"""SAM2 video-memory segmentation backend for AURORA's projected [SEG] token."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from types import MethodType
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.build_sam import build_sam2

from swift_plugin.segmentation import SEG_TOKEN, dice_loss, load_gt_masks


SAM2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
SAM2_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TRAINABLE_SAM2_MODULES = (
    "sam_mask_decoder",
    "memory_attention",
    "memory_encoder",
    "obj_ptr_proj",
    "mask_downsample",
)


def preprocess_sam2_frames(
    frame_paths: Sequence[str], num_frames: int, image_size: int = 1024
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]:
    paths = list(frame_paths)
    if not paths:
        raise ValueError("sam_frame_paths must contain at least one frame")
    while len(paths) < num_frames:
        paths.append(paths[-1])
    paths = paths[:num_frames]
    tensors: List[torch.Tensor] = []
    original_sizes: List[Tuple[int, int]] = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_sizes.append((image.shape[0], image.shape[1]))
        # Official SAM2 video loading stretches every frame to a square.
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - SAM2_MEAN) / SAM2_STD
        tensors.append(tensor)
    return torch.stack(tensors), [(image_size, image_size)] * num_frames, original_sizes


class TextPromptEncoderProxy(nn.Module):
    """Append AURORA's text prompt to SAM2's native sparse prompt tokens."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.current_text_embeds: torch.Tensor | None = None

    @property
    def mask_input_size(self):
        return self.encoder.mask_input_size

    def get_dense_pe(self):
        return self.encoder.get_dense_pe()

    def forward(self, points=None, boxes=None, masks=None):
        sparse, dense = self.encoder(points=points, boxes=boxes, masks=masks)
        text = self.current_text_embeds
        if text is not None:
            text = text.to(device=sparse.device, dtype=sparse.dtype)
            if text.shape[0] != sparse.shape[0] or text.shape[-1] != sparse.shape[-1]:
                raise ValueError(
                    f"SAM2 text prompt mismatch: text={tuple(text.shape)}, sparse={tuple(sparse.shape)}"
                )
            sparse = torch.cat([sparse, text], dim=1)
        return sparse, dense


class SAM2MaskDecoderSaveProxy(nn.Module):
    """Make SAM2's keyword call compatible with PEFT ModulesToSaveWrapper."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    @property
    def conv_s0(self):
        return self.decoder.conv_s0

    @property
    def conv_s1(self):
        return self.decoder.conv_s1

    @staticmethod
    def _target_module(module: nn.Module) -> nn.Module:
        modules = getattr(module, "modules_to_save", None)
        if modules is not None and "default" in modules:
            module = modules["default"]
        if isinstance(module, SAM2MaskDecoderSaveProxy):
            module = module.decoder
        return module

    def forward(self, image_embeddings=None, *args, **kwargs):
        # Positional forwarding satisfies PEFT AuxiliaryTrainingWrapper; when
        # the inner module is the original proxy it restores SAM2's keyword API.
        dtype = next(self._target_module(self.decoder).parameters()).dtype
        image_embeddings = image_embeddings.to(dtype=dtype)
        for key in ("image_pe", "sparse_prompt_embeddings", "dense_prompt_embeddings"):
            if isinstance(kwargs.get(key), torch.Tensor):
                kwargs[key] = kwargs[key].to(dtype=dtype)
        if kwargs.get("high_res_features") is not None:
            kwargs["high_res_features"] = [value.to(dtype=dtype) for value in kwargs["high_res_features"]]
        return self.decoder(image_embeddings, *args, **kwargs)


class SAM2MemoryAttentionSaveProxy(nn.Module):
    """Give SAM2 memory attention a positional first argument for PEFT."""

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention

    def forward(self, curr=None, *args, **kwargs):
        target = self.attention
        modules = getattr(target, "modules_to_save", None)
        if modules is not None and "default" in modules:
            target = modules["default"]
        if isinstance(target, SAM2MemoryAttentionSaveProxy):
            target = target.attention
        dtype = next(target.parameters()).dtype
        curr = [value.to(dtype=dtype) for value in curr]
        if kwargs.get("curr_pos") is not None:
            kwargs["curr_pos"] = [value.to(dtype=dtype) for value in kwargs["curr_pos"]]
        for key in ("memory", "memory_pos"):
            if isinstance(kwargs.get(key), torch.Tensor):
                kwargs[key] = kwargs[key].to(dtype=dtype)
        return self.attention(curr, *args, **kwargs)


class SAM2MemoryEncoderSaveProxy(nn.Module):
    """Cast SAM2's float32 mask probabilities for a mixed-precision encoder."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pix_feat, masks, *args, **kwargs):
        target = self.encoder
        modules = getattr(target, "modules_to_save", None)
        if modules is not None and "default" in modules:
            target = modules["default"]
        if isinstance(target, SAM2MemoryEncoderSaveProxy):
            target = target.encoder
        dtype = next(target.parameters()).dtype
        return self.encoder(pix_feat.to(dtype=dtype), masks.to(dtype=dtype), *args, **kwargs)


class AuroraSAM2VideoHead(nn.Module):
    """Decode a ten-frame mask sequence with SAM2's recurrent memory bank."""

    def __init__(self, checkpoint: str, config: str):
        super().__init__()
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Set AURORA_SAM2_CHECKPOINT, got {checkpoint!r}")
        sam2 = build_sam2(
            config_file=config,
            ckpt_path=checkpoint,
            device="cpu",
            mode="train",
            apply_postprocessing=False,
        )
        # The pretrained object-presence head has never seen AURORA text tokens.
        # Its hard negative branch replaces masks with a constant NO_OBJ_SCORE,
        # which blocks all segmentation gradients at initialization. REF-AVS
        # provides a target mask on every supervised frame, so train continuous
        # mask logits and let BCE/Dice learn the new text interface first.
        sam2.pred_obj_scores = False
        sam2.sam_mask_decoder = SAM2MaskDecoderSaveProxy(sam2.sam_mask_decoder)
        sam2.memory_attention = SAM2MemoryAttentionSaveProxy(sam2.memory_attention)
        sam2.memory_encoder = SAM2MemoryEncoderSaveProxy(sam2.memory_encoder)
        sam2.sam_prompt_encoder = TextPromptEncoderProxy(sam2.sam_prompt_encoder)
        sam2.requires_grad_(False)
        for name in TRAINABLE_SAM2_MODULES:
            module = getattr(sam2, name, None)
            if module is not None:
                module.requires_grad_(True)
        self.sam2 = sam2
        self.config_name = config
        self.checkpoint = checkpoint

    @property
    def image_size(self) -> int:
        return int(self.sam2.image_size)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = {}
        for name in TRAINABLE_SAM2_MODULES:
            module = getattr(self.sam2, name, None)
            if module is None:
                continue
            module = self._active(module)
            for key, value in module.state_dict().items():
                state[f"sam2.{name}.{key}"] = value.detach().cpu().contiguous()
        return state

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [key for key in unexpected if key.startswith("sam2.")]
        if unexpected:
            raise RuntimeError(f"Unexpected SAM2 decoder keys: {unexpected[:10]}")
        expected = {key for key in self.state_dict() if any(
            key.startswith(f"sam2.{name}.") for name in TRAINABLE_SAM2_MODULES
        )}
        absent = sorted(expected - set(state))
        if absent:
            raise RuntimeError(f"SAM2 decoder checkpoint misses {len(absent)} keys: {absent[:10]}")

    @staticmethod
    def _active(module: nn.Module) -> nn.Module:
        # A post-PEFT outer proxy may wrap ModulesToSaveWrapper; normalize to
        # its active copied module for portable checkpoint sidecars.
        if isinstance(module, SAM2MaskDecoderSaveProxy) and hasattr(module.decoder, "modules_to_save"):
            module = module.decoder
        if isinstance(module, SAM2MemoryAttentionSaveProxy) and hasattr(module.attention, "modules_to_save"):
            module = module.attention
        if isinstance(module, SAM2MemoryEncoderSaveProxy) and hasattr(module.encoder, "modules_to_save"):
            module = module.encoder
        modules = getattr(module, "modules_to_save", None)
        if modules is not None and "default" in modules:
            return modules["default"]
        return module

    def _ensure_keyword_compatible_decoder(self) -> None:
        decoder = self.sam2.sam_mask_decoder
        if hasattr(decoder, "modules_to_save"):
            # Official SAM2 calls its decoder with image_embeddings=... while
            # PEFT AuxiliaryTrainingWrapper requires a positional first arg.
            self.sam2.sam_mask_decoder = SAM2MaskDecoderSaveProxy(decoder)
        attention = self.sam2.memory_attention
        if hasattr(attention, "modules_to_save"):
            self.sam2.memory_attention = SAM2MemoryAttentionSaveProxy(attention)

    def _forward_image(self, image: torch.Tensor):
        sam2 = self.sam2
        sam2.image_encoder.eval()
        encoder_dtype = next(sam2.image_encoder.parameters()).dtype
        context = torch.autocast("cuda", enabled=False) if image.is_cuda else nullcontext()
        with context, torch.no_grad():
            backbone_out = sam2.image_encoder(image.to(dtype=encoder_dtype))
        if sam2.use_high_res_features_in_sam:
            decoder = self._active(sam2.sam_mask_decoder)
            backbone_out["backbone_fpn"][0] = decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return sam2._prepare_backbone_features(backbone_out)

    def forward(
        self,
        images: torch.Tensor,
        text_prompt: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [T,3,H,W] SAM2 images, got {tuple(images.shape)}")
        encoded_frames = self.encode_frames(images)
        return self.decode_encoded_frames(encoded_frames, text_prompt, output_size)

    def encode_frames(self, images: torch.Tensor) -> List[Tuple]:
        """Encode video frames independently of the text prompt.

        GRPO samples several completions for the same video. Exposing this
        prompt-independent stage lets its mask reward reuse the expensive
        Hiera image features across those completions, while SFT still calls
        :meth:`forward` and keeps the same differentiable decoder path.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected [T,3,H,W] SAM2 images, got {tuple(images.shape)}")
        return [self._forward_image(images[index : index + 1]) for index in range(images.shape[0])]

    def decode_encoded_frames(
        self,
        encoded_frames: List[Tuple],
        text_prompt: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode a mask sequence from cached per-frame SAM2 features."""
        if not encoded_frames:
            raise ValueError("SAM2 requires at least one encoded frame")
        sam2 = self.sam2
        self._ensure_keyword_compatible_decoder()
        prompt_encoder = sam2.sam_prompt_encoder
        prompt_encoder.current_text_embeds = text_prompt.view(1, 1, -1)
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        masks: List[torch.Tensor] = []
        try:
            for frame_index, encoded in enumerate(encoded_frames):
                _, feats, pos, sizes = encoded
                current_out = sam2.track_step(
                    frame_idx=frame_index,
                    is_init_cond_frame=frame_index == 0,
                    current_vision_feats=feats,
                    current_vision_pos_embeds=pos,
                    feat_sizes=sizes,
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=len(encoded_frames),
                    track_in_reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=None,
                )
                target_dict = output_dict["cond_frame_outputs"] if frame_index == 0 else output_dict["non_cond_frame_outputs"]
                target_dict[frame_index] = current_out
                high_res = current_out["pred_masks_high_res"]
                high_res = F.interpolate(high_res.float(), size=output_size, mode="bilinear", align_corners=False)
                masks.append(high_res[:, 0])
        finally:
            prompt_encoder.current_text_embeds = None
        return torch.cat(masks, dim=0)


def attach_aurora_sam2_segmentation(model: nn.Module, processor: Any, checkpoint: str, config: str) -> nn.Module:
    original_forward = model.forward
    model.sam2_head = AuroraSAM2VideoHead(checkpoint, config)
    hidden_size = model.thinker.config.text_config.hidden_size
    projector_hidden_size = int(os.environ.get("AURORA_PROJECTOR_HIDDEN_SIZE", hidden_size))
    model.text_hidden_fcs = nn.Sequential(
        nn.Linear(hidden_size, projector_hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(projector_hidden_size, 256),
        nn.Dropout(0.0),
    )
    model.aurora_projector_config = {
        "schema": 2,
        "backend": "sam2_video_memory",
        "input_hidden_size": hidden_size,
        "projector_hidden_size": projector_hidden_size,
        "output_hidden_size": 256,
        "layer_norm": False,
        "model_timeline_fps": float(os.environ.get("AURORA_VIDEO_FPS", "1.0")),
        "sam_frames": int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10")),
        "sam2_config": config,
        "sam2_image_size": model.sam2_head.image_size,
        "sam2_trainable_modules": list(TRAINABLE_SAM2_MODULES),
    }
    if os.environ.get("RANK", "0") == "0":
        print("[AURORA-SAM2] projector_config=" + json.dumps(model.aurora_projector_config, sort_keys=True), flush=True)
    model.aurora_tokenizer = processor.tokenizer
    model.aurora_ce_weight = float(os.environ.get("AURORA_CE_WEIGHT", "1.0"))
    model.aurora_bce_weight = float(os.environ.get("AURORA_BCE_WEIGHT", "2.0"))
    model.aurora_dice_weight = float(os.environ.get("AURORA_DICE_WEIGHT", "0.5"))

    def aurora_sam2_forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        sam_pixel_values=None,
        gt_masks=None,
        sam_resize_sizes=None,
        sam_original_sizes=None,
        **kwargs,
    ):
        skip_hidden_states = bool(kwargs.pop("aurora_skip_hidden_states", False))
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        language_labels = labels if self.aurora_ce_weight != 0 else None
        outputs = original_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=language_labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        if sam_pixel_values is None or labels is None:
            if skip_hidden_states:
                outputs.hidden_states = None
            return outputs
        seg_id = self.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        seg_mask = labels.eq(seg_id)
        counts = seg_mask.sum(dim=-1)
        if not torch.all(counts == 1):
            raise ValueError(f"Each SFT answer must contain exactly one [SEG], got {counts.tolist()}")
        hidden = outputs.hidden_states[-1]
        projector_dtype = next(self.text_hidden_fcs.parameters()).dtype
        seg_hidden = hidden[seg_mask.to(hidden.device)]
        prompts = self.text_hidden_fcs(seg_hidden.to(projector_dtype))
        # Evaluation can persist the exact language state and 256-d prompt
        # consumed by SAM2 without retaining the full layer stack.
        self.aurora_last_seg_hidden = seg_hidden.detach()
        self.aurora_last_seg_prompts = prompts.detach()
        images = sam_pixel_values.to(hidden.device)
        predicted_masks = []
        bce = hidden.new_zeros((), dtype=torch.float32)
        dice = hidden.new_zeros((), dtype=torch.float32)
        for batch_index in range(images.shape[0]):
            target = gt_masks[batch_index].to(device=hidden.device, dtype=torch.float32)
            prediction = self.sam2_head(images[batch_index], prompts[batch_index], tuple(target.shape[-2:]))
            if prediction.shape != target.shape:
                raise ValueError(f"SAM2 target mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
            bce = bce + F.binary_cross_entropy_with_logits(prediction, target)
            dice = dice + dice_loss(prediction, target)
            predicted_masks.append(prediction)
        bce = bce / images.shape[0]
        dice = dice / images.shape[0]
        ce = outputs.loss.float() if outputs.loss is not None else hidden.new_zeros((), dtype=torch.float32)
        mask_loss = self.aurora_bce_weight * bce + self.aurora_dice_weight * dice
        outputs.loss = self.aurora_ce_weight * ce + mask_loss
        outputs.hidden_states = None
        self.aurora_last_mask_loss = mask_loss
        self.aurora_last_losses = {
            "ce_loss": ce.detach(), "mask_bce_loss": bce.detach(), "mask_dice_loss": dice.detach(),
            "mask_loss": mask_loss.detach(), "loss": outputs.loss.detach(),
        }
        self.aurora_last_pred_masks = [mask.detach() for mask in predicted_masks]
        return outputs

    model.forward = MethodType(aurora_sam2_forward, model)
    return model
