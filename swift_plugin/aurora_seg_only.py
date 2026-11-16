"""Freeze an AURORA PEFT model to its segmentation branch for Stage-2 SFT.

Load this file through ``swift sft --external_plugins``. It patches the
ms-swift model-preparation boundary, after the Stage-1 adapter has been
restored but before Trainer builds DDP and the optimizer. Only the active
PEFT ``modules_to_save`` copies of ``text_hidden_fcs`` and ``mask_decoder``
remain trainable.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch
from peft.utils.other import ModulesToSaveWrapper

from swift.pipelines.train.tuner import TunerMixin
from swift.trainers import Seq2SeqTrainer
from swift.trainers.trainer_factory import TrainerFactory


_ENABLED = os.environ.get("AURORA_SEGMENTATION_ONLY", "0") == "1"
_SAM2_SUFFIXES = (
    "sam_mask_decoder", "memory_attention", "memory_encoder",
    "obj_ptr_proj", "mask_downsample",
)
_TARGET_SUFFIXES = (
    ("text_hidden_fcs", *_SAM2_SUFFIXES)
    if os.environ.get("AURORA_SEG_BACKEND", "sam").lower() == "sam2"
    else ("text_hidden_fcs", "mask_decoder")
)


def _active_adapter_names(wrapper: ModulesToSaveWrapper) -> list[str]:
    active = getattr(wrapper, "active_adapter", None)
    if active is None:
        active = getattr(wrapper, "active_adapters", None)
    if isinstance(active, str):
        names = [active]
    elif active:
        names = list(active)
    else:
        names = list(wrapper.modules_to_save.keys())
    return [name for name in names if name in wrapper.modules_to_save]


def freeze_to_segmentation_branch(model: torch.nn.Module) -> dict[str, int]:
    """Freeze everything except the active projector and SAM decoder copies."""
    model.requires_grad_(False)
    found: dict[str, int] = defaultdict(int)

    for module_name, module in model.named_modules():
        if not isinstance(module, ModulesToSaveWrapper):
            continue
        target = next(
            (suffix for suffix in _TARGET_SUFFIXES if module_name.endswith(suffix)),
            None,
        )
        if target is None:
            continue
        active_names = _active_adapter_names(module)
        if not active_names:
            raise RuntimeError(
                f"No active adapter copy found for segmentation module {module_name!r}"
            )
        for adapter_name in active_names:
            active_module = module.modules_to_save[adapter_name]
            active_module.requires_grad_(True)
            found[target] += sum(p.numel() for p in active_module.parameters())

    missing = [name for name in _TARGET_SUFFIXES if found[name] == 0]
    if missing:
        raise RuntimeError(
            "Segmentation-only training requires PEFT modules_to_save wrappers for "
            f"{missing}; verify that --adapters points at the joint SFT checkpoint."
        )

    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(f"{target}.modules_to_save." in name for target in _TARGET_SUFFIXES)
    ]
    if unexpected:
        preview = ", ".join(unexpected[:10])
        raise RuntimeError(f"Unexpected trainable parameters after freezing: {preview}")

    total = sum(found.values())
    details = ", ".join(f"{name}={found[name]:,}" for name in _TARGET_SUFFIXES)
    print(f"[aurora-seg-only] trainable parameters: {details}, total={total:,}", flush=True)
    return dict(found)


def _find_aurora_losses(model: torch.nn.Module) -> dict[str, torch.Tensor] | None:
    for module in model.modules():
        losses = getattr(module, "aurora_last_losses", None)
        if losses:
            return losses
    return None


class AuroraSegOnlyTrainer(Seq2SeqTrainer):
    """SFT trainer that exposes the component mask losses in regular logs."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        parts = _find_aurora_losses(model)
        if parts:
            mode = "train" if model.training else "eval"
            for name in ("ce_loss", "mask_bce_loss", "mask_dice_loss", "mask_loss"):
                value = parts.get(name)
                if value is not None:
                    self.custom_metrics[mode][f"aurora/{name}"].update(value.detach().float())
        return result


if _ENABLED:
    _original_prepare_model = TunerMixin.prepare_model.__func__

    @classmethod
    def _prepare_seg_only_model(cls, args: Any, model, **kwargs):
        prepared = _original_prepare_model(cls, args, model, **kwargs)
        freeze_to_segmentation_branch(prepared)
        return prepared

    TunerMixin.prepare_model = _prepare_seg_only_model
    TrainerFactory.TRAINER_MAPPING["causal_lm"] = (
        "swift_plugin.aurora_seg_only.AuroraSegOnlyTrainer"
    )
