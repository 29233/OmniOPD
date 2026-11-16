"""Thinker-only model contract. No AURORA, SAM, or historical trainer imports."""
from __future__ import annotations

from types import SimpleNamespace

import torch


def is_omni(config):
    return getattr(config, "model_type", "") == "qwen2_5_omni_thinker"


def language_parameters_only(model, *, role):
    """Freeze both perception towers, embeddings/head; optimize decoder only."""
    forbidden = ("sam", "text_hidden_fcs", "mask_decoder", "talker", "token2wav")
    if any(any(x in name.lower() for x in forbidden) for name, _ in model.named_parameters()):
        raise ValueError("Reasoning-only model unexpectedly contains segmentation/speech heads")
    names = []
    for name, param in model.named_parameters():
        enabled = role == "actor" and (
            name.startswith("model.layers.") or name.startswith("model.norm.")
        )
        param.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if role == "actor" and not names:
        raise ValueError("No native Omni language decoder parameters selected")
    return names


def rope_positions(config, encoded):
    """Invoke HF's actual Omni mRoPE implementation without loading weights."""
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration as Thinker

    helper = SimpleNamespace(
        config=config, spatial_merge_size=config.vision_config.spatial_merge_size
    )
    for name in ("get_llm_pos_ids_for_vision", "get_chunked_index"):
        setattr(helper, name, getattr(Thinker, name).__get__(helper))
    audio_lengths = encoded["feature_attention_mask"].sum(-1)
    return Thinker.get_rope_index(
        helper,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        video_grid_thw=encoded["video_grid_thw"],
        audio_seqlens=audio_lengths,
        second_per_grids=encoded["video_second_per_grid"],
        use_audio_in_video=False,
    )[0]


def extend_positions(prompt_positions, response_length):
    """[B,3,P] -> [B,3,P+T], continuing after the last valid prompt text."""
    if prompt_positions.ndim != 3 or prompt_positions.shape[1] != 3:
        raise ValueError("Expected Omni [B,3,P] mRoPE positions")
    offset = torch.arange(
        1, response_length + 1, device=prompt_positions.device
    ).view(1, 1, -1)
    tail = prompt_positions[:, :, -1:] + offset
    return torch.cat([prompt_positions, tail], dim=-1)


def assert_shared_vocabulary(student, teacher):
    if (
        student.get_vocab() != teacher.get_vocab()
        or student.eos_token_id != teacher.eos_token_id
        or student.pad_token_id != teacher.pad_token_id
    ):
        raise ValueError("Omni Student/teachers must have exactly the same token-ID mapping")
