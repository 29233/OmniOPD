"""Verifier-only rewards and conservative Decision-category token masks."""
from __future__ import annotations

import math
import re

import torch

from swift_plugin.sam2_grpo_rewards import _TARGET_PATTERN


def mask_primary_rewards(raw: torch.Tensor, valid: torch.Tensor, alpha: float) -> torch.Tensor:
    """Native reward_weights remain [.5,.5]; these columns yield alpha*C+(1-alpha)*I."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Class reward alpha must be in [0,1]")
    if raw.ndim != 2 or raw.shape[1] != 2 or valid.shape != raw.shape[:1]:
        raise ValueError("Expected raw [B,2] class/IoU and valid [B]")
    if not bool(torch.isfinite(raw).all()) or bool(((raw < 0) | (raw > 1)).any()):
        raise ValueError("Raw class/IoU rewards must be finite and in [0,1]")
    factors = raw.new_tensor([2 * alpha, 2 * (1 - alpha)])
    return raw * valid[:, None].to(raw.dtype) * factors


def valid_seg_text(text: str) -> bool:
    # The reward decoder is fail-closed. Missing/truncated SEG is otherwise
    # indistinguishable from a genuine zero-overlap mask in the raw IoU column.
    return str(text).count("[SEG]") == 1


def decision_category_mask(pieces: list[str], completion_mask: torch.Tensor) -> torch.Tensor:
    """Select only BPE tokens overlapping the single Decision category span.

    Header/boilerplate, punctuation-only tokens, evidence, [SEG] and EOS are
    excluded. A BPE token may contain a leading space or boundary punctuation.
    Ambiguous/multiple Decisions or inconsistent reconstructions fail closed.
    """
    result = torch.zeros_like(completion_mask, dtype=torch.bool)
    positions = torch.nonzero(completion_mask, as_tuple=False).flatten()
    if abs(len(pieces) - len(positions)) > 2:
        raise ValueError("Decision response/completion positions mismatch")
    text = "".join(pieces)
    headers = list(re.finditer(r"(?m)^Decision:\s*", text))
    if len(headers) != 1:
        return result
    start = headers[0].end()
    end_match = re.search(r"(?m)^[A-Za-z][A-Za-z -]*:", text[start:])
    end = start + end_match.start() if end_match else len(text)
    matches = list(_TARGET_PATTERN.finditer(text, start, end))
    if len(matches) != 1:
        return result
    left, right = matches[0].span(1)
    cursor = 0
    for i, piece in enumerate(pieces[:len(positions)]):
        next_cursor = cursor + len(piece)
        if max(cursor, left) < min(next_cursor, right) and "[SEG]" not in piece:
            result[positions[i]] = True
        cursor = next_cursor
    return result


def decision_license_valid(student_class, student_iou, teacher_class, teacher_iou,
                           valid, min_iou=.5, tolerance=.01):
    """Decision-only teachers must rescue class without a material mask regression."""
    if not (0 <= min_iou <= 1 and 0 <= tolerance <= 1):
        raise ValueError("Invalid minimum teacher IoU or regression tolerance")
    return (valid.bool() & teacher_class.bool() & ~student_class.bool()[:, None]
            & (teacher_iou >= min_iou)
            & (teacher_iou >= student_iou[:, None] - tolerance))
