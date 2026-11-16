"""Dependency-light reward helpers for the SAM2 GRPO experiment."""

from __future__ import annotations

import re
from typing import Optional


_TARGET_PATTERN = re.compile(
    r"\bthe\s+target\s+is\s+(?:the\s+)?[\"']?"
    r"([a-z0-9]+(?:[ _-][a-z0-9]+)*)"
    r"[\"']?(?=\s*(?:[.,;:!?\n\r]|$))",
    re.IGNORECASE,
)


def class_name_from_sample_id(sample_id: str) -> str:
    """Recover REF-AVS' category field from ``<clip>_<class>_<instance>``."""
    parts = str(sample_id or "").rsplit("_", 2)
    return parts[-2] if len(parts) == 3 else ""


def canonical_class_name(value: object) -> str:
    """Normalize formatting variants without accepting semantic synonyms."""
    text = str(value or "").strip().lower().strip("\"'`.,;:!?")
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text)


def extract_target_class(completion: object) -> Optional[str]:
    """Return the last ``The target is ...`` prediction in a completion."""
    matches = list(_TARGET_PATTERN.finditer(str(completion or "")))
    if not matches:
        return None
    value = canonical_class_name(matches[-1].group(1))
    return value or None


def class_exact_score(
    completion: object,
    class_name: object = "",
    sample_id: object = "",
) -> float:
    """Binary category reward requested for GRPO: exact normalized match."""
    target = canonical_class_name(class_name)
    if not target:
        target = canonical_class_name(class_name_from_sample_id(str(sample_id or "")))
    prediction = extract_target_class(completion)
    return float(bool(target) and prediction == target)
