#!/usr/bin/env python3
"""Build an OPSD dataset (teacher_prompt column) from Stage-1 SFT JSONL.

Each row keeps its original ``messages`` (student view: plain referring
expression) and gains a ``teacher_prompt`` column — the full user content with
the privileged information (GT class name) injected. ms-swift 4.x's OPSD path
(``OnPolicySample.build_teacher_view``) replaces the last user message with
``teacher_prompt`` verbatim, so this column MUST carry the complete user turn
including the ``<video><audio>`` media placeholders — not just the privileged
sentence.

Privileged-information design (validated by scripts/probe_opsd_privilege.py on
test_s, 200 cases):
  - class privilege: mIoU 0.358 -> 0.651, negative control (wrong_class) passes
    (0.274, model rationalizes rather than parrots)
  - cot privilege: dead ([SEG] emission rate 3% — the model "consumes" [SEG]
    while reading the reference solution), so it is NOT emitted here

Three teacher privileges are supported:

``class``
    Append the target class name, matching the original OPSD experiment.
``first-mask-image``
    Add an ``<image>`` placeholder and describe it as the target object's GT
    mask in the first video frame. The AURORA OPSD plugin supplies
    ``mask_paths[0]`` to this placeholder on the teacher side only.
``audio-degrade``
    Swap the student's audio for a noise-degraded copy: the row's ``audios``
    is rewritten to the degraded file (student rollout + training encoding +
    eval all hear it automatically) and a ``teacher_audio_paths`` column keeps
    the ORIGINAL audio for the teacher. The AURORA OPSD plugin
    (``AURORA_OPSD_TEACHER_AUDIO=original``) swaps it in on the teacher side
    only. ``teacher_prompt`` stays IDENTICAL to the student user content — the
    audio-degrade privilege is purely acoustic, and the
    ``teacher_audio_paths`` column is both the teacher-side source and the
    dataset-side marker the plugin's consistency guard checks.
    Degraded files come from scripts/degrade_refavs_audio.py (run it first;
    the ``--degraded-audio-name`` default here mirrors its naming convention).
``class-audio-degrade``
    Student hears degraded audio. A single frozen teacher hears original audio
    and receives the GT class prompt. This preserves the proven class-OPSD
    objective while testing audio robustness without multi-teacher routing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def class_from_sample_id(sample_id: str) -> str:
    """REFAVS sample_id format: ``<video_id>_<class>_<idx>``.

    The video id itself contains underscores, so parse from the right: drop the
    trailing index, then the trailing segment is the class name. Same parsing
    as the probe / eval scripts (test scripts use rsplit('_'))."""
    return sample_id.rsplit('_', 1)[0].rsplit('_', 1)[-1]


def has_valid_class_privilege(gt_class: str) -> bool:
    """Return whether a category is useful as privileged information."""
    return (
        re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]*", gt_class) is not None
        and gt_class.lower() not in {"null", "none", "unknown", "na"}
    )


def build_teacher_prompt(user_content: str, gt_class: str, privilege_kind: str = "class") -> str:
    """Inject the selected teacher privilege into a complete user turn.

    The class privilege is a short, explicit teacher-only field appended after
    the referring task. It contains no extra generation-style instruction.

    ``audio-degrade`` returns the user content UNCHANGED: the privilege is
    acoustic (teacher hears the original audio), so the text must not differ —
    the dataset-side marker is the ``teacher_audio_paths`` column instead.
    """
    if privilege_kind in {"class", "class-audio-degrade"}:
        privilege = f"\nHint: The target object is: {gt_class}."
        # user_content ends with the referring expression; append, do not rewrite.
        return user_content.rstrip() + privilege
    if privilege_kind == "first-mask-image":
        marker = "<video><audio>"
        prompt = user_content.rstrip()
        if marker in prompt:
            prompt = prompt.replace(marker, marker + "<image>", 1)
        else:
            prompt = "<image>" + prompt
        return prompt + (
            "\n(The additional image is the ground-truth segmentation mask of "
            "the target object in the first video frame. Use this privileged "
            "cue to solve the task with your own reasoning.)"
        )
    if privilege_kind == "audio-degrade":
        return user_content
    raise ValueError(f"Unsupported privilege kind: {privilege_kind}")


def convert_file(
    src: str,
    dst: str,
    max_samples: int = -1,
    privilege_kind: str = "class",
    degraded_audio_name: str = "audio_snr10db.wav",
) -> dict:
    n_in = n_out = n_bad = n_without_class_privilege = 0
    classes = set()
    missing: list[str] = []
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_in += 1
            if max_samples > 0 and n_out >= max_samples:
                break
            messages = row.get("messages") or []
            user = next((m for m in messages if m.get("role") == "user"), None)
            sample_id = row.get("sample_id") or ""
            if user is None or not sample_id:
                n_bad += 1
                continue
            gt_class = (row.get("class_name") or "").strip() or class_from_sample_id(sample_id)
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9-]*$", gt_class):
                n_bad += 1
                continue

            if privilege_kind in {"audio-degrade", "class-audio-degrade"}:
                audios = row.get("audios") or []
                if not audios:
                    n_bad += 1
                    continue
                orig_audio = Path(audios[0])
                degraded = orig_audio.parent / degraded_audio_name
                # HARD failure on missing files: silently falling back to the
                # original audio would turn this into an unmarked no-privilege
                # run that is only discovered days later in eval.
                if not orig_audio.is_file():
                    missing.append(f"{sample_id}: original audio missing: {orig_audio}")
                    continue
                if not degraded.is_file():
                    missing.append(
                        f"{sample_id}: degraded audio missing: {degraded} "
                        f"(run scripts/degrade_refavs_audio.py first)")
                    continue
                # Student side: rollout + training encode + eval all read the
                # row's audios, so rewriting it degrades the whole student path.
                row["audios"] = [str(degraded)]
                # Teacher side: plugin's to_teacher_template_dict patch swaps
                # these in (AURORA_OPSD_TEACHER_AUDIO=original).
                row["teacher_audio_paths"] = [str(orig_audio)]

            if privilege_kind in {"class", "class-audio-degrade"} and not has_valid_class_privilege(gt_class):
                # Keep negative/null samples for segmentation training, but do
                # not turn a missing category into the bogus hint
                # "Hint: The target object is: null".
                row["teacher_prompt"] = user["content"]
                n_without_class_privilege += 1
            else:
                row["teacher_prompt"] = build_teacher_prompt(user["content"], gt_class, privilege_kind)
                classes.add(gt_class)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    if missing:
        print(f"[error] {len(missing)} rows with missing audio "
              f"({privilege_kind}); first 10:", file=sys.stderr)
        for entry in missing[:10]:
            print(f"  {entry}", file=sys.stderr)
        sys.exit(1)

    return {
        "input": n_in,
        "output": n_out,
        "skipped": n_bad,
        "classes": len(classes),
        "without_class_privilege": n_without_class_privilege,
    }


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = os.path.join(root, "outputs", "data", "refavs")
    parser.add_argument("--src", default=os.path.join(data_dir, "refavs_full.jsonl"))
    parser.add_argument("--dst", default=os.path.join(data_dir, "refavs_full_opsd.jsonl"))
    parser.add_argument("--val-src", default=os.path.join(data_dir, "refavs_val.jsonl"))
    parser.add_argument("--val-dst", default=os.path.join(data_dir, "refavs_val_opsd.jsonl"))
    parser.add_argument("--max-samples", type=int, default=-1, help="cap output rows (smoke tests)")
    parser.add_argument(
        "--privilege-kind",
        choices=["class", "first-mask-image", "audio-degrade", "class-audio-degrade"],
        default="class",
        help="teacher-only privileged information",
    )
    parser.add_argument(
        "--degraded-audio-name",
        default="audio_snr10db.wav",
        help="degraded audio filename (audio-degrade only); must match "
             "scripts/degrade_refavs_audio.py's naming (audio_snr{N}db.wav)",
    )
    args = parser.parse_args()

    for src, dst in [(args.src, args.dst), (args.val_src, args.val_dst)]:
        if not os.path.exists(src):
            print(f"[skip] {src} not found", file=sys.stderr)
            continue
        stats = convert_file(src, dst, args.max_samples, args.privilege_kind, args.degraded_audio_name)
        print(f"[ok] {src} -> {dst}")
        print(f"     rows: {stats['output']}/{stats['input']} (skipped {stats['skipped']}), classes: {stats['classes']}")
        if stats["without_class_privilege"]:
            print(f"     null/unknown rows kept without class privilege: {stats['without_class_privilege']}")


if __name__ == "__main__":
    main()
