#!/usr/bin/env python3
"""Stdlib-only checkout audit; no model weights, dataset, CUDA or secrets needed."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "AGENTS.md",
    "README.md",
    "COLLABORATION_HANDOFF.md",
    "CONTRIBUTING.md",
    "LOCAL_PACKAGE_GUIDE.md",
    ".env.example",
    ".github/workflows/source-contract.yml",
    ".github/pull_request_template.md",
    "swift_plugin/aurora_qwen2_5_omni.py",
    "swift_plugin/sam2_segmentation.py",
    "swift_plugin/aurora_sam2_grpo.py",
    "swift_plugin/aurora_online_verified_opd.py",
    "scripts/data/prepare_sft_distributed_padding.py",
    "scripts/data/prepare_sam2_grpo_data.py",
    "scripts/data/prepare_student_only_opd_data.py",
    "scripts/data/relocate_media_paths.py",
    "scripts/sft/train_merged_sam2_sft_3epoch_8gpu.sh",
    "scripts/grpo/run_sam2_verified_reward_grpo_8gpu.sh",
    "scripts/opd/run_current_policy_opd_8gpu.sh",
    "scripts/eval/test_qwen2_5_omni_refavs.py",
    "scripts/setup/bootstrap_ms_swift.sh",
    "scripts/setup/ms-swift-v4.5.2-teacher-adapters.patch",
    "scripts/setup/requirements_swift_observed_20261117.txt",
    "scripts/setup/requirements_mopd_overlay_observed_20261116.txt",
    "configs/deepspeed_zero2_fp16_v100.json",
    "model/segment_anything/__init__.py",
    "model/segment_anything/LICENSE",
    "third_party/sam2/sam2/build_sam.py",
    "third_party/sam2/LICENSE",
    "third_party/MOPD-Router-Omni/LICENSE",
    "third_party/MOPD-Router-Omni/verl/LICENSE",
    "third_party/MOPD-Router-Omni/verl/verl/models/__init__.py",
)
SOURCE_DIRS = ("swift_plugin", "scripts", "model/segment_anything",
               "third_party/sam2/sam2", "third_party/MOPD-Router-Omni/omni_mopd",
               "third_party/MOPD-Router-Omni/verl/verl/models")


def check(root: Path = ROOT, *, run_shell_syntax: bool = True) -> dict:
    root = Path(root)
    errors, warnings, infos = [], [], []
    missing = [path for path in REQUIRED if not (root / path).is_file()]
    errors.extend(f"Required source absent: {name}" for name in missing)

    count = 0
    for directory in SOURCE_DIRS:
        if not (root / directory).is_dir():
            continue
        for source in (root / directory).rglob("*.py"):
            if "__pycache__" in source.parts or "third_party/ms-swift" in str(source):
                continue
            try:
                ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Python syntax: {source.relative_to(root)}: {exc}")
            count += 1
    if run_shell_syntax:
        for path in sorted((root / "scripts").rglob("*.sh")):
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f"Shell syntax: {path.relative_to(root)}: {result.stderr.strip()[:250]}")
        launcher = root / "third_party/MOPD-Router-Omni/run_omni.sh"
        if launcher.is_file():
            result = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f"Shell syntax: {launcher.relative_to(root)}: {result.stderr.strip()[:250]}")

    for document in ("README.md", "COLLABORATION_HANDOFF.md", "LOCAL_PACKAGE_GUIDE.md"):
        path = root / document
        if not path.is_file():
            continue
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if not (path.parent / link.split("#", 1)[0]).exists():
                errors.append(f"Broken local link in {document}: {link}")
    if not (root / "LICENSE").is_file():
        warnings.append("No project-root LICENSE: owner must decide publication terms")
    if not (root / "third_party/ms-swift").is_dir():
        infos.append("ms-swift intentionally not bundled: run scripts/setup/bootstrap_ms_swift.sh")
    else:
        infos.append("Local third_party/ms-swift checkout is ignored; do not depend on it in Git")
    for name in ("prepare_merged_sft_dataset.py", "prepare_sam2_sft_initializer.py",
                 "generate_refavs_structured_cot.py"):
        if not (root / "scripts/data" / name).is_file():
            warnings.append(f"Historical data synthesis source not bundled: {name}")

    git_commit = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            git_commit = result.stdout.strip()
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
        )
        if tracked.returncode == 0:
            for raw in tracked.stdout.split(b"\0"):
                if not raw:
                    continue
                path = raw.decode("utf-8", errors="replace")
                name = Path(path).name.lower()
                if (Path(path).suffix.lower() in {
                    ".safetensors", ".pt", ".pth", ".ckpt", ".wav", ".mp3", ".mp4", ".pem", ".key"
                } or name in {".env", "credentials.json", "secrets.json"}
                        or path.startswith(("outputs/", "third_party/ms-swift/"))):
                    errors.append(f"Do not track weights/media/credentials or environment checkout: {path}")
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                 "--untracked-files=normal"],
                                capture_output=True, text=True)
        if status.returncode == 0 and status.stdout.strip():
            warnings.append("Working tree contains uncommitted changes; a fresh clone is not this tested code")
    return {
        "schema": 1, "root": str(root.resolve()), "git_commit": git_commit,
        "python_files_parsed": count, "required_files": len(REQUIRED),
        "errors": errors, "warnings": warnings, "info": infos,
        "checkout_structurally_valid": not errors,
        "public_release_ready": not errors and not warnings,
        "note": "Weights, REF-AVS media, dataset/model rights, and test split integrity require separate checks",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Nonzero exit if required source/scripts/links are broken")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = check()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.strict and report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
