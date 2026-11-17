"""Dependency-free checkout checks suitable for a public GitHub CI runner."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.data.relocate_media_paths import relocate
from scripts.setup.repo_doctor import (
    REQUIRED,
    ROOT,
    check,
    find_non_overridable_launcher_assignments,
)


class RepositoryContractTests(unittest.TestCase):
    def test_required_source_and_local_documentation(self):
        report = check(run_shell_syntax=True)
        self.assertEqual(report["errors"], [])
        self.assertGreater(report["python_files_parsed"], 100)

    def test_missing_source_is_a_structural_error(self):
        with tempfile.TemporaryDirectory() as directory:
            report = check(Path(directory), run_shell_syntax=False)
            self.assertFalse(report["checkout_structurally_valid"])
            self.assertIn("Required source absent: AGENTS.md", report["errors"])

    def test_repo_doctor_strict_catches_compile_errors_missed_by_ast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel in REQUIRED:
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
            bad = root / "scripts/eval/bad_future.py"
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_text(
                "status#!/usr/bin/env python3\nfrom __future__ import annotations\n",
                encoding="utf-8",
            )
            report = check(root, run_shell_syntax=False)
            self.assertTrue(
                any("bad_future.py" in err and "from __future__" in err for err in report["errors"]),
                report["errors"],
            )
            proc = subprocess.run(
                [sys.executable, "-S", str(ROOT / "scripts/setup/repo_doctor.py"), "--strict", "--root", str(root)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("bad_future.py", proc.stdout)

    def test_launchers_allow_env_overrides_and_reject_unconditional_literals(self):
        self.assertEqual(find_non_overridable_launcher_assignments(ROOT), [])
        forbidden = re.compile(
            r"^\s*(?:export\s+)?(SFT_CHECKPOINT|DATA_ROOT|MODEL|SAM2_CHECKPOINT)=/"
        )
        launchers = []
        for subdir in ("sft", "grpo", "opd", "eval"):
            launchers.extend(sorted((ROOT / "scripts" / subdir).glob("*.sh")))
        launchers.append(ROOT / "third_party/MOPD-Router-Omni/run_omni.sh")
        for launcher in launchers:
            for lineno, line in enumerate(launcher.read_text(encoding="utf-8").splitlines(), 1):
                self.assertIsNone(
                    forbidden.match(line),
                    f"Unconditional path assignment in {launcher.relative_to(ROOT)}:{lineno}: {line}",
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_sh = root / "scripts/opd/bad.sh"
            bad_sh.parent.mkdir(parents=True, exist_ok=True)
            bad_sh.write_text(
                '#!/usr/bin/env bash\nexport SFT_CHECKPOINT=/hardcoded/ckpt\n',
                encoding="utf-8",
            )
            violations = find_non_overridable_launcher_assignments(root)
            self.assertEqual(len(violations), 1)
            self.assertIn("SFT_CHECKPOINT=/hardcoded/ckpt", violations[0])

    def test_env_example_documents_portable_keys_without_host_paths(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("/mnt/", text)
        self.assertNotIn("/root/", text)
        for key in (
            "MODEL",
            "SAM2_CHECKPOINT",
            "INIT_ADAPTER",
            "DATA_DIR",
            "SFT_CHECKPOINT",
            "DATA_ROOT",
            "DATASET_AUDIT",
            "TRAIN_PROMPT_SOURCE",
            "PYTHON_BIN",
            "SWIFT_BIN",
        ):
            self.assertRegex(text, rf"(?m)^{key}=")

    def test_media_relocation_is_fail_closed_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "old"
            after = root / "new"
            before.mkdir()
            after.mkdir()
            frames = [before / f"frame_{index}.jpg" for index in range(10)]
            masks = [before / f"mask_{index}.png" for index in range(10)]
            audio = before / "audio.wav"
            for original in [*frames, *masks, audio]:
                (after / original.name).touch()
            source = root / "train.jsonl"
            row = {
                "sample_id": "clip_piano_1", "class_name": "piano",
                "messages": [{"role": "user", "content": "<video><audio>piano"}],
                "videos": [[str(path) for path in frames]],
                "sam_frame_paths": [str(path) for path in frames],
                "mask_paths": [str(path) for path in masks],
                "audios": [str(audio)],
            }
            source.write_text(json.dumps(row) + "\n")
            output = root / "relocated.jsonl"
            result = relocate(source, output, before, after)
            reread = json.loads(output.read_text())
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["unique_media_files_checked"], 21)
            self.assertEqual(reread["messages"], row["messages"])
            self.assertEqual(reread["class_name"], "piano")
            self.assertTrue(reread["audios"][0].startswith(str(after)))
            with self.assertRaises(ValueError):
                relocate(source, output, before, after)
            bad = dict(row, audios=["/another/absolute/audio.wav"])
            source.write_text(json.dumps(bad) + "\n")
            other_output = root / "bad.jsonl"
            with self.assertRaisesRegex(ValueError, "Unexpected media root"):
                relocate(source, other_output, before, after)
            self.assertFalse(other_output.exists())

    def test_repo_root_is_not_a_generated_output(self):
        self.assertTrue((ROOT / "AGENTS.md").is_file())
        self.assertFalse(str(ROOT).endswith("/outputs"))


if __name__ == "__main__":
    unittest.main()
