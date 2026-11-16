"""Dependency-free checkout checks suitable for a public GitHub CI runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data.relocate_media_paths import relocate
from scripts.setup.repo_doctor import ROOT, check


class RepositoryContractTests(unittest.TestCase):
    def test_required_source_and_local_documentation(self):
        report = check(run_shell_syntax=False)
        self.assertEqual(report["errors"], [])
        self.assertGreater(report["python_files_parsed"], 100)

    def test_missing_source_is_a_structural_error(self):
        with tempfile.TemporaryDirectory() as directory:
            report = check(Path(directory), run_shell_syntax=False)
            self.assertFalse(report["checkout_structurally_valid"])
            self.assertIn("Required source absent: AGENTS.md", report["errors"])

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
