"""CPU-only contracts for the standalone code checkout (not a training benchmark)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from scripts.eval.eval_omni_two_stage import _frame_metrics
from swift_plugin.sam2_grpo_rewards import class_exact_score
from swift_plugin.segmentation import load_gt_masks


class IndependentCoreTests(unittest.TestCase):
    def test_imports_resolve_inside_clone(self):
        import model.segment_anything
        import sam2
        import swift
        from swift_plugin import aurora_qwen2_5_omni, aurora_opsd, aurora_sam2_grpo
        from scripts.eval import test_qwen2_5_omni_refavs

        root = Path(__file__).resolve().parents[1]
        self.assertTrue(Path(swift.__file__).resolve().is_relative_to(root / "third_party/ms-swift"),
                        f"ms-swift must come from this checkout: {swift.__file__}")
        for module in (sam2, model.segment_anything, aurora_qwen2_5_omni,
                       aurora_opsd, aurora_sam2_grpo, test_qwen2_5_omni_refavs):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(root),
                            f"External source code imported: {module.__file__}")

    def test_class_reward_exact_and_last_decision(self):
        self.assertEqual(class_exact_score("The target is guitar.", "guitar"), 1.0)
        self.assertEqual(class_exact_score(
            "The target is guitar.\nDecision: The target is man.", "guitar"
        ), 0.0)

    def test_mask_io_and_frame_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8)).save(path)
            gt = load_gt_masks([str(path)], (2, 2), 1)
            self.assertEqual(tuple(gt.shape), (1, 2, 2))
            pred = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
            iou, dice = _frame_metrics(pred, gt[0])
            self.assertAlmostEqual(iou, 1.0)
            self.assertAlmostEqual(dice, 1.0)

    def test_training_and_test_task_style_must_match(self):
        import json
        from scripts.eval.test_qwen2_5_omni_refavs import _check_task_prompt_style

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, raw, structured = (root / name for name in
                                      ("train.jsonl", "raw.jsonl", "structured.jsonl"))
            def write(path, header):
                path.write_text(json.dumps({
                    "messages": [{"role": "user", "content":
                                  header + "<video><audio>The reference is: car."}]
                }) + "\n")
            write(train, "")
            write(raw, "")
            write(structured, "Task:\nReference analysis:\n")
            self.assertEqual(
                _check_task_prompt_style(train, {"test_s": raw})["header_chars"], 0
            )
            with self.assertRaisesRegex(ValueError, "different task prompt"):
                _check_task_prompt_style(train, {"test_s": structured})

    def test_refavs_relocation_preserves_labels_and_verifies_media(self):
        import json
        from scripts.data.relocate_media_paths import relocate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "old", root / "new"
            old.mkdir()
            (new / "media").mkdir(parents=True)
            (new / "gt_mask").mkdir()
            frames = [old / "media" / f"frame_{i}.jpg" for i in range(10)]
            masks = [old / "gt_mask" / f"mask_{i}.png" for i in range(10)]
            wav = old / "media" / "audio.wav"
            for path in [*frames, *masks, wav]:
                relocated = new / path.relative_to(old)
                relocated.parent.mkdir(parents=True, exist_ok=True)
                relocated.touch()
            source = root / "test_s.jsonl"
            output = root / "new_test_s.jsonl"
            source.write_text(json.dumps({
                "messages": [{"role": "user", "content": "<video><audio>find piano"}],
                "class_name": "piano", "sample_id": "case_piano_1",
                "videos": [[str(path) for path in frames]],
                "sam_frame_paths": [str(path) for path in frames],
                "mask_paths": [str(path) for path in masks], "audios": [str(wav)],
            }) + "\n")
            manifest = relocate(source, output, old, new)
            rewritten = json.loads(output.read_text())
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["unique_media_files_checked"], 21)
            self.assertEqual(rewritten["class_name"], "piano")
            self.assertEqual(rewritten["messages"][0]["content"], "<video><audio>find piano")
            self.assertTrue(rewritten["audios"][0].startswith(str(new)))
            self.assertTrue(output.with_suffix(".media_manifest.json").is_file())

    def test_grpo_reward_service_fails_closed(self):
        from swift_plugin import aurora_grpo

        with patch.dict(aurora_grpo._REWARD_CONTEXT, {"trainer": object()}), \
                patch.dict("os.environ", {"AURORA_GRPO_REWARD_FAIL_OPEN": "0"}), \
                patch.object(aurora_grpo, "_completion_iou",
                             side_effect=ConnectionError("offline")):
            with self.assertRaisesRegex(RuntimeError, "refusing an invalid GRPO update"):
                aurora_grpo.AuroraIoUORM()(
                    completions=["It is [SEG]."], messages=[[{"role": "user", "content": "test"}]],
                    sam_frame_paths=[["frame.jpg"]], mask_paths=[["mask.png"]], audios=[["audio.wav"]],
                )


if __name__ == "__main__":
    unittest.main()
