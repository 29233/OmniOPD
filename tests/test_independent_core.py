"""CPU-only contracts for the standalone code checkout (not a training benchmark)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
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

    def test_omni_eval_settings_cli_generates_audit_json(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            for split, clip_name in (("test_s", "clipA_0_10000"), ("test_u", "clipB_0_10000")):
                clip_dir = tmp_path / "REFAVS" / "media" / clip_name
                frames_dir = clip_dir / "frames"
                frames_dir.mkdir(parents=True)
                frames = []
                for frame_idx in range(10):
                    frame_path = frames_dir / f"{frame_idx}.jpg"
                    frame_path.touch()
                    frames.append(str(frame_path))
                wav_path = clip_dir / "audio.wav"
                with wave.open(str(wav_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(8000)
                    wav_file.writeframes(b"\x00\x00" * 80000)
                (data_dir / f"refavs_{split}.jsonl").write_text(
                    json.dumps({"videos": [frames], "audios": [str(wav_path)]}) + "\n",
                    encoding="utf-8",
                )
            output_path = tmp_path / "nested/artifacts/audit.json"
            clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            proc = subprocess.run(
                [
                    "python3",
                    "-S",
                    str(root / "scripts/eval/omni_eval_settings.py"),
                    "--data-dir",
                    str(data_dir),
                    "--splits",
                    "test_s",
                    "test_u",
                    "--output",
                    str(output_path),
                ],
                cwd="/tmp",
                env=clean_env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(output_path.is_file())
            written = json.loads(output_path.read_text(encoding="utf-8"))
            printed = json.loads(proc.stdout)
            self.assertEqual(written, printed)
            self.assertEqual(written["nominal_fps"], 1.0)
            self.assertEqual(written["unique_clips_checked"], 2)
            self.assertEqual(written["rows_checked"], 2)
            self.assertEqual(written["audio_duration_range_sec"], [10.0, 10.0])

    def test_qwen2_5_omni_add_import_paths_and_sam2_backend_checks(self):
        from scripts.eval.test_qwen2_5_omni_refavs import (
            _add_import_paths,
            _configure_projector_environment,
            _repo_root,
            _resolve_sam2_checkpoint,
        )

        root = _repo_root()
        sam2_root = root / "third_party/sam2"
        stripped = [p for p in sys.path if p not in {str(root), str(sam2_root)}]
        with patch.object(sys, "path", list(stripped)):
            _add_import_paths()
            self.assertIn(str(root), sys.path[:2])
            self.assertIn(str(sam2_root), sys.path[:2])
            length_after_first = len(sys.path)
            _add_import_paths()
            self.assertEqual(len(sys.path), length_after_first)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_dir = tmp_path / "checkpoint-10"
            ckpt_dir.mkdir()
            (ckpt_dir / "aurora_projector_config.json").write_text(
                json.dumps({
                    "backend": "sam2_video_memory",
                    "projector_hidden_size": 1024,
                    "layer_norm": False,
                    "model_timeline_fps": 1.0,
                    "sam_frames": 10,
                    "sam2_config": "configs/sam2/sam2_hiera_l.yaml",
                }),
                encoding="utf-8",
            )
            env: dict[str, str] = {}
            _configure_projector_environment(ckpt_dir, env)
            self.assertEqual(env["AURORA_SEG_BACKEND"], "sam2")
            self.assertEqual(env["AURORA_PROJECTOR_HIDDEN_SIZE"], "1024")
            self.assertEqual(env["AURORA_SAM2_CONFIG"], "configs/sam2/sam2_hiera_l.yaml")

            with self.assertRaisesRegex(ValueError, "AURORA_SEG_BACKEND"):
                _configure_projector_environment(ckpt_dir, {"AURORA_SEG_BACKEND": "sam"})

            with self.assertRaisesRegex(FileNotFoundError, "AURORA_SAM2_CHECKPOINT"):
                _resolve_sam2_checkpoint({"AURORA_SEG_BACKEND": "sam2"})

            with self.assertRaisesRegex(FileNotFoundError, "missing file"):
                _resolve_sam2_checkpoint({
                    "AURORA_SEG_BACKEND": "sam2",
                    "AURORA_SAM2_CHECKPOINT": str(tmp_path / "does_not_exist.pt"),
                })

            sam2_file = tmp_path / "sam2_hiera_large.pt"
            sam2_file.write_bytes(b"weights")
            fallback_env = {"AURORA_SEG_BACKEND": "sam2", "SAM2_CHECKPOINT": str(sam2_file)}
            resolved = _resolve_sam2_checkpoint(fallback_env)
            self.assertEqual(resolved, sam2_file)
            self.assertEqual(fallback_env["AURORA_SAM2_CHECKPOINT"], str(sam2_file))

    def test_sam_iou_server_projector_hidden_size_contract(self):
        from scripts.grpo.sam_iou_server import _build_projector_from_state

        proj_3b = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 256),
            torch.nn.Dropout(0.0),
        )
        built_3b = _build_projector_from_state(proj_3b.state_dict(), projector_hidden_size=2048)
        self.assertEqual(built_3b[0].in_features, 2048)
        self.assertEqual(built_3b[0].out_features, 2048)

        proj_7b = torch.nn.Sequential(
            torch.nn.Linear(3584, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 256),
            torch.nn.Dropout(0.0),
        )
        built_7b = _build_projector_from_state(proj_7b.state_dict(), projector_hidden_size=1024)
        self.assertEqual(built_7b[0].in_features, 3584)
        self.assertEqual(built_7b[0].out_features, 1024)

        with self.assertRaisesRegex(ValueError, "projector-hidden-size 1024"):
            _build_projector_from_state(proj_7b.state_dict(), projector_hidden_size=2048)

    def test_native_eval_relocated_source_provenance_and_audit(self):
        root = Path(__file__).resolve().parents[1]
        mopd_root = str(root / "third_party/MOPD-Router-Omni")
        added = mopd_root not in sys.path
        if added:
            sys.path.insert(0, mopd_root)
        try:
            from omni_mopd import eval_official_language
            from scripts.data.relocate_media_paths import relocate

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                audit_path = tmp_path / "audit.json"
                audit_path.write_text(json.dumps({"nominal_fps": 1.0, "unique_clips_checked": 2}))
                timing = eval_official_language.verify_timing_audit(audit_path)
                self.assertEqual(timing["model_timeline_fps"], 1.0)
                audit_path.write_text(json.dumps({"nominal_fps": 2.0, "unique_clips_checked": 2}))
                with self.assertRaises(ValueError):
                    eval_official_language.verify_timing_audit(audit_path)

                old, new = tmp_path / "old", tmp_path / "new"
                old.mkdir()
                (new / "media").mkdir(parents=True)
                (new / "gt_mask").mkdir()
                frames = [old / "media" / f"frame_{i}.jpg" for i in range(10)]
                masks = [old / "gt_mask" / f"mask_{i}.png" for i in range(10)]
                wav = old / "media" / "audio.wav"
                for path in [*frames, *masks, wav]:
                    rel = new / path.relative_to(old)
                    rel.parent.mkdir(parents=True, exist_ok=True)
                    rel.touch()
                orig = tmp_path / "orig_test_s.jsonl"
                orig.write_text(json.dumps({
                    "messages": [{"role": "user", "content": "<video><audio>find piano"}],
                    "sample_id": "clip_piano_1",
                    "videos": [[str(p) for p in frames]],
                    "sam_frame_paths": [str(p) for p in frames],
                    "mask_paths": [str(p) for p in masks],
                    "audios": [str(wav)],
                }) + "\n")
                relocated = tmp_path / "relocated_test_s.jsonl"
                relocate(orig, relocated, old, new)
                orig_sha = eval_official_language.sha(orig)

                with patch.dict(eval_official_language.OFFICIAL_SPLIT_COUNTS, {"test_s": 1}), \
                        patch.dict(eval_official_language.OFFICIAL_SPLIT_SHA256, {"test_s": orig_sha}):
                    rows, prov = eval_official_language.verify_official_split_source(
                        relocated, "test_s", root=tmp_path
                    )
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(prov["original_source_sha256"], orig_sha)
                    self.assertIsNotNone(prov["media_manifest"])

                    relocated.write_text(relocated.read_text() + " ")
                    with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                        eval_official_language.verify_official_split_source(
                            relocated, "test_s", root=tmp_path
                        )
        finally:
            if added and mopd_root in sys.path:
                sys.path.remove(mopd_root)


if __name__ == "__main__":
    unittest.main()
