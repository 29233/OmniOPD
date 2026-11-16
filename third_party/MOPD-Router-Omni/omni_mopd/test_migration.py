"""CPU contracts and native actor update tests; not a model-benefit benchmark."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from omni_mopd.data import audit_timing, prompt_messages, render_prompt
from omni_mopd.export_thinker import language_adapter_key
from omni_mopd.modeling import (
    assert_shared_vocabulary, extend_positions, language_parameters_only, rope_positions,
)
from omni_mopd.preflight import inspect_export, verify_native_core
from omni_mopd.rollout import OmniHFRollout


def tiny_thinker():
    from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

    config = Qwen2_5OmniThinkerConfig(
        text_config={
            "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "tie_word_embeddings": False,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "mrope_section": [1, 1, 2]},
        },
        audio_config={
            "num_mel_bins": 8, "encoder_layers": 1, "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32, "d_model": 16, "output_dim": 32,
            "max_source_positions": 100, "n_window": 16,
        },
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "out_hidden_size": 32, "patch_size": 2,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "window_size": 8, "fullatt_block_indexes": [0],
        },
        image_token_id=120, video_token_id=121, audio_token_id=122,
        vision_start_token_id=123, audio_start_token_id=124,
        audio_end_token_id=125,
    )
    config._attn_implementation = "sdpa"
    return Qwen2_5OmniThinkerForConditionalGeneration(config)


class MigrationTests(unittest.TestCase):
    def test_video_fetch_uses_swift_frame_resolution_policy(self):
        """SFT's video list has no explicit per-frame max_pixels override."""
        from types import SimpleNamespace
        from omni_mopd.data import OmniEncoder

        frames = [f"/example/frames/{index:02d}.jpg" for index in range(10)]
        row = {
            "messages": [{"role": "user", "content": "<video><audio>find it"}],
            "videos": [frames], "sam_frame_paths": frames, "audios": ["/example/audio.wav"],
        }

        class DummyProcessor:
            feature_extractor = SimpleNamespace(sampling_rate=16000)
            video_processor = SimpleNamespace(temporal_patch_size=2)

            def __call__(self, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "video_grid_thw": torch.tensor([[5, 2, 2]]),
                    "video_second_per_grid": torch.tensor([2.0]),
                }

        with patch("pathlib.Path.is_file", return_value=True), \
                patch("librosa.load", return_value=(np.zeros(64), 16000)), \
                patch("qwen_omni_utils.fetch_video", return_value=torch.zeros(10, 3, 4, 4)) as fetch, \
                patch("omni_mopd.data.rope_positions", return_value=torch.ones(3, 1, 2)):
            encoded, _, _ = OmniEncoder(
                DummyProcessor(), SimpleNamespace(), fps=1.0, max_pixels=200704
            ).encode(row)
        self.assertEqual(fetch.call_args.args[0], {"video": frames, "sample_fps": 1.0})
        self.assertEqual(encoded["video_second_per_grid"].tolist(), [2.0])
        with patch("pathlib.Path.is_file", return_value=True), \
                patch("librosa.load", return_value=(np.zeros(64), 16000)), \
                patch("qwen_omni_utils.fetch_video", return_value=torch.zeros(10, 3, 4, 4)) as fetch, \
                patch("omni_mopd.data.rope_positions", return_value=torch.ones(3, 1, 2)):
            OmniEncoder(
                DummyProcessor(), SimpleNamespace(), fps=1.0, max_pixels=200704,
                video_preprocessing="historical_native_capped",
            ).encode(row)
        self.assertEqual(fetch.call_args.args[0],
                         {"video": frames, "sample_fps": 1.0, "max_pixels": 200704})

    def test_legacy_checkpoint_media_selection_is_fail_closed(self):
        from omni_mopd.data import LEGACY_VIDEO_ENCODER_SHA256, video_preprocessing_for_run
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "omni_training_manifest.json"
            manifest.write_text(json.dumps({
                "omni_adapter_source_sha256": {"omni_mopd/data.py": LEGACY_VIDEO_ENCODER_SHA256},
            }))
            self.assertEqual(video_preprocessing_for_run(manifest), "historical_native_capped")
            manifest.write_text(json.dumps({"media": {"video_preprocessing": "swift_training_parity"}}))
            self.assertEqual(video_preprocessing_for_run(manifest), "swift_training_parity")
            manifest.write_text(json.dumps({"omni_adapter_source_sha256": {"omni_mopd/data.py": "unknown"}}))
            with self.assertRaises(ValueError):
                video_preprocessing_for_run(manifest)

    def test_native_core_untouched(self):
        hashes = verify_native_core(Path(__file__).resolve().parents[1])
        self.assertEqual(len(hashes), 5)

    def test_no_answer_or_privilege_leakage(self):
        row = {"messages": [
            {"role": "user", "content": "<video><audio>referring"},
            {"role": "assistant", "content": "GT ANSWER"},
        ], "audio_privilege_text": "TEACHER SECRET"}
        rendered = render_prompt(prompt_messages(row))
        self.assertNotIn("GT ANSWER", rendered)
        self.assertNotIn("TEACHER SECRET", rendered)
        self.assertIn("<|AUDIO|>", rendered)
        self.assertIn("<|VIDEO|>", rendered)

    def test_missing_audio_rejected(self):
        with self.assertRaises(ValueError):
            prompt_messages({"messages": [{"role": "user", "content": "<video>only"}]})

    def test_adapter_filter_excludes_sam(self):
        for suffix in ("model.embed_tokens.weight", "lm_head.weight",
                       "model.layers.0.self_attn.q_proj.lora_A.weight"):
            self.assertTrue(language_adapter_key("base_model.model.thinker." + suffix))
        for name in ("text_hidden_fcs.0.weight", "sam_mask_decoder.weight", "memory_encoder.weight"):
            self.assertFalse(language_adapter_key("base_model.model." + name))

    def test_position_extension(self):
        positions = torch.arange(5)[None, None, :].expand(2, 3, -1)
        result = extend_positions(positions, 3)
        torch.testing.assert_close(result, torch.arange(8)[None, None, :].expand(2, 3, -1))

    def test_language_freeze_real_omni(self):
        model = tiny_thinker()
        names = language_parameters_only(model, role="actor")
        self.assertGreater(len(names), 0)
        self.assertTrue(all(n.startswith(("model.layers.", "model.norm.")) for n in names))
        self.assertFalse(any(p.requires_grad for p in model.visual.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.audio_tower.parameters()))
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertFalse(model.model.embed_tokens.weight.requires_grad)
        language_parameters_only(model, role="ref")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_vocabulary_mismatch_rejected(self):
        def token(v):
            return SimpleNamespace(get_vocab=lambda: v, eos_token_id=2, pad_token_id=0)
        assert_shared_vocabulary(token({"a": 1}), token({"a": 1}))
        with self.assertRaises(ValueError):
            assert_shared_vocabulary(token({"a": 1}), token({"a": 3}))

    def test_partial_export_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "config.json").write_text('{"model_type":"qwen2_5_omni_thinker"}')
            (p / "omni_export_manifest.json").write_text('{"complete":false}')
            with self.assertRaises(ValueError):
                inspect_export(p)

    def test_wrong_fps_audit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "audit.json"
            p.write_text(json.dumps({"nominal_fps": 2, "unique_clips_checked": 3566}))
            with self.assertRaises(ValueError):
                audit_timing(p)

    def test_rollout_retains_audio_and_positions(self):
        class Generator(torch.nn.Module):
            def generate(self, **kw):
                self.captured = kw
                return torch.cat([kw["input_ids"], torch.tensor([[7, 2]])], -1)
        from verl import DataProto
        model = Generator()
        config = OmegaConf.create({
            "do_sample": True, "temperature": 1., "top_p": 1., "top_k": 0,
            "response_length": 4,
        })
        media = {name: torch.ones(1, 1) for name in (
            "input_features", "feature_attention_mask", "pixel_values_videos",
            "video_grid_thw", "video_second_per_grid",
        )}
        array = np.empty(1, dtype=object)
        array[0] = media
        prompts = DataProto(
            batch=TensorDict({
                "input_ids": torch.tensor([[5, 6]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "position_ids": torch.arange(2)[None, None, :].expand(1, 3, -1),
            }, batch_size=[1]),
            non_tensor_batch={"multi_modal_inputs": array},
            meta_info={"eos_token_id": 2, "pad_token_id": 0},
        )
        output = OmniHFRollout(model, config).generate_sequences(prompts)
        self.assertTrue(model.training)
        self.assertIn("input_features", model.captured)
        self.assertIn("pixel_values_videos", model.captured)
        self.assertEqual(tuple(output.batch["position_ids"].shape), (1, 3, 6))
        self.assertEqual(output.batch["attention_mask"][0].tolist(), [1, 1, 1, 1, 0, 0])
        self.assertIn("multi_modal_inputs", output.non_tensor_batch)
        with self.assertRaises(ValueError):
            OmniHFRollout(model, config)._one(DataProto(batch=prompts.batch))

    def test_native_expertalign_preserves_zero_signal(self):
        from verl.workers.actor.dp_actor import _delta_route_teacher_signal
        s = torch.zeros(1, 3, 4)
        teachers = torch.stack([s, s])
        weights, _, _ = _delta_route_teacher_signal(s, teachers, s)
        self.assertEqual(weights.count_nonzero().item(), 0)
        # No task adapter subsequently forces any teacher weight to 1.

    def test_native_entropy_and_nonzero_novelty(self):
        from verl.workers.actor.dp_actor import _entropy_route_teacher_signal
        from verl.utils.teacher_routing import accessible_novelty_from_topk

        ids = torch.tensor([[[0, 1, 2, 3]]])
        student = torch.tensor([[[.7, .2, .08, .02]]]).log()
        teacher = torch.tensor([[[.1, .2, .3, .4]]]).log()
        novel = accessible_novelty_from_topk(ids, student, ids, teacher)["utility"]
        same = accessible_novelty_from_topk(ids, student, ids, student)["utility"]
        self.assertGreater(novel.item(), 0)
        self.assertEqual(same.item(), 0)
        logps = torch.zeros(2, 1, 1)
        _, novelty_weights = _entropy_route_teacher_signal(
            logps, None, teacher_utilities=torch.stack([novel, same]),
            aggregation="accessible_novelty",
        )
        self.assertGreater(novelty_weights[0].item(), novelty_weights[1].item())
        _, entropy_weights = _entropy_route_teacher_signal(
            logps, torch.tensor([[[1.]], [[2.]]]), aggregation="entropy"
        )
        self.assertGreater(entropy_weights[0].item(), entropy_weights[1].item())

    def test_real_omni_audio_video_mrope_replay(self):
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        torch.manual_seed(9)
        model = tiny_thinker().eval()
        ids = torch.tensor([[5, 123, 121, 126, 124, 122, 122, 122, 122, 125, 6, 7, 8]])
        encoded = {
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values_videos": torch.randn(4, 24),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "video_second_per_grid": torch.tensor([2.]),
            "input_features": torch.randn(1, 8, 16),
            "feature_attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        positions = rope_positions(model.config, encoded)
        with torch.no_grad():
            direct = model(**encoded, position_ids=positions, use_cache=False).logits
            zero_audio = dict(encoded, input_features=torch.zeros_like(encoded["input_features"]))
            different = model(**zero_audio, position_ids=positions, use_cache=False).logits
        self.assertTrue(direct.isfinite().all())
        self.assertGreater((direct[:, -2:] - different[:, -2:]).abs().max().item(), 0)
        cfg = OmegaConf.create({
            "use_remove_padding": False, "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": 1, "entropy_from_logits_with_chunking": False,
            "use_torch_compile": False, "entropy_checkpointing": False,
            "fsdp_config": {"dtype": "float32"},
        })
        with patch("torch.distributed.get_rank", return_value=0):
            actor = DataParallelPPOActor(cfg, model)
        actor.device_name = "cpu"
        media = {k: v for k, v in encoded.items() if k not in {"input_ids", "attention_mask"}}
        fields = dict(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            position_ids=positions.transpose(0, 1), responses=ids[:, -2:],
            multi_modal_inputs=[media],
        )
        with torch.no_grad():
            _, logp = actor._forward_micro_batch(fields, temperature=1.)
        expected = direct[:, -3:-1].float().log_softmax(-1).gather(-1, ids[:, -2:, None]).squeeze(-1)
        torch.testing.assert_close(logp, expected, rtol=1e-5, atol=1e-5)

    def test_native_reverse_kl_real_omni_optimizer(self):
        from verl import DataProto
        from verl.workers.actor.dp_actor import DataParallelPPOActor

        torch.manual_seed(42)
        model = tiny_thinker()
        language_parameters_only(model, role="actor")
        config = OmegaConf.create({
            "use_remove_padding": False, "use_fused_kernels": False,
            "ulysses_sequence_parallel_size": 1, "entropy_from_logits_with_chunking": False,
            "use_torch_compile": False, "entropy_checkpointing": False,
            "fsdp_config": {"dtype": "float32"},
            "ppo_mini_batch_size": 1, "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1, "use_dynamic_bsz": False, "entropy_coeff": 0.,
            "loss_agg_mode": "seq-mean-token-mean", "grad_clip": 1.,
            "use_kl_loss": False, "clip_ratio": .2, "clip_ratio_low": .2,
            "clip_ratio_high": .2, "clip_ratio_c": 3.,
            "policy_loss": {
                "loss_mode": "vanilla", "only_reverse_kl_advantages": True,
                "lambda_vals": 1., "multi_teacher_distill": True,
                "entropy_aware_router": True, "teacher_signal_aggregation": "delta",
                "entropy_router_log_sample": False,
            },
        })
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        with patch("torch.distributed.get_rank", return_value=0):
            actor = DataParallelPPOActor(config, model, optimizer)
        actor.device_name = "cpu"
        ids = torch.tensor([[5, 6, 7, 8]])
        positions = torch.arange(4)[None, None, :].expand(1, 3, -1)
        fields = {
            "input_ids": ids, "attention_mask": torch.ones_like(ids),
            "position_ids": positions, "responses": ids[:, -2:],
        }
        model.eval()
        with torch.no_grad():
            _, old = actor._forward_micro_batch(fields, temperature=1.)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        fields.update({
            "response_mask": torch.ones(1, 2), "old_log_probs": old,
            "advantages": torch.full_like(old, 999),  # must be replaced by native OPD
            "teacher_log_probs": torch.stack([old + .4, old - .2], -1),
            "delta_router_weights": torch.tensor([[[.75, .25], [.75, .25]]]),
            "delta_router_alignment": torch.ones(1, 2, 2),
            "delta_router_specialization_norm": torch.ones(1, 2, 2),
        })
        data = DataProto.from_dict(tensors=fields, meta_info={
            "temperature": 1., "micro_batch_size": 1, "use_dynamic_bsz": False,
        })
        from verl.trainer.ppo.core_algos import get_policy_loss_fn
        native_loss = get_policy_loss_fn("vanilla")
        captured = {}
        def capture(**kwargs):
            captured["advantage"] = kwargs["advantages"].detach().clone()
            return native_loss(**kwargs)
        with patch("verl.workers.actor.dp_actor.get_device_id", return_value="cpu"), \
             patch("verl.workers.actor.dp_actor.get_policy_loss_fn", return_value=capture):
            metrics = actor.update_policy(data)
        # .75*(+.4)+.25*(-.2)=+.25, not 999, -0.25, nor nonnegative k3.
        torch.testing.assert_close(captured["advantage"], torch.full_like(old, .25), atol=1e-5, rtol=1e-5)
        changed = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
        self.assertGreater(len(changed), 0)
        self.assertTrue(all(n.startswith(("model.layers.", "model.norm.")) for n in changed))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertIn("actor/grad_norm", metrics)


if __name__ == "__main__":
    unittest.main()
