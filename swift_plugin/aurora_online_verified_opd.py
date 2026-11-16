"""Current-policy privileged self-distillation: no offline Teacher scores.

Teacher generation, SAM2 verification, and Student-token scoring all occur
before the optimizer update, on the current policy with adapters ENABLED.
Only verifier outcomes from this rollout batch may license a Teacher.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from copy import copy, deepcopy

import torch

from swift.rl_core.data import GRPOSample
from swift.rlhf_trainers.gkd_helpers import remap_teacher_logps_to_student_frame
from swift.trainers import disable_gradient_checkpointing
from swift.trainers.trainer_factory import TrainerFactory
from swift_plugin.aurora_sam2_grpo import AuroraSAM2MaskIoUORM
from swift_plugin.aurora_verified_routed_opd import (
    AuroraVerifiedRoutedOPDTrainer, flatten_response_ids, sample_template_modality,
)
from swift_plugin.sam2_grpo_rewards import class_exact_score, extract_target_class


@contextmanager
def stop_gradient_policy(trainer):
    """Detach Teacher branches without disabling LoRA or mutating requires_grad."""
    was_training = trainer.model.training
    trainer.model.eval()
    try:
        with torch.no_grad(), disable_gradient_checkpointing(
            trainer.model, trainer.args.gradient_checkpointing_kwargs
        ):
            yield
    finally:
        trainer.model.train(was_training)


class AuroraOnlineVerifiedOPDTrainer(AuroraVerifiedRoutedOPDTrainer):
    online_policy_teacher = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_fast_infer or self.dynamic_num_samples:
            raise ValueError("Online Teachers currently require synchronous HF inference and fixed samples")
        if self.args.num_iterations != 1 or self.args.steps_per_generation != 1:
            raise ValueError("Refresh Teacher outcomes every update: num_iterations=steps_per_generation=1")
        self.online_teacher_temperature = float(os.environ.get("AURORA_ONLINE_TEACHER_TEMPERATURE", "1.0"))
        self.online_teacher_top_p = float(os.environ.get("AURORA_ONLINE_TEACHER_TOP_P", "0.9"))
        self.write_verified_audit({
            "kind": "online_teacher_recipe",
            "teacher_source": "current Student policy; same active adapter; stop-gradient",
            "refresh": "every rollout/update; no offline Teacher completion/score reuse",
            "temperature": self.online_teacher_temperature,
            "top_p": self.online_teacher_top_p,
            "trajectories_per_enabled_teacher_per_local_student": 1,
        })

    def _new_teacher_sample(self, student, name):
        # Never copy Student response tensors/advantages into generation.
        prompt = []
        for message in student.messages:
            if message["role"] == "assistant":
                break
            prompt.append(deepcopy(message))
        active = name != "audio" or sample_template_modality(student) in {"Audio", "Audio-Visual"}
        if active:
            prompt[-1]["content"] = self._teacher_prompt(student, name)
        return GRPOSample(
            messages=prompt, videos=deepcopy(student.videos),
            audios=deepcopy(student.audios), images=deepcopy(student.images),
            extra=deepcopy(student.extra),
        )

    def _run_online_teacher(self, samples, name):
        applicable = [
            name != "audio" or sample_template_modality(s) in {"Audio", "Audio-Visual"}
            for s in samples
        ]
        active = torch.tensor(int(any(applicable)), device=self.accelerator.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX)
        if not active.item():
            return [None] * len(samples)
        # Every rank enters the native generation collectives. A rank without
        # an audio-dependent sample uses an unprivileged placeholder which is
        # discarded, never licensed or added to the optimization objective.
        teachers = [self._new_teacher_sample(s, name) for s in samples]
        original = self.request_config
        self.request_config = copy(original)
        self.request_config.temperature = self.online_teacher_temperature
        self.request_config.top_p = self.online_teacher_top_p
        try:
            with stop_gradient_policy(self):
                teachers = self._generate_completions(teachers)
        finally:
            self.request_config = original
        texts = [
            self.template.tokenizer.decode(flatten_response_ids(t.response_token_ids),
                                           skip_special_tokens=False)
            for t in teachers
        ]
        # Replay precisely the privileged prompt and generated Teacher response,
        # using the trained, frozen projector/SAM2 reward decoder.
        messages = []
        for teacher, text in zip(teachers, texts):
            msgs = deepcopy(teacher.messages)
            msgs[-1] = {"role": "assistant", "content": text}
            messages.append(msgs)
        with stop_gradient_policy(self):
            ious = AuroraSAM2MaskIoUORM()(
                completions=texts, messages=messages,
                sam_frame_paths=[s.extra["sam_frame_paths"] for s in samples],
                mask_paths=[s.extra["mask_paths"] for s in samples],
                audios=[s.audios for s in samples],
            )
        results = []
        for student, text, iou, enabled in zip(samples, texts, ious, applicable):
            valid_seg = text.count("[SEG]") == 1
            results.append({
                "completion": text,
                "predicted_class": extract_target_class(text),
                "class_correct": bool(class_exact_score(
                    text, student.extra.get("class_name"), student.extra.get("sample_id")
                )),
                "iou": float(iou) if valid_seg else 0.0,
                "seg_valid": valid_seg,
                "policy_step": int(self.state.global_step),
            } if enabled else None)
        return results

    def _score_completions(self, samples):
        # Student verifiers always score the actual current Student trajectory.
        samples = super()._score_completions(samples)
        for name, enabled in zip(("audio", "visual"), self.enabled_teachers):
            results = self._run_online_teacher(samples, name) if enabled else [None] * len(samples)
            for sample, result in zip(samples, results):
                # Explicitly overwrite possible historical cache fields, even
                # on disabled views, so stale licenses cannot leak through.
                result = result or {
                    "completion": "", "class_correct": False, "iou": 0.0,
                    "seg_valid": False, "policy_step": int(self.state.global_step),
                    "predicted_class": "",
                }
                for key, value in result.items():
                    sample.extra[f"{name}_teacher_{key}"] = value
                self.write_verified_audit({
                    "kind": "online_teacher_trajectory",
                    "step": int(self.state.global_step), "teacher": name,
                    "sample_id": sample.extra.get("sample_id"),
                    "modality": sample_template_modality(sample),
                    "enabled": bool(result["completion"]), **result,
                })
        return samples

    def _compute_teacher_logps(self, model_inputs, grpo_batch, origin_data=None,
                               teacher_model_inputs=None, teacher_grpo_batch=None):
        if teacher_model_inputs is None or teacher_grpo_batch is None:
            raise ValueError("Online self-distillation requires the explicit privileged prompt")
        with stop_gradient_policy(self):
            logps = self._get_per_token_logps_and_entropies(
                self.model, teacher_model_inputs, teacher_grpo_batch, origin_data=None
            )[0]
        return remap_teacher_logps_to_student_frame(
            logps.detach(), teacher_grpo_batch.completion_mask, grpo_batch.completion_mask
        )

    def _postprocess_batch(self, samples, batches):
        for sample in samples:
            for name in ("audio", "visual"):
                if sample.extra.get(f"{name}_teacher_policy_step") != self.state.global_step:
                    raise RuntimeError("Teacher outcome is not from the current policy step")
        return super()._postprocess_batch(samples, batches)


TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "swift_plugin.aurora_online_verified_opd.AuroraOnlineVerifiedOPDTrainer"
)
