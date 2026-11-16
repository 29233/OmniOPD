"""Task-aware audio-privileged OPD with a frozen decision/SEG anchor.

This is a deliberately smaller follow-up to the dual privileged-top10 run:

* only the empirically useful audio privileged view teaches;
* its bottom-quantile supervision is restricted to Audio evidence and
  Cross-modal reasoning tokens;
* an unprivileged frozen SFT view anchors Decision and Segmentation tokens;
* only q/v LoRA parameters are updated.

It remains pure on-policy distillation (no reward and no mask loss). The anchor
is intended to preserve target binding and the [SEG] prompt interface while the
audio view improves the evidence-bearing reasoning stages.
"""

from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from swift.rl_core.data import GKDBatch
from swift.rlhf_trainers.utils import encode_sample
from swift.trainers import disable_gradient_checkpointing
from swift.trainers.trainer_factory import TrainerFactory
from swift.utils import to_device

from swift_plugin.aurora_opsd import AuroraGKDTrainer
from swift_plugin.dual_opsd_loss import active_response, assert_response_alignment


STAGES = (
    "Reference analysis",
    "Audio evidence",
    "Visual evidence",
    "Cross-modal reasoning",
    "Decision",
    "Segmentation",
)
STAGE_PATTERN = re.compile(r"(?m)^(" + "|".join(re.escape(value) for value in STAGES) + r"):")
TARGET_PATTERN = re.compile(r"The target is[^.\n]*", re.IGNORECASE)


def canonical_stage_name(value: str) -> str:
    """Normalize display labels and environment values to one stage key."""

    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


@contextmanager
def temporary_eval(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.training = was_training


def inject_audio(user_content: str, evidence: str) -> str:
    marker = "<video><audio>"
    if user_content.count(marker) != 1:
        raise ValueError(f"Expected exactly one {marker} marker")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Missing audio_privilege_text")
    return user_content.replace(
        marker,
        marker + f"Privileged clip-level audio evidence: {evidence.strip()}\n",
        1,
    )


def make_views(sample):
    if sample.teacher_prompt or sample.extra.get("teacher_audio_paths"):
        raise ValueError("Audio-stage anchor OPD expects text privilege columns only")
    student, baseline, audio = [deepcopy(sample) for _ in range(3)]
    for view in (student, baseline, audio):
        view.extra.pop("sam_frame_paths", None)
        view.extra.pop("mask_paths", None)
    user = next((message for message in reversed(audio.messages) if message["role"] == "user"), None)
    if user is None or not isinstance(user.get("content"), str):
        raise ValueError("Expected a string user prompt")
    user["content"] = inject_audio(user["content"], sample.extra.get("audio_privilege_text"))
    if student.messages != baseline.messages or student.audios != baseline.audios:
        raise AssertionError("Unprivileged anchor view differs from the student input")
    if student.audios != audio.audios or student.videos != audio.videos:
        raise AssertionError("Audio text privilege unexpectedly changed media")
    return student, baseline, audio


def token_stage_mask(tokenizer, token_ids: torch.Tensor, allowed: set[str]) -> tuple[torch.Tensor, list[str]]:
    ids = token_ids.detach().cpu().tolist()
    pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in ids
    ]
    text = "".join(pieces)
    matches = list(STAGE_PATTERN.finditer(text))
    spans = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append((match.start(), end, canonical_stage_name(match.group(1))))
    target = TARGET_PATTERN.search(text)
    stages: list[str] = []
    cursor = 0
    for piece in pieces:
        start, end = cursor, cursor + len(piece)
        if "[SEG]" in piece or "[SEG]" in text[start:end]:
            stage = "seg_token"
        elif target and max(start, target.start()) < min(end, target.end()):
            stage = "decision_target"
        else:
            stage = "other"
            for left, right, name in spans:
                if max(start, left) < min(end, right):
                    stage = name
                    break
        stages.append(stage)
        cursor = end
    mask = torch.tensor([stage in allowed for stage in stages], device=token_ids.device, dtype=torch.bool)
    return mask, stages


def select_low_likelihood(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    allowed: torch.Tensor,
    fraction: float,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if allowed.shape != token_ids.shape:
        raise ValueError("Audio-stage filter shape mismatch")
    with torch.no_grad():
        log_probs = F.log_softmax(logits.detach().float(), dim=-1)
        likelihood = log_probs.gather(-1, token_ids[:, None]).squeeze(-1)
        if not bool(allowed.any()):
            empty = torch.zeros_like(allowed)
            return empty, empty.clone(), likelihood
        candidates = torch.nonzero(allowed, as_tuple=False).flatten()
        count = max(1, int(math.ceil(len(candidates) * fraction)))
        order = candidates[torch.argsort(likelihood[candidates], stable=True)[:count]]
        seed = torch.zeros_like(allowed)
        seed[order] = True
        selected = seed.clone()
        for offset in range(1, radius + 1):
            selected[offset:] |= seed[:-offset]
            selected[:-offset] |= seed[offset:]
        selected &= allowed
    return seed, selected, likelihood


def masked_forward_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    if student.shape != teacher.shape or mask.shape != (student.shape[0],):
        raise ValueError("Masked KL alignment mismatch")
    total = student.sum() * 0.0
    for start in range(0, student.shape[0], chunk_size):
        end = min(start + chunk_size, student.shape[0])
        log_s = F.log_softmax(student[start:end].float() / temperature, dim=-1)
        log_t = F.log_softmax(teacher[start:end].detach().float() / temperature, dim=-1)
        kl = (log_t.exp() * (log_t - log_s)).sum(-1).clamp_min(0)
        total = total + (kl * mask[start:end]).sum()
    return total * temperature**2 / mask.sum().clamp_min(1)


class LoRAUpdateAudit(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.before = {}

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model") or self.trainer.model
        self.before = {}
        finite = nonfinite = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            ok = bool(torch.isfinite(parameter.grad).all())
            finite += int(ok)
            nonfinite += int(not ok)
            group = "lora_B" if "lora_B" in name else "lora_A" if "lora_A" in name else None
            if group is None or any(value[-1] == group for value in self.before.values()):
                continue
            count = min(2048, parameter.numel())
            ids = (
                torch.arange(count, device=parameter.device, dtype=torch.long)
                * (parameter.numel() - 1)
                // max(count - 1, 1)
            )
            self.before[name] = (parameter, ids, parameter.detach().flatten()[ids].clone(), group)
        self.trainer.write_audit({
            "kind": "pre_optimizer",
            "step": state.global_step + 1,
            "finite_gradient_tensors": finite,
            "nonfinite_gradient_tensors": nonfinite,
        })

    def on_optimizer_step(self, args, state, control, **kwargs):
        updates = {}
        for name, (parameter, ids, previous, group) in self.before.items():
            delta = (parameter.detach().flatten()[ids] - previous).float()
            updates[group] = {
                "name": name,
                "changed_elements": int(delta.ne(0).sum()),
                "mean_abs_update": float(delta.abs().mean()),
                "all_finite": bool(torch.isfinite(delta).all()),
            }
        self.trainer.write_audit({
            "kind": "optimizer_update",
            "step": state.global_step + 1,
            "optimizer_step_was_skipped": bool(self.trainer.accelerator.optimizer_step_was_skipped),
            "updates": updates,
        })
        self.before = {}


class AuroraAudioStageAnchorOPSDTrainer(AuroraGKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._is_self_distillation or self.use_teacher_api or self._teacher_use_disable_adapter or self.teacher_model is None:
            raise ValueError("Audio-stage anchor OPD requires an independent local frozen SFT teacher")
        if self.lmbda != 1 or self.beta != 0 or self.gkd_logits_topk is not None or self.args.sft_alpha != 0:
            raise ValueError("Audio-stage anchor OPD requires lmbda=1, beta=0, full vocabulary and sft_alpha=0")
        if float(os.environ.get("AURORA_OPSD_MASK_WEIGHT", "0")) != 0:
            raise ValueError("Audio-stage anchor OPD is language-only; mask weight must be zero")
        self.fraction = float(os.environ.get("AURORA_AUDIO_STAGE_FRACTION", "0.10"))
        self.radius = int(os.environ.get("AURORA_AUDIO_STAGE_RADIUS", "1"))
        self.anchor_weight = float(os.environ.get("AURORA_AUDIO_STAGE_ANCHOR_WEIGHT", "0.5"))
        self.anchor_mode = os.environ.get("AURORA_AUDIO_STAGE_ANCHOR_MODE", "stages").strip().lower()
        self.temperature = float(os.environ.get("AURORA_AUDIO_STAGE_TEMPERATURE", "1.0"))
        self.chunk_size = int(os.environ.get("AURORA_AUDIO_STAGE_CHUNK_SIZE", "16"))
        self.audio_stages = {
            canonical_stage_name(value)
            for value in os.environ.get(
                "AURORA_AUDIO_STAGE_NAMES", "audio_evidence,cross_modal_reasoning"
            ).split(",")
            if value.strip()
        }
        self.anchor_stages = {
            canonical_stage_name(value)
            for value in os.environ.get(
                "AURORA_ANCHOR_STAGE_NAMES", "decision,decision_target,segmentation,seg_token"
            ).split(",")
            if value.strip()
        }
        known_stages = {
            *(canonical_stage_name(value) for value in STAGES),
            "decision_target",
            "seg_token",
        }
        unknown = (self.audio_stages | self.anchor_stages) - known_stages
        if unknown:
            raise ValueError(f"Unknown normalized CoT stages: {sorted(unknown)}")
        overlap = self.audio_stages & self.anchor_stages
        if overlap:
            raise ValueError(f"Audio and anchor stage sets must be disjoint: {sorted(overlap)}")
        if not (0 < self.fraction <= 1) or self.radius < 0 or self.anchor_weight < 0:
            raise ValueError("Invalid audio-stage anchor configuration")
        if self.anchor_mode not in {"stages", "complement"}:
            raise ValueError("AURORA_AUDIO_STAGE_ANCHOR_MODE must be stages or complement")
        teacher = self.accelerator.unwrap_model(self.teacher_model)
        if not getattr(teacher, "peft_config", None):
            raise ValueError("Frozen teacher must load the SFT adapter")
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        teacher.eval()
        trainable_names = []
        for name, parameter in self.model.named_parameters():
            enabled = "lora_A" in name or "lora_B" in name
            parameter.requires_grad_(enabled)
            if enabled:
                parameter.data = parameter.data.float()
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError("No q/v LoRA parameters remain trainable")
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.add_callback(LoRAUpdateAudit(self))
        self.write_audit({
            "kind": "recipe",
            "objective": "audio-only stage-localized forward KL plus unprivileged decision/SEG anchor",
            "audio_stages": sorted(self.audio_stages),
            "anchor_stages": sorted(self.anchor_stages),
            "fraction": self.fraction,
            "radius": self.radius,
            "anchor_weight": self.anchor_weight,
            "anchor_mode": self.anchor_mode,
            "temperature": self.temperature,
            "trainable": "q/v LoRA only",
            "teacher": "independent frozen SAM2 SFT checkpoint",
        })

    def write_audit(self, record):
        folder = Path(self.args.output_dir) / "audio_stage_anchor_audit"
        folder.mkdir(parents=True, exist_ok=True)
        record = {**record, "rank": self.accelerator.process_index}
        with (folder / f"rank{self.accelerator.process_index}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _engine_infer(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            return super()._engine_infer(*args, **kwargs)

    def _prepare_batch_inputs(self, samples):
        batches = []
        for chunk in self.split_by_mini_batches(samples):
            if len(chunk) != 1:
                raise ValueError("Audio-stage anchor OPD requires per-device microbatch=1")
            original = chunk[0]
            views = make_views(original)
            encoded = []
            with self._template_context(self.template):
                for view in views:
                    row = encode_sample(view, self.template)
                    row.pop("_extra_kwargs", None)
                    encoded.append(to_device(self.template.data_collator([row]), self.accelerator.device))
            student, baseline, audio = encoded
            assert_response_alignment(student["labels"], baseline["labels"])
            assert_response_alignment(student["labels"], audio["labels"])
            batches.append({
                "model_inputs": student,
                "teacher_inputs": [baseline, audio],
                "evidence": {
                    "sample_id": original.extra.get("sample_id"),
                    "audio_evidence": original.extra.get("audio_privilege_text"),
                },
                "gkd_batch": GKDBatch(data_source=self._data_source),
            })
        return batches

    def _compute_aurora_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_inputs = dict(inputs["model_inputs"])
        teacher_inputs = [dict(value) for value in inputs["teacher_inputs"]]
        for teacher in teacher_inputs:
            assert_response_alignment(student_inputs["labels"], teacher["labels"])
        if self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1):
            for value in [student_inputs, *teacher_inputs]:
                self.prepare_logits_to_keep(value)
        raw_teacher = self.accelerator.unwrap_model(self.teacher_model)
        teacher_scores = []
        teacher_ids = []
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
            temporary_eval(raw_teacher),
            disable_gradient_checkpointing(self.teacher_model, self.args.gradient_checkpointing_kwargs),
            self.load_teacher_model_context(),
        ):
            for value in teacher_inputs:
                output = self.teacher_model(
                    **{key: item for key, item in value.items() if key != "labels"},
                    aurora_skip_hidden_states=True,
                )
                scores, token_ids = active_response(output.logits, value["labels"])
                teacher_scores.append(scores.detach())
                teacher_ids.append(token_ids)
                del output
        outputs = model(
            **{key: value for key, value in student_inputs.items() if key != "labels"},
            aurora_skip_hidden_states=True,
            output_hidden_states=False,
        )
        student_scores, token_ids = active_response(outputs.logits, student_inputs["labels"])
        if any(not torch.equal(token_ids, value) for value in teacher_ids):
            raise ValueError("Teacher/student sampled response tokens do not align")
        audio_allowed, stages = token_stage_mask(raw_teacher.aurora_tokenizer, token_ids, self.audio_stages)
        audio_seed, audio_selected, audio_likelihood = select_low_likelihood(
            teacher_scores[1], token_ids, audio_allowed, self.fraction, self.radius
        )
        if self.anchor_mode == "complement":
            anchor_allowed = ~audio_selected
        else:
            anchor_allowed, _ = token_stage_mask(
                raw_teacher.aurora_tokenizer, token_ids, self.anchor_stages
            )
        audio_loss = masked_forward_kl(
            student_scores, teacher_scores[1], audio_selected, self.temperature, self.chunk_size
        )
        anchor_loss = masked_forward_kl(
            student_scores, teacher_scores[0], anchor_allowed, self.temperature, self.chunk_size
        )
        loss = audio_loss + self.anchor_weight * anchor_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite audio-stage anchor OPD loss")
        metrics = {
            "audio_stage/kd_loss": loss.detach(),
            "audio_stage/audio_kl": audio_loss.detach(),
            "audio_stage/anchor_kl": anchor_loss.detach(),
            "audio_stage/audio_allowed_rate": audio_allowed.float().mean(),
            "audio_stage/audio_seed_rate": audio_seed.float().mean(),
            "audio_stage/audio_selected_rate": audio_selected.float().mean(),
            "audio_stage/anchor_token_rate": anchor_allowed.float().mean(),
        }
        self._aurora_mask_parts = {
            key: self.accelerator.reduce(value, reduction="mean") for key, value in metrics.items()
        }
        self.write_audit({
            "kind": "loss",
            "step": self.state.global_step,
            "training": self.model.training,
            **inputs["evidence"],
            "metrics": {key: float(value) for key, value in metrics.items()},
            "token_ids": token_ids.detach().cpu().tolist(),
            "token_pieces": [
                raw_teacher.aurora_tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for token_id in token_ids.detach().cpu().tolist()
            ],
            "stages": stages,
            "audio_likelihood": audio_likelihood.detach().cpu().tolist(),
            "audio_seed": audio_seed.detach().cpu().tolist(),
            "audio_selected": audio_selected.detach().cpu().tolist(),
            "anchor_selected": anchor_allowed.detach().cpu().tolist(),
        })
        return (loss, outputs) if return_outputs else loss


TrainerFactory.TRAINER_MAPPING["gkd"] = (
    "swift_plugin.aurora_audio_stage_anchor_opd.AuroraAudioStageAnchorOPSDTrainer"
)
