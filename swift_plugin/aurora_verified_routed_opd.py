"""Joint verifiable-reward GRPO and verified two-teacher reverse-KL OPD.

Pipeline for each sampled student trajectory:

1. score category correctness and ten-frame SAM2 mIoU;
2. read cached audio/visual Teacher probe outcomes for the same prompt;
3. license a Teacher on category rescue, or on IoU improvement when both
   Student and Teacher predict the GT category;
4. score the Student's sampled tokens under both privileged Teacher prompts;
5. localize each Teacher to its expected CoT stages and bottom-likelihood
   tokens, route overlap to one Teacher, and add ms-swift-native signed
   Teacher/Student log-ratio advantages to the ordinary GRPO task advantage.

The Teacher probe trajectory decides *who may teach*.  Teacher-forced scoring
of the Student trajectory decides *which tokens*.  These are intentionally
separate, matching the VG-OPD design.
"""

from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from swift.rlhf_trainers.utils import encode_sample
from swift.trainers.trainer_factory import TrainerFactory
from swift_plugin.aurora_sam2_grpo import AuroraSAM2GRPOTrainer
from swift_plugin.verified_opd_routing import (
    VerifiedRoutingConfig,
    add_verified_reverse_kl_advantage,
    compute_verified_routing,
    parse_verified_teacher_selection,
)


STAGES = (
    "Reference analysis",
    "Audio evidence",
    "Visual evidence",
    "Cross-modal reasoning",
    "Decision",
    "Segmentation",
)
STAGE_PATTERN = re.compile(
    r"(?m)^(" + "|".join(re.escape(value) for value in STAGES) + r"):"
)


def canonical_stage(value: str) -> str:
    return re.sub(
        r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    ).strip("_")


def inject_privilege(user_content: str, prefix: str, evidence: str) -> str:
    marker = "<video><audio>"
    if user_content.count(marker) != 1:
        raise ValueError(f"Expected exactly one {marker} marker")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f"Missing privileged evidence for {prefix!r}")
    return user_content.replace(
        marker, marker + f"{prefix}{evidence.strip()}\n", 1
    )


def flatten_response_ids(value: Any) -> list[int]:
    result: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, dict) and "token_ids" in item:
            visit(item["token_ids"])

    visit(value)
    return result


def token_stages(tokenizer, token_ids: list[int]) -> tuple[list[str], list[str]]:
    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    text = "".join(pieces)
    matches = list(STAGE_PATTERN.finditer(text))
    spans = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append((match.start(), end, canonical_stage(match.group(1))))
    stages = []
    cursor = 0
    for piece in pieces:
        start, end = cursor, cursor + len(piece)
        if "[SEG]" in piece:
            stage = "seg_token"
        else:
            stage = "other"
            for left, right, name in spans:
                if max(start, left) < min(end, right):
                    stage = name
                    break
        stages.append(stage)
        cursor = end
    return stages, pieces


def build_stage_mask(
    tokenizer,
    response_ids: list[int],
    completion_mask: torch.Tensor,
    allowed: set[str],
) -> tuple[torch.Tensor, list[str], list[str]]:
    stages, pieces = token_stages(tokenizer, response_ids)
    valid_positions = torch.nonzero(completion_mask, as_tuple=False).flatten()
    if abs(len(stages) - len(valid_positions)) > 2:
        raise ValueError(
            f"Response token/frame mismatch: ids={len(stages)}, "
            f"completion={len(valid_positions)}"
        )
    usable = min(len(stages), len(valid_positions))
    result = torch.zeros_like(completion_mask, dtype=torch.bool)
    if usable:
        selected = torch.tensor(
            [stage in allowed for stage in stages[:usable]],
            device=completion_mask.device,
            dtype=torch.bool,
        )
        result[valid_positions[:usable]] = selected
    # Template-added EOS positions belong to the final segmentation stage but
    # are not privileged evidence and therefore remain unselected.
    return result, stages, pieces


def sample_template_modality(sample) -> str:
    """Return the manifest's canonical REF-AVS modality for a rollout sample."""
    extra = getattr(sample, "extra", {}) or {}
    match = extra.get("template_match") or {}
    modality = str(match.get("template_reference") or extra.get("modality") or "")
    if modality not in {"Audio", "Audio-Visual", "Visual", "Semantic"}:
        raise ValueError(
            f"Missing/unknown template modality for sample "
            f"{extra.get('sample_id')!r}: {modality!r}"
        )
    if match.get("matcher_version") and (
        match.get("needs_review") or match.get("requires_audio") is None
        or bool(match.get("requires_audio")) != (modality in {"Audio", "Audio-Visual"})
    ):
        raise ValueError(
            f"Unresolved or modality-incompatible annotation must not silently route OPD: "
            f"{extra.get('sample_id')!r}"
        )
    return modality


def lowest_likelihood_mask(
    teacher_logps: torch.Tensor,
    allowed: torch.Tensor,
    fraction: float,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_logps.shape != allowed.shape:
        raise ValueError("Teacher log-probability/stage-mask shapes differ")
    candidates = torch.nonzero(allowed, as_tuple=False).flatten()
    seed = torch.zeros_like(allowed, dtype=torch.bool)
    if not len(candidates):
        return seed, seed.clone()
    count = max(1, int(math.ceil(len(candidates) * fraction)))
    chosen = candidates[
        torch.argsort(teacher_logps[candidates].detach().float(), stable=True)[:count]
    ]
    seed[chosen] = True
    selected = seed.clone()
    for offset in range(1, radius + 1):
        selected[offset:] |= seed[:-offset]
        selected[:-offset] |= seed[offset:]
    return seed, selected & allowed


class AuroraVerifiedRoutedOPDTrainer(AuroraSAM2GRPOTrainer):
    """SAM2 GRPO with cached Teacher probes and native OPD-RL advantages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.online_policy_teacher = getattr(self, "online_policy_teacher", False)
        if self.online_policy_teacher:
            if self.teacher_model is not None or self.use_teacher_api or self._teacher_use_disable_adapter:
                raise ValueError("Online Teachers must reuse the current policy with its adapters enabled")
        elif self.teacher_model is None or self.use_teacher_api or self._teacher_use_disable_adapter:
            raise ValueError(
                "Verified routed OPD requires an independent local frozen SFT Teacher"
            )
        if float(self.teacher_kl_coef) != 0:
            raise ValueError(
                "Set --teacher_kl_coef 0; this plugin injects verified k3 advantages itself"
            )
        if self.dynamic_sample:
            raise ValueError(
                "Native DAPO dynamic_sample drops all-zero KD-rescue groups; "
                "use the refreshed empirical difficulty mixture instead"
            )
        reward_names = list(self.reward_func_names or [])
        accepted_reward_names = (
            ["aurora_class_exact", "aurora_sam2_mask_iou"],
            ["AuroraClassExactORM", "AuroraSAM2MaskIoUORM"],
        )
        if reward_names not in accepted_reward_names:
            raise ValueError(
                "Verified OPD expects raw class and mask-IoU reward functions in that "
                f"order, got {reward_names!r}"
            )
        weights = [float(value) for value in self.reward_weights.detach().cpu().tolist()]
        if weights != [0.5, 0.5]:
            raise ValueError(
                "Use reward_weights 0.5 0.5; mask reward is category-gated in this trainer"
            )
        self.verified_config = VerifiedRoutingConfig(
            iou_margin=float(os.environ.get("AURORA_VERIFIED_IOU_MARGIN", "0.01")),
            category_rescue_bonus=float(
                os.environ.get("AURORA_VERIFIED_CATEGORY_BONUS", "2.0")
            ),
            iou_gain_weight=float(
                os.environ.get("AURORA_VERIFIED_IOU_GAIN_WEIGHT", "0.75")
            ),
            absolute_iou_weight=float(
                os.environ.get("AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT", "0.25")
            ),
            route_temperature=float(
                os.environ.get("AURORA_VERIFIED_ROUTE_TEMPERATURE", "0.25")
            ),
            k3_monitor_clip=float(
                os.environ.get("AURORA_VERIFIED_K3_CLIP", "10.0")
            ),
            kd_coef=float(os.environ.get("AURORA_VERIFIED_KD_COEF", "0.25")),
        )
        self.fraction = float(
            os.environ.get("AURORA_VERIFIED_TOKEN_FRACTION", "0.10")
        )
        self.radius = int(os.environ.get("AURORA_VERIFIED_TOKEN_RADIUS", "1"))
        self.task_reward_coef = float(
            os.environ.get("AURORA_VERIFIED_TASK_REWARD_COEF", "1.0")
        )
        if not 0 <= self.task_reward_coef <= 1:
            raise ValueError("AURORA_VERIFIED_TASK_REWARD_COEF must be in [0,1]")
        self.enabled_teachers = parse_verified_teacher_selection(
            os.environ.get("AURORA_VERIFIED_TEACHERS", "both")
        )
        if not 0 < self.fraction <= 1 or self.radius < 0:
            raise ValueError("Invalid verified token localization settings")
        if not self.online_policy_teacher:
            teacher = self.accelerator.unwrap_model(self.teacher_model)
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            teacher.eval()
        trainable = []
        for name, parameter in self.model.named_parameters():
            enabled = "lora_A" in name or "lora_B" in name
            parameter.requires_grad_(enabled)
            if enabled:
                parameter.data = parameter.data.float()
                trainable.append(name)
        if not trainable:
            raise RuntimeError("Verified routed OPD left no q/v LoRA parameters trainable")
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.write_verified_audit({
            "kind": "recipe",
            "objective": (
                "category-gated verifiable GRPO + licensed two-teacher sampled-token "
                "reverse-KL native signed log-ratio advantage"
                if self.task_reward_coef else
                "licensed two-teacher native signed reverse-KL OPD only; "
                "task reward advantage disabled; reference-policy KL regularization retained"
            ),
            "routing": self.verified_config.__dict__,
            "fraction": self.fraction,
            "task_reward_coef": self.task_reward_coef,
            "teacher_policy": "current_student_stop_gradient" if self.online_policy_teacher else "frozen_sft",
            "radius": self.radius,
            "teacher_order": ["audio", "visual"],
            "enabled_teachers": {
                "audio": self.enabled_teachers[0],
                "visual": self.enabled_teachers[1],
            },
            "teacher_stage_masks": {
                "audio": ["audio_evidence", "cross_modal_reasoning"],
                "visual": ["visual_evidence", "cross_modal_reasoning"],
            },
            "difficulty_sampling": "offline empirical mixture; native dynamic_sample disabled",
            "trainable": "q/v LoRA only; embeddings, LM head, projector and SAM2 frozen",
        })

    def write_verified_audit(self, row: dict[str, Any]) -> None:
        folder = Path(self.args.output_dir) / "verified_routed_opd_audit"
        folder.mkdir(parents=True, exist_ok=True)
        row = {**row, "rank": self.accelerator.process_index}
        with (folder / f"rank{self.accelerator.process_index}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _score_completions(self, samples):
        samples = super()._score_completions(samples)
        if self._rewards_per_func.shape[1] != 2:
            raise ValueError("Expected class and mask-IoU reward columns")
        # Attach raw verifier outcomes to the local samples before category
        # gating changes the mask reward used by GRPO.
        from swift.rlhf_trainers.utils import get_even_process_data

        local = get_even_process_data(self, self._rewards_per_func)
        if len(local) != len(samples):
            raise ValueError("Local verifier rewards do not align with sampled trajectories")
        for sample, values in zip(samples, local):
            sample.extra["_verified_student_class"] = float(values[0])
            sample.extra["_verified_student_iou"] = float(values[1])
        self._rewards_per_func = self._transform_verified_rewards(samples, self._rewards_per_func)
        return samples

    def _transform_verified_rewards(self, samples, raw_rewards):
        """Default historical category gate; subclasses may replace this explicitly."""
        result = raw_rewards.clone()
        result[:, 1] = result[:, 0] * result[:, 1]
        return result

    def _select_verified_tokens(self, tokenizer, response_ids, completion_mask,
                                teacher_logps, allowed):
        allowed_mask, stages, pieces = build_stage_mask(
            tokenizer, response_ids, completion_mask, allowed
        )
        seed, selected = lowest_likelihood_mask(
            teacher_logps, allowed_mask, self.fraction, self.radius
        )
        return seed, selected, stages, pieces

    def _compute_verified_routing(self, chunk, student_class, student_iou,
                                  teacher_class, teacher_iou, teacher_valid):
        return compute_verified_routing(
            student_class, student_iou, teacher_class, teacher_iou,
            teacher_valid, self.verified_config,
        )

    def _teacher_prompt(self, sample, teacher_name: str) -> str:
        user = next(
            (
                message
                for message in reversed(sample.messages)
                if message.get("role") == "user"
            ),
            None,
        )
        if user is None or not isinstance(user.get("content"), str):
            raise ValueError("Sample has no string user prompt")
        if teacher_name == "audio":
            return inject_privilege(
                user["content"],
                "Privileged clip-level audio evidence: ",
                sample.extra.get("audio_privilege_text"),
            )
        return inject_privilege(
            user["content"],
            "Hint: Visual evidence about the target: ",
            sample.extra.get("visual_privilege_text"),
        )

    def _teacher_view_logps(self, chunk, batch_encoded, teacher_name: str):
        previous = [
            (sample.teacher_prompt, sample.teacher_messages) for sample in chunk
        ]
        try:
            for sample in chunk:
                sample.teacher_prompt = self._teacher_prompt(sample, teacher_name)
                sample.teacher_messages = None
                sample.build_teacher_view()
            teacher_inputs, teacher_batch = self._collate_teacher_opsd_batch(
                chunk, self.template
            )
            return self._compute_teacher_logps(
                batch_encoded["model_inputs"],
                batch_encoded["grpo_batch"],
                teacher_model_inputs=teacher_inputs,
                teacher_grpo_batch=teacher_batch,
            )
        finally:
            for sample, (prompt, messages) in zip(chunk, previous):
                sample.teacher_prompt = prompt
                sample.teacher_messages = messages
                sample.encoded = encode_sample(sample, self.template)
                sample.encoded.pop("_extra_kwargs", None)

    def _prepare_batch_inputs(self, samples):
        # Suppress the base trainer's unprivileged single-Teacher forward.  The
        # independent Teacher remains loaded and is scored below under two
        # privileged prompts.
        original = self._has_teacher_explicit
        original_dynamic = self._is_dynamic_self_distillation
        self._has_teacher_explicit = lambda: False
        self._is_dynamic_self_distillation = False
        try:
            batches = super()._prepare_batch_inputs(samples)
        finally:
            self._has_teacher_explicit = original
            self._is_dynamic_self_distillation = original_dynamic
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batches):
            raise ValueError("Verified Teacher chunks do not align with GRPO batches")
        for batch_encoded, chunk in zip(batches, chunks):
            grpo_batch = batch_encoded["grpo_batch"]
            grpo_batch.teacher_per_token_logps = None
            computed_logps = {
                name: self._teacher_view_logps(chunk, batch_encoded, name)
                for name, enabled in zip(
                    ("audio", "visual"), self.enabled_teachers
                )
                if enabled
            }
            reference_logps = next(iter(computed_logps.values()))
            teacher_logps = torch.stack(
                [
                    computed_logps.get(name, torch.zeros_like(reference_logps))
                    for name in ("audio", "visual")
                ],
                dim=1,
            )
            if teacher_logps.shape[2] != grpo_batch.completion_mask.shape[1]:
                raise ValueError("Privileged Teacher logps do not align with completion frame")
            selected_rows = []
            seed_rows = []
            stage_records = []
            piece_records = []
            for row_index, sample in enumerate(chunk):
                response_ids = flatten_response_ids(sample.response_token_ids)
                modality = sample_template_modality(sample)
                audio_related = modality in {"Audio", "Audio-Visual"}
                cross_modal = modality == "Audio-Visual"
                per_teacher_selected = []
                per_teacher_seed = []
                sample_stages = sample_pieces = None
                for teacher_index, allowed in enumerate((
                    (
                        {"audio_evidence"}
                        | ({"cross_modal_reasoning"} if cross_modal else set())
                    ) if self.enabled_teachers[0] and audio_related else set(),
                    (
                        {"visual_evidence"}
                        | ({"cross_modal_reasoning"} if cross_modal else set())
                    ) if self.enabled_teachers[1] else set(),
                )):
                    seed, selected, stages, pieces = self._select_verified_tokens(
                        self.template.tokenizer,
                        response_ids,
                        grpo_batch.completion_mask[row_index],
                        teacher_logps[row_index, teacher_index],
                        allowed,
                    )
                    per_teacher_seed.append(seed)
                    per_teacher_selected.append(selected)
                    sample_stages, sample_pieces = stages, pieces
                seed_rows.append(torch.stack(per_teacher_seed))
                selected_rows.append(torch.stack(per_teacher_selected))
                stage_records.append(sample_stages)
                piece_records.append(sample_pieces)
            batch_encoded["_verified_teacher_logps"] = teacher_logps
            batch_encoded["_verified_seed_masks"] = torch.stack(seed_rows)
            batch_encoded["_verified_selected_masks"] = torch.stack(selected_rows)
            batch_encoded["_verified_samples"] = chunk
            batch_encoded["_verified_stages"] = stage_records
            batch_encoded["_verified_pieces"] = piece_records
        return batches

    def _postprocess_batch(self, samples, batch_encoded_inputs):
        # Base GRPO computes task advantages from the category-gated reward.
        for batch in batch_encoded_inputs:
            batch["grpo_batch"].teacher_per_token_logps = None
        super()._postprocess_batch(samples, batch_encoded_inputs)
        mode = "train" if self.model.training else "eval"
        for batch in batch_encoded_inputs:
            chunk = batch["_verified_samples"]
            grpo_batch = batch["grpo_batch"]
            student_class = torch.tensor(
                [
                    float(sample.extra["_verified_student_class"]) > 0.5
                    for sample in chunk
                ],
                device=grpo_batch.completion_mask.device,
            )
            student_iou = torch.tensor(
                [float(sample.extra["_verified_student_iou"]) for sample in chunk],
                device=grpo_batch.completion_mask.device,
            )
            teacher_class = torch.tensor(
                [
                    [
                        bool(sample.extra.get("audio_teacher_class_correct")),
                        bool(sample.extra.get("visual_teacher_class_correct")),
                    ]
                    for sample in chunk
                ],
                device=grpo_batch.completion_mask.device,
            )
            teacher_iou = torch.tensor(
                [
                    [
                        float(sample.extra.get("audio_teacher_iou") or 0),
                        float(sample.extra.get("visual_teacher_iou") or 0),
                    ]
                    for sample in chunk
                ],
                device=grpo_batch.completion_mask.device,
            )
            teacher_valid = torch.tensor(
                [
                    [
                        self.enabled_teachers[0]
                        and sample_template_modality(sample)
                        in {"Audio", "Audio-Visual"}
                        and bool(sample.extra.get("audio_teacher_completion")),
                        self.enabled_teachers[1]
                        and bool(sample.extra.get("visual_teacher_completion")),
                    ]
                    for sample in chunk
                ],
                device=grpo_batch.completion_mask.device,
            )
            routing = self._compute_verified_routing(
                chunk,
                student_class,
                student_iou,
                teacher_class,
                teacher_iou,
                teacher_valid,
            )
            combined, details = add_verified_reverse_kl_advantage(
                grpo_batch.advantages,
                grpo_batch.old_per_token_logps,
                batch["_verified_teacher_logps"],
                batch["_verified_selected_masks"],
                routing,
                grpo_batch.completion_mask,
                self.verified_config,
                task_reward_coef=self.task_reward_coef,
            )
            grpo_batch.advantages = combined
            for name, value in details["metrics"].items():
                self._metrics[mode][f"verified_opd/{name}"].append(float(value))

            for row_index, sample in enumerate(chunk):
                valid_positions = torch.nonzero(
                    grpo_batch.completion_mask[row_index], as_tuple=False
                ).flatten()
                token_count = min(
                    len(batch["_verified_pieces"][row_index]), len(valid_positions)
                )
                position_values = valid_positions[:token_count]
                active_local = details["teacher_index"][
                    row_index, position_values
                ].ge(0)
                active_positions = torch.nonzero(
                    active_local, as_tuple=False
                ).flatten()
                active_frame_positions = position_values[active_positions]
                self.write_verified_audit({
                    "kind": "trajectory",
                    "step": self.state.global_step,
                    "training": self.model.training,
                    "sample_id": sample.extra.get("sample_id"),
                    "difficulty_bucket": sample.extra.get("difficulty_bucket"),
                    "student": {
                        "class_correct": bool(student_class[row_index]),
                        "iou": float(student_iou[row_index]),
                    },
                    "teachers": {
                        name: {
                            "class_correct": bool(teacher_class[row_index, index]),
                            "iou": float(teacher_iou[row_index, index]),
                            "class_license": bool(routing["class_license"][row_index, index]),
                            "iou_license": bool(routing["iou_license"][row_index, index]),
                            "route_prob": float(routing["route_prob"][row_index, index]),
                            "verified_weight": float(
                                routing["verified_weight"][row_index, index]
                            ),
                        }
                        for index, name in enumerate(("audio", "visual"))
                    },
                    "token_pieces": batch["_verified_pieces"][row_index][:token_count],
                    "stages": batch["_verified_stages"][row_index][:token_count],
                    "seed_masks": batch["_verified_seed_masks"][
                        row_index, :, position_values
                    ].detach().cpu().tolist(),
                    "selected_masks": batch["_verified_selected_masks"][
                        row_index, :, position_values
                    ].detach().cpu().tolist(),
                    "teacher_assignment": details["teacher_index"][
                        row_index, position_values
                    ].detach().cpu().tolist(),
                    "token_weight": details["token_weight"][
                        row_index, position_values
                    ].detach().cpu().tolist(),
                    "signed_logratio": details["signed_logratio"][
                        row_index, position_values
                    ].detach().cpu().tolist(),
                    "k3": details["k3"][
                        row_index, position_values
                    ].detach().cpu().tolist(),
                    "kd_advantage": details["kd_advantage"][
                        row_index, position_values
                    ].detach().cpu().tolist(),
                    # Raw sampled-token log-probabilities were not present in
                    # the original audit schema.  Retain them for active OPD
                    # tokens so future runs can plot signed Teacher/Student
                    # log-probability distributions without replaying models.
                    "active_positions": active_positions.detach().cpu().tolist(),
                    "active_student_logps": grpo_batch.old_per_token_logps[
                        row_index, active_frame_positions
                    ].detach().cpu().float().tolist(),
                    "active_teacher_logps": details["chosen_teacher_logps"][
                        row_index, active_frame_positions
                    ].detach().cpu().float().tolist(),
                    "active_logp_delta": (
                        details["chosen_teacher_logps"][
                            row_index, active_frame_positions
                        ]
                        - grpo_batch.old_per_token_logps[
                            row_index, active_frame_positions
                        ]
                    ).detach().cpu().float().tolist(),
                })


TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "swift_plugin.aurora_verified_routed_opd.AuroraVerifiedRoutedOPDTrainer"
)
