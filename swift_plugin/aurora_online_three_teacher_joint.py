"""Three-teacher REF-AVS Joint training on the historical ms-swift code base.

Teachers 0/1 are the existing current-policy privileged Audio/Visual views.
Teacher 2 is an independent frozen GRPO checkpoint.  The frozen model scores
the Student's sampled response on the ordinary Student prompt and also
generates a trajectory for the same live class/mask verifier, so its routing
license is not fabricated from the Student outcome.

This file intentionally preserves the previous verified OPD protocol (the
category/IoU license, stage localization, low-likelihood selection and native
signed log-ratio advantage).  It is a separate experiment from the newer
paper-native MOPD-Router DDP run.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch

from swift.rlhf_trainers.utils import encode_sample
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from swift.trainers import disable_gradient_checkpointing
from swift.trainers.trainer_factory import TrainerFactory
from swift_plugin.aurora_online_verified_opd import (
    AuroraOnlineVerifiedOPDTrainer,
    stop_gradient_policy,
)
from swift_plugin.aurora_verified_routed_opd import (
    STAGES,
    build_stage_mask,
    flatten_response_ids,
    sample_template_modality,
)
from swift_plugin.aurora_sam2_grpo import AuroraSAM2MaskIoUORM
from swift_plugin.sam2_grpo_rewards import class_exact_score, extract_target_class


TEACHER_NAMES = ("audio", "visual", "grpo_anchor")


class AuroraOnlineThreeTeacherJointTrainer(AuroraOnlineVerifiedOPDTrainer):
    """Historical verified Joint trainer extended with a frozen third teacher."""

    # The parent online trainer normally rejects ``teacher_model`` because it
    # reuses the current Student for both privileged views.  Here the explicit
    # teacher_model is the frozen GRPO checkpoint used as teacher 2.
    online_policy_teacher = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.teacher_model is None:
            raise ValueError("Three-teacher Joint requires --teacher_model and --teacher_adapters")
        if self.use_fast_infer:
            raise ValueError("Three-teacher Joint requires synchronous HF generation")
        self.enabled_teachers = (True, True, True)
        self.write_verified_audit({
            "kind": "three_teacher_recipe",
            "teacher_order": list(TEACHER_NAMES),
            "teacher_0_1": "current Student policy with Audio/Visual privilege and stop-gradient",
            "teacher_2": "frozen GRPO checkpoint on the ordinary Student prompt",
            "teacher_2_generation": "online frozen-model trajectory, verifier-scored before update",
            "teacher_2_token_scoring": "frozen GRPO model scores Student sampled tokens",
            "routing": "existing verified category/IoU routing generalized to K=3",
        })

    @contextmanager
    def _frozen_teacher_context(self):
        """Load the frozen model for generation/reward/likelihood and offload it."""
        with self.load_teacher_model_context(), disable_gradient_checkpointing(
            self.teacher_model, self.args.gradient_checkpointing_kwargs
        ):
            teacher = self.accelerator.unwrap_model(self.teacher_model)
            was_training = teacher.training
            teacher.eval()
            try:
                yield teacher
            finally:
                teacher.train(was_training)

    def _teacher_prompt_samples(self, samples):
        """Clone prompt-only samples for frozen-model generation."""
        result = []
        for sample in samples:
            prompt = []
            for message in sample.messages:
                if message.get("role") == "assistant":
                    break
                prompt.append(deepcopy(message))
            result.append(type(sample)(
                messages=prompt,
                videos=deepcopy(sample.videos),
                audios=deepcopy(sample.audios),
                images=deepcopy(sample.images),
                tools=deepcopy(sample.tools),
                objects=deepcopy(sample.objects),
                extra=deepcopy(sample.extra),
            ))
        return result

    def _generate_frozen_teacher(self, samples):
        """Generate one trajectory per sample using the frozen GRPO model."""
        prompts = self._teacher_prompt_samples(samples)
        for sample in prompts:
            sample.encoded = encode_sample(sample, self.template)
            sample.encoded.pop("_extra_kwargs", None)
        batch = self.template.data_collator([sample.encoded for sample in prompts])
        device = self.accelerator.device
        model_inputs = {}
        for key, value in batch.items():
            if key in {
                "labels", "loss_scale", "response_start",
                "text_position_ids", "cu_seq_lens_q", "cu_seq_lens_k",
                "max_length_q", "max_length_k", "logits_to_keep",
                "sam_pixel_values", "gt_masks", "sam_resize_sizes",
                "sam_original_sizes", "sam_frame_paths", "mask_paths",
                "class_name", "sample_id", "dataset",
            }:
                continue
            model_inputs[key] = value.to(device) if torch.is_tensor(value) else value
        input_length = int(model_inputs["input_ids"].shape[1])
        with self._frozen_teacher_context() as teacher:
            with torch.no_grad(), self.template.generate_context():
                generated = teacher.generate(
                    **model_inputs,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=self.online_teacher_temperature,
                    top_p=self.online_teacher_top_p,
                    eos_token_id=self.template.tokenizer.eos_token_id,
                    pad_token_id=self.template.tokenizer.pad_token_id,
                )
        responses = generated[:, input_length:].detach().cpu()
        eos_id = self.template.tokenizer.eos_token_id
        pad_id = self.template.tokenizer.pad_token_id
        trimmed_responses = []
        texts = []
        for ids in responses:
            seq = ids.tolist()
            if eos_id is not None and eos_id in seq:
                seq = seq[: seq.index(eos_id) + 1]
            if pad_id is not None and pad_id != eos_id and pad_id in seq:
                seq = seq[: seq.index(pad_id)]
            trimmed_responses.append(torch.tensor(seq, dtype=ids.dtype))
            texts.append(
                self.template.tokenizer.decode(
                    seq, skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        return prompts, trimmed_responses, texts

    def _score_frozen_teacher(self, samples):
        prompts, response_ids, texts = self._generate_frozen_teacher(samples)
        messages = []
        for prompt, text in zip(prompts, texts):
            messages.append(deepcopy(prompt.messages) + [
                {"role": "assistant", "content": text}
            ])
        frame_paths = [s.extra["sam_frame_paths"] for s in samples]
        mask_paths = [s.extra["mask_paths"] for s in samples]
        audios = [s.audios for s in samples]
        # The mask verifier must use the frozen teacher's projector/SAM2 sidecar,
        # not the trainable Student decoder.  AuroraSAM2MaskIoUORM obtains its
        # model through the trainer, so temporarily switch that reference.
        original_model = self.model
        with self._frozen_teacher_context():
            self.model = self.teacher_model
            try:
                ious = AuroraSAM2MaskIoUORM()(
                    completions=texts, messages=messages,
                    sam_frame_paths=frame_paths, mask_paths=mask_paths,
                    audios=audios,
                )
            finally:
                self.model = original_model
        results = []
        for sample, ids, text, iou in zip(samples, response_ids, texts, ious):
            valid_seg = text.count("[SEG]") == 1
            results.append({
                "completion": text,
                "response_ids": ids,
                "predicted_class": extract_target_class(text),
                "class_correct": bool(class_exact_score(
                    text, sample.extra.get("class_name"), sample.extra.get("sample_id")
                )),
                "iou": float(iou) if valid_seg else 0.0,
                "seg_valid": valid_seg,
                "policy_step": int(self.state.global_step),
            })
        return results

    def _score_completions(self, samples):
        # Keep the two existing current-policy privileged Teacher trajectories.
        samples = super()._score_completions(samples)
        frozen = self._score_frozen_teacher(samples)
        for sample, result in zip(samples, frozen):
            for key, value in result.items():
                sample.extra[f"grpo_anchor_teacher_{key}"] = value
            self.write_verified_audit({
                "kind": "online_teacher_trajectory",
                "step": int(self.state.global_step),
                "teacher": "grpo_anchor",
                "sample_id": sample.extra.get("sample_id"),
                "modality": sample_template_modality(sample),
                "enabled": True,
                **{key: value for key, value in result.items() if key != "response_ids"},
            })
        return samples

    def _frozen_teacher_logps(self, batch_encoded):
        """Score the Student prompt/response with the frozen GRPO model."""
        grpo_batch = batch_encoded["grpo_batch"]
        model_inputs = batch_encoded["model_inputs"]
        with self._frozen_teacher_context():
            return self._get_per_token_logps_and_entropies(
                self.teacher_model,
                model_inputs,
                grpo_batch,
                origin_data=None,
            )[0].detach()

    def _select_anchor_tokens(self, chunk, grpo_batch, anchor_logps):
        """Use the existing 10%-plus-radius selector over all response stages."""
        selected_rows, seed_rows = [], []
        allowed = {stage.lower().replace(" ", "_") for stage in STAGES}
        for row_index, sample in enumerate(chunk):
            response_ids = flatten_response_ids(sample.response_token_ids)
            seed, selected, _, _ = self._select_verified_tokens(
                self.template.tokenizer,
                response_ids,
                grpo_batch.completion_mask[row_index],
                anchor_logps[row_index],
                allowed,
            )
            seed_rows.append(seed)
            selected_rows.append(selected)
        return torch.stack(seed_rows), torch.stack(selected_rows)

    def _prepare_batch_inputs(self, samples):
        batches = super()._prepare_batch_inputs(samples)
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batches):
            raise ValueError("Three-teacher chunks do not align with GRPO batches")
        for batch_encoded, chunk in zip(batches, chunks):
            anchor_logps = self._frozen_teacher_logps(batch_encoded)
            seed, selected = self._select_anchor_tokens(
                chunk, batch_encoded["grpo_batch"], anchor_logps
            )
            batch_encoded["_verified_teacher_logps"] = torch.cat(
                [batch_encoded["_verified_teacher_logps"], anchor_logps[:, None, :]],
                dim=1,
            )
            batch_encoded["_verified_seed_masks"] = torch.cat(
                [batch_encoded["_verified_seed_masks"], seed[:, None, :]], dim=1
            )
            batch_encoded["_verified_selected_masks"] = torch.cat(
                [batch_encoded["_verified_selected_masks"], selected[:, None, :]], dim=1
            )
        return batches

    def _postprocess_batch(self, samples, batch_encoded_inputs):
        # Same task-advantage construction as the prior code base, generalized
        # to three verifier-licensed teachers.
        for batch in batch_encoded_inputs:
            batch["grpo_batch"].teacher_per_token_logps = None
        # Reuse the Student reward postprocessing done by the GRPO parent.
        # (Calling AuroraOnlineVerifiedOPDTrainer here would execute its
        # two-teacher routing implementation, so call the native GRPO method.)
        GRPOTrainer._postprocess_batch(self, samples, batch_encoded_inputs)

        from swift_plugin.verified_opd_routing import add_verified_reverse_kl_advantage
        mode = "train" if self.model.training else "eval"
        for batch in batch_encoded_inputs:
            chunk = batch["_verified_samples"]
            grpo_batch = batch["grpo_batch"]
            device = grpo_batch.completion_mask.device
            student_class = torch.tensor(
                [bool(sample.extra["_verified_student_class"] > 0.5) for sample in chunk],
                device=device,
            )
            student_iou = torch.tensor(
                [float(sample.extra["_verified_student_iou"]) for sample in chunk],
                device=device,
            )
            teacher_class = torch.tensor([
                [
                    bool(sample.extra.get("audio_teacher_class_correct")),
                    bool(sample.extra.get("visual_teacher_class_correct")),
                    bool(sample.extra.get("grpo_anchor_teacher_class_correct")),
                ]
                for sample in chunk
            ], device=device)
            teacher_iou = torch.tensor([
                [
                    float(sample.extra.get("audio_teacher_iou") or 0),
                    float(sample.extra.get("visual_teacher_iou") or 0),
                    float(sample.extra.get("grpo_anchor_teacher_iou") or 0),
                ]
                for sample in chunk
            ], device=device)
            teacher_valid = torch.tensor([
                [
                    sample_template_modality(sample) in {"Audio", "Audio-Visual"}
                    and bool(sample.extra.get("audio_teacher_completion")),
                    bool(sample.extra.get("visual_teacher_completion")),
                    bool(sample.extra.get("grpo_anchor_teacher_completion")),
                ]
                for sample in chunk
            ], device=device)
            routing = self._compute_verified_routing(
                chunk, student_class, student_iou,
                teacher_class, teacher_iou, teacher_valid,
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
                positions = valid_positions[:token_count]
                assigned = details["teacher_index"][row_index, positions]
                self.write_verified_audit({
                    "kind": "trajectory",
                    "step": self.state.global_step,
                    "training": self.model.training,
                    "sample_id": sample.extra.get("sample_id"),
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
                            "verified_weight": float(routing["verified_weight"][row_index, index]),
                        }
                        for index, name in enumerate(TEACHER_NAMES)
                    },
                    "token_pieces": batch["_verified_pieces"][row_index][:token_count],
                    "stages": batch["_verified_stages"][row_index][:token_count],
                    "seed_masks": batch["_verified_seed_masks"][
                        row_index, :, positions
                    ].detach().cpu().tolist(),
                    "selected_masks": batch["_verified_selected_masks"][
                        row_index, :, positions
                    ].detach().cpu().tolist(),
                    "teacher_assignment": assigned.detach().cpu().tolist(),
                    "token_weight": details["token_weight"][
                        row_index, positions
                    ].detach().cpu().tolist(),
                    "signed_logratio": details["signed_logratio"][
                        row_index, positions
                    ].detach().cpu().tolist(),
                    "k3": details["k3"][row_index, positions].detach().cpu().tolist(),
                    "kd_advantage": details["kd_advantage"][
                        row_index, positions
                    ].detach().cpu().tolist(),
                })


TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "swift_plugin.aurora_online_three_teacher_joint.AuroraOnlineThreeTeacherJointTrainer"
)
