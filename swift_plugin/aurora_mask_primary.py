"""No LM judge: segmentation-primary GRPO, optionally current-policy Decision OPD.

All legacy entrypoints retain their previous recipe; this plugin is opt-in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from accelerate.utils import gather_object
from swift.rlhf_trainers.utils import get_even_process_data
from swift.trainers.trainer_factory import TrainerFactory

from swift_plugin.aurora_online_verified_opd import AuroraOnlineVerifiedOPDTrainer
from swift_plugin.aurora_sam2_grpo import AuroraSAM2GRPOTrainer
from swift_plugin.aurora_verified_reward_grpo import AuroraVerifiedRewardGRPOTrainer
from swift_plugin.aurora_verified_routed_opd import token_stages
from swift_plugin.mask_primary_utils import (
    decision_category_mask, decision_license_valid, mask_primary_rewards, valid_seg_text,
)
from swift_plugin.verified_opd_routing import compute_verified_routing


def sampled_text(sample):
    return next((m["content"] for m in reversed(sample.messages) if m["role"] == "assistant"), "")


def transform_rewards(trainer, samples, raw):
    alpha = float(os.environ.get("AURORA_MASK_CLASS_ALPHA", ".2"))
    local_valid = [valid_seg_text(sampled_text(s)) for s in samples]
    flags = gather_object(local_valid)
    valid = torch.tensor(flags, device=raw.device, dtype=torch.bool)
    converted = mask_primary_rewards(raw, valid, alpha)
    local_raw = get_even_process_data(trainer, raw)
    local_converted = get_even_process_data(trainer, converted)
    folder = Path(trainer.args.output_dir) / "mask_primary_audit"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"rank{trainer.accelerator.process_index}.jsonl").open("a") as sink:
        for s, v, r, c in zip(samples, local_valid, local_raw, local_converted):
            sink.write(json.dumps({
                "kind": "reward", "step": trainer.state.global_step,
                "sample_id": s.extra.get("sample_id"), "valid_seg": v,
                "raw_class": float(r[0]), "raw_iou": float(r[1]),
                "alpha": alpha, "effective_reward": float(c.sum() * .5),
                "raw_columns": r.detach().cpu().tolist(),
                "native_scaled_columns": c.detach().cpu().tolist(),
            }) + "\n")
    return converted


class AuroraMaskPrimaryGRPOTrainer(AuroraVerifiedRewardGRPOTrainer):
    def _score_completions(self, samples):
        # Deliberately bypass the historical category-multiplied reward column.
        samples = AuroraSAM2GRPOTrainer._score_completions(self, samples)
        self._rewards_per_func = transform_rewards(self, samples, self._rewards_per_func)
        return samples


class AuroraMaskPrimaryDecisionTrainer(AuroraOnlineVerifiedOPDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_reward_coef != 1:
            raise ValueError("Decision experiment requires the same GRPO task reward as its control")
        self.write_verified_audit({
            "kind": "mask_primary_decision_recipe",
            "overrides_historical_recipe": True,
            "reward": "valid_SEG*(.2*class_exact+.8*mean_10_frame_iou)",
            "teacher_stage_masks": {"audio": ["Decision category span only"],
                                    "visual": ["Decision category span only"]},
            "license": "class rescue only; teacher valid SEG; IoU>=min and no material mask regression",
            "min_teacher_iou": float(os.environ.get("AURORA_DECISION_MIN_IOU", ".5")),
            "mask_regression_tolerance": float(os.environ.get("AURORA_DECISION_IOU_TOLERANCE", ".01")),
            "teacher": "current policy adapters enabled; refreshed every update; no LM judge",
        })

    def _transform_verified_rewards(self, samples, raw_rewards):
        return transform_rewards(self, samples, raw_rewards)

    def _select_verified_tokens(self, tokenizer, response_ids, completion_mask,
                                teacher_logps, allowed):
        stages, pieces = token_stages(tokenizer, response_ids)
        # allowed is empty for disabled views and audio-inapplicable samples.
        mask = (decision_category_mask(pieces, completion_mask) if allowed
                else torch.zeros_like(completion_mask, dtype=torch.bool))
        return mask, mask.clone(), stages, pieces

    def _compute_verified_routing(self, chunk, student_class, student_iou,
                                  teacher_class, teacher_iou, teacher_valid):
        seg_valid = torch.tensor(
            [[bool(s.extra.get(f"{name}_teacher_seg_valid"))
              for name in ("audio", "visual")] for s in chunk],
            device=teacher_valid.device, dtype=torch.bool,
        )
        valid = decision_license_valid(
            student_class, student_iou, teacher_class, teacher_iou,
            teacher_valid & seg_valid,
            min_iou=float(os.environ.get("AURORA_DECISION_MIN_IOU", ".5")),
            tolerance=float(os.environ.get("AURORA_DECISION_IOU_TOLERANCE", ".01")),
        )
        return compute_verified_routing(
            student_class, student_iou, teacher_class, teacher_iou,
            valid, self.verified_config,
        )


ARM = os.environ.get("AURORA_MASK_PRIMARY_ARM")
if ARM not in {"grpo", "decision"}:
    raise ValueError("AURORA_MASK_PRIMARY_ARM must be grpo or decision")
TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "swift_plugin.aurora_mask_primary.AuroraMaskPrimaryGRPOTrainer" if ARM == "grpo" else
    "swift_plugin.aurora_mask_primary.AuroraMaskPrimaryDecisionTrainer"
)
