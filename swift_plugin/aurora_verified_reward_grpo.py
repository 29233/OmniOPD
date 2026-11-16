"""Matched reward-only ablation for verified routed OPD.

It uses exactly the same category-gated class/mask reward and q/v-only policy
parameterization as ``aurora_verified_routed_opd.py``, but contains no Teacher,
license, routing, localization or distillation signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from swift.trainers.trainer_factory import TrainerFactory
from swift_plugin.aurora_sam2_grpo import AuroraSAM2GRPOTrainer


class AuroraVerifiedRewardGRPOTrainer(AuroraSAM2GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ms-swift may mark dynamic self-distillation capability as available
        # from dataset metadata even when no explicit Teacher model/server is
        # configured.  The reward-only arm never calls that path, so validate
        # the actual explicit Teacher handles instead of the aggregate flag.
        if self.teacher_model is not None or self.use_teacher_api:
            raise ValueError("Reward-only ablation must not load an explicit Teacher")
        reward_names = list(self.reward_func_names or [])
        accepted = (
            ["aurora_class_exact", "aurora_sam2_mask_iou"],
            ["AuroraClassExactORM", "AuroraSAM2MaskIoUORM"],
        )
        if reward_names not in accepted:
            raise ValueError(f"Unexpected reward-only functions: {reward_names!r}")
        weights = [float(value) for value in self.reward_weights.detach().cpu().tolist()]
        if weights != [0.5, 0.5]:
            raise ValueError(f"Unexpected reward-only weights: {weights}")
        trainable = []
        for name, parameter in self.model.named_parameters():
            enabled = "lora_A" in name or "lora_B" in name
            parameter.requires_grad_(enabled)
            if enabled:
                parameter.data = parameter.data.float()
                trainable.append(name)
        if not trainable:
            raise RuntimeError("Reward-only ablation left no q/v LoRA parameters trainable")
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        folder = Path(self.args.output_dir) / "verified_reward_audit"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"rank{self.accelerator.process_index}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps({
                "kind": "recipe",
                "objective": "category-gated class/mask GRPO only",
                "teacher": None,
                "distillation": None,
                "trainable": "q/v LoRA only",
                "rank": self.accelerator.process_index,
            }) + "\n")

    def _score_completions(self, samples):
        samples = super()._score_completions(samples)
        if self._rewards_per_func.shape[1] != 2:
            raise ValueError("Expected class and mask-IoU reward columns")
        self._rewards_per_func[:, 1] = (
            self._rewards_per_func[:, 0] * self._rewards_per_func[:, 1]
        )
        return samples


TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "swift_plugin.aurora_verified_reward_grpo.AuroraVerifiedRewardGRPOTrainer"
)
