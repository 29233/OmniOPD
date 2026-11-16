"""Opt-in full-completion ablation of the original online Joint OPD.

Only token localization changes: every native completion_mask position,
including SEG/EOS/template-added suffixes, is a candidate for applicable views.
Original rewards, category/IoU licenses, hard routing and native signed OPD
advantages are inherited unchanged. No changes to historical entrypoints.
"""
from __future__ import annotations

from collections import Counter

import torch
from swift.trainers.trainer_factory import TrainerFactory

from swift_plugin.aurora_online_verified_opd import AuroraOnlineVerifiedOPDTrainer
from swift_plugin.aurora_verified_routed_opd import token_stages
from swift_plugin.verified_opd_routing import add_verified_reverse_kl_advantage


def full_completion_mask(pieces, completion_mask, enabled):
    """Never select prompt/padding; do include all valid native suffix tokens."""
    count = int(completion_mask.bool().sum())
    if abs(len(pieces) - count) > 2:
        raise ValueError("Full-response pieces/native-completion positions mismatch")
    return (completion_mask.detach().bool().clone() if enabled
            else torch.zeros_like(completion_mask, dtype=torch.bool))


class AuroraFullResponseOPDTrainer(AuroraOnlineVerifiedOPDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.task_reward_coef != 1 or self.fraction != 1 or self.radius != 0:
            raise ValueError("Full-response Joint requires task_coef=1/fraction=1/radius=0")
        if self.enabled_teachers != (True, True) or self.verified_config.kd_coef != .25:
            raise ValueError("Full-response ablation must preserve both Teachers and kd_coef=.25")
        self.write_verified_audit({
            "kind": "full_response_recipe", "overrides_historical_recipe": True,
            "teacher_stage_masks": {"audio": "all native completion positions when audio-applicable",
                                    "visual": "all native completion positions"},
            "includes": ["Reference analysis", "Audio evidence", "Visual evidence",
                         "Cross-modal reasoning", "Decision", "Segmentation", "SEG", "EOS",
                         "valid template-added completion suffix positions"],
            "excludes": ["prompt", "media input positions", "padding"],
            "fraction": 1., "radius": 0, "lowest_likelihood_selection": False,
            "license": "UNCHANGED original category rescue OR both-correct IoU gain>.01",
            "reward": ".5*class_exact + .5*class_exact*mean_10_frame_iou",
            "task_reward_coef": 1., "kd_coef": .25,
            "direction": "native teacher_logp - old_student_logp, not k3",
        })

    def _select_verified_tokens(self, tokenizer, response_ids, completion_mask,
                                teacher_logps, allowed):
        stages, pieces = token_stages(tokenizer, response_ids)
        selected = full_completion_mask(pieces, completion_mask, bool(allowed))
        return selected.clone(), selected, stages, pieces

    def _compute_verified_routing(self, *args, **kwargs):
        routing = super()._compute_verified_routing(*args, **kwargs)
        self._full_response_routing.append(routing)
        return routing

    def _postprocess_batch(self, samples, batches):
        self._full_response_routing = []
        super()._postprocess_batch(samples, batches)
        if len(self._full_response_routing) != len(batches):
            raise RuntimeError("Full-response audit routing/batch alignment failed")
        for batch, routing in zip(batches, self._full_response_routing):
            g = batch["grpo_batch"]
            masks = batch["_verified_selected_masks"]
            # Recompute detached diagnostics using exactly the native advantage
            # utility already used by the inherited production optimizer path.
            _, details = add_verified_reverse_kl_advantage(
                torch.zeros_like(g.advantages), g.old_per_token_logps,
                batch["_verified_teacher_logps"], masks, routing, g.completion_mask,
                self.verified_config, task_reward_coef=1.,
            )
            for i, sample in enumerate(batch["_verified_samples"]):
                valid = torch.nonzero(g.completion_mask[i], as_tuple=False).flatten()
                active = details["teacher_index"][i, valid].ge(0)
                pieces = batch["_verified_pieces"][i]
                named = min(len(valid), len(pieces))
                assignments = details["teacher_index"][i, valid]
                licensed = bool(routing["any_licensed"][i])
                if licensed and not bool(active.all()):
                    raise RuntimeError("Licensed full-response trajectory has inactive response positions")
                if not licensed and bool(active.any()):
                    raise RuntimeError("Unlicensed full-response trajectory received OPD")
                nonzero = details["kd_advantage"][i, valid].ne(0)
                self.write_verified_audit({
                    "kind": "full_response_coverage", "step": self.state.global_step,
                    "sample_id": sample.extra.get("sample_id"),
                    "native_completion_tokens": len(valid), "audited_piece_tokens": named,
                    "unlogged_template_suffix_tokens": len(valid) - named,
                    "candidate_tokens_per_teacher": masks[i, :, valid].sum(-1).cpu().tolist(),
                    "active_tokens": int(active.sum()), "nonzero_tokens": int((nonzero & active).sum()),
                    "any_licensed": licensed,
                    "teacher_counts": {name: int(assignments.eq(k).sum())
                                       for k, name in enumerate(("audio", "visual"))},
                    "active_stage_counts": dict(Counter(
                        batch["_verified_stages"][i][j] for j in range(named) if active[j])),
                    "active_suffix_tokens": int(active[named:].sum()),
                    "suffix_teacher_assignment": assignments[named:].cpu().tolist(),
                    "suffix_student_logps": g.old_per_token_logps[i, valid[named:]].float().cpu().tolist(),
                    "suffix_teacher_logps": details["chosen_teacher_logps"][i, valid[named:]].float().cpu().tolist(),
                    "suffix_kd_advantage": details["kd_advantage"][i, valid[named:]].float().cpu().tolist(),
                    "kd_abs_sum": float(details["kd_advantage"][i, valid].abs().sum()),
                })
        self._full_response_routing = []


TrainerFactory.TRAINER_MAPPING["grpo"] = (
    "swift_plugin.aurora_full_response_opd.AuroraFullResponseOPDTrainer"
)
