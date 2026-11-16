"""Pure tensor utilities for verified two-teacher OPD routing.

The verifier is deliberately lexicographic:

* a GT-correct teacher may rescue a GT-wrong student regardless of IoU;
* when both are category-correct, the teacher must improve IoU by a margin;
* a category-wrong teacher is never licensed, even if its decoded mask happens
  to overlap the GT object.

Teacher probes decide *who may teach*.  Token likelihood masks decide *where*.
The optimization signal follows ms-swift's native OPD-RL implementation:
``teacher_logp - student_logp`` on the sampled token.  This signed k1
log-ratio is the policy-gradient reward for minimizing
``KL(student || teacher)``.  The non-negative k3 estimator is retained only as
an audit metric; it must not be detached and inserted as a PPO advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from swift.rl_core.advantage import compute_teacher_logratio


def parse_verified_teacher_selection(value: str) -> tuple[bool, bool]:
    """Return ``(audio_enabled, visual_enabled)`` for an OPD ablation mode."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "both": (True, True),
        "audio_visual": (True, True),
        "audio+visual": (True, True),
        "audio": (True, False),
        "audio_only": (True, False),
        "visual": (False, True),
        "visual_only": (False, True),
    }
    if normalized not in aliases:
        raise ValueError(
            "AURORA_VERIFIED_TEACHERS must be one of both, audio_only, or visual_only"
        )
    return aliases[normalized]


@dataclass(frozen=True)
class VerifiedRoutingConfig:
    iou_margin: float = 0.01
    category_rescue_bonus: float = 2.0
    iou_gain_weight: float = 0.75
    absolute_iou_weight: float = 0.25
    route_temperature: float = 0.25
    k3_monitor_clip: float = 10.0
    kd_coef: float = 0.25

    def __post_init__(self) -> None:
        values = torch.tensor([
            self.iou_margin,
            self.category_rescue_bonus,
            self.iou_gain_weight,
            self.absolute_iou_weight,
            self.route_temperature,
            self.k3_monitor_clip,
            self.kd_coef,
        ])
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Verified routing configuration must be finite")
        if self.iou_margin < 0 or self.category_rescue_bonus <= 1:
            raise ValueError("IoU margin must be nonnegative and category rescue must dominate IoU")
        if min(
            self.iou_gain_weight,
            self.absolute_iou_weight,
            self.route_temperature,
            self.k3_monitor_clip,
            self.kd_coef,
        ) <= 0:
            raise ValueError("Routing weights, temperature, clip and KD coefficient must be positive")
        if self.iou_gain_weight + self.absolute_iou_weight > 1.0 + 1e-8:
            raise ValueError("IoU score weights must sum to at most one")


def compute_verified_routing(
    student_class_correct: torch.Tensor,
    student_iou: torch.Tensor,
    teacher_class_correct: torch.Tensor,
    teacher_iou: torch.Tensor,
    teacher_valid: torch.Tensor | None = None,
    config: VerifiedRoutingConfig = VerifiedRoutingConfig(),
) -> dict[str, torch.Tensor]:
    """License and route ``K`` teachers for ``B`` sampled student trajectories.

    Shapes:
      * student scores: ``[B]``
      * teacher scores: ``[B, K]``

    ``class_license`` is a category rescue. ``iou_license`` requires both
    trajectories to be category-correct and a positive IoU margin.  Routing
    uses all three IoUs: teacher-vs-student gain plus absolute teacher IoU.
    """

    student_class = student_class_correct.detach().to(dtype=torch.bool)
    teacher_class = teacher_class_correct.detach().to(dtype=torch.bool)
    student_iou = student_iou.detach().float().clamp(0, 1)
    teacher_iou = teacher_iou.detach().float().clamp(0, 1)
    if student_class.ndim != 1 or student_iou.shape != student_class.shape:
        raise ValueError("Student verifier scores must have shape [B]")
    if teacher_class.ndim != 2 or teacher_iou.shape != teacher_class.shape:
        raise ValueError("Teacher verifier scores must have shape [B,K]")
    if teacher_class.shape[0] != student_class.shape[0]:
        raise ValueError("Student/teacher verifier batch sizes differ")
    valid = (
        torch.ones_like(teacher_class)
        if teacher_valid is None
        else teacher_valid.detach().to(device=teacher_class.device, dtype=torch.bool)
    )
    if valid.shape != teacher_class.shape:
        raise ValueError("teacher_valid must have shape [B,K]")

    student_class_k = student_class[:, None]
    iou_gain = teacher_iou - student_iou[:, None]
    class_license = valid & teacher_class & ~student_class_k
    iou_license = (
        valid
        & teacher_class
        & student_class_k
        & (iou_gain > config.iou_margin)
    )
    licensed = class_license | iou_license

    # Category rescue always outranks any mask-only gain because the bonus is
    # greater than the maximum possible total IoU contribution.
    score = (
        config.category_rescue_bonus * class_license.float()
        + config.iou_gain_weight * iou_gain.clamp_min(0)
        + config.absolute_iou_weight * teacher_iou
    )
    score = score.masked_fill(~licensed, float("-inf"))
    any_licensed = licensed.any(dim=1)
    route_prob = torch.zeros_like(teacher_iou)
    if bool(any_licensed.any()):
        route_prob[any_licensed] = torch.softmax(
            score[any_licensed] / config.route_temperature, dim=1
        )
    hard_teacher = torch.full(
        (len(student_class),), -1, device=teacher_iou.device, dtype=torch.long
    )
    hard_teacher[any_licensed] = route_prob[any_licensed].argmax(dim=1)

    max_score = config.category_rescue_bonus + config.iou_gain_weight + config.absolute_iou_weight
    strength = torch.zeros_like(teacher_iou)
    strength[licensed] = (
        score[licensed].clamp(min=0, max=max_score) / max_score
    )
    verified_weight = route_prob * strength
    return {
        "class_license": class_license,
        "iou_license": iou_license,
        "licensed": licensed,
        "iou_gain": iou_gain,
        "score": score,
        "route_prob": route_prob,
        "strength": strength,
        "verified_weight": verified_weight,
        "hard_teacher": hard_teacher,
        "any_licensed": any_licensed,
    }


def sampled_reverse_kl_k3(
    teacher_logps: torch.Tensor,
    student_logps: torch.Tensor,
    clip: float = 10.0,
) -> torch.Tensor:
    """Non-negative sampled-token estimator of ``KL(student || teacher)``."""

    if teacher_logps.shape != student_logps.shape:
        raise ValueError("Teacher/student sampled-token log-probability shapes differ")
    delta = teacher_logps.detach().float() - student_logps.detach().float()
    # Limit exp overflow before applying the paper's estimator clamp.
    delta = delta.clamp(min=-30.0, max=30.0)
    return (torch.exp(delta) - delta - 1.0).clamp(min=0.0, max=clip)


def route_token_teacher(
    selected_masks: torch.Tensor,
    verified_weight: torch.Tensor,
    completion_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Assign every selected token to at most one licensed teacher.

    Args:
        selected_masks: ``[B,K,T]`` localization masks.
        verified_weight: ``[B,K]`` teacher routing/strength weights.
        completion_mask: ``[B,T]`` valid response positions.
    """

    if selected_masks.ndim != 3:
        raise ValueError("selected_masks must have shape [B,K,T]")
    batch, teachers, tokens = selected_masks.shape
    if verified_weight.shape != (batch, teachers) or completion_mask.shape != (batch, tokens):
        raise ValueError("Token routing shapes do not align")
    candidates = (
        selected_masks.detach().bool()
        & completion_mask[:, None, :].detach().bool()
        & verified_weight[:, :, None].gt(0)
    )
    token_scores = verified_weight[:, :, None].expand_as(candidates).masked_fill(
        ~candidates, float("-inf")
    )
    any_candidate = candidates.any(dim=1)
    teacher_index = torch.full(
        (batch, tokens), -1, device=selected_masks.device, dtype=torch.long
    )
    teacher_index[any_candidate] = token_scores.argmax(dim=1)[any_candidate]
    token_weight = torch.zeros(
        (batch, tokens), device=verified_weight.device, dtype=verified_weight.dtype
    )
    if bool(any_candidate.any()):
        gathered = verified_weight.gather(
            1, teacher_index.clamp_min(0)
        )
        token_weight[any_candidate] = gathered[any_candidate]
    one_hot = torch.zeros_like(candidates)
    if bool(any_candidate.any()):
        one_hot.scatter_(1, teacher_index.clamp_min(0)[:, None, :], True)
        one_hot &= any_candidate[:, None, :]
    return {
        "teacher_index": teacher_index,
        "token_weight": token_weight,
        "assignment": one_hot,
        "any_candidate": any_candidate,
        "overlap_before_routing": candidates.sum(dim=1).gt(1),
    }


def add_verified_reverse_kl_advantage(
    base_advantages: torch.Tensor,
    old_student_logps: torch.Tensor,
    teacher_logps: torch.Tensor,
    selected_masks: torch.Tensor,
    routing: dict[str, torch.Tensor],
    completion_mask: torch.Tensor,
    config: VerifiedRoutingConfig = VerifiedRoutingConfig(),
    task_reward_coef: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add ms-swift-native signed OPD-RL token advantages.

    The native ms-swift signal is the sampled-token signed log-ratio
    ``teacher_logp - student_logp``.  Positive values increase the probability
    of a sampled token that the Teacher prefers; negative values suppress a
    sampled token that the Teacher disfavors.  Routing and verified weights
    only decide where and how strongly that native signal is applied.

    ``k3`` remains in the returned details for non-negative reverse-KL
    monitoring, but it is not used as a detached policy advantage.
    """

    if teacher_logps.ndim != 3:
        raise ValueError("teacher_logps must have shape [B,K,T]")
    if old_student_logps.shape != completion_mask.shape:
        raise ValueError("Student log-probability/completion-mask shapes differ")
    if teacher_logps.shape[0] != old_student_logps.shape[0] or teacher_logps.shape[2] != old_student_logps.shape[1]:
        raise ValueError("Teacher log-probability frame differs from student frame")
    token_route = route_token_teacher(
        selected_masks, routing["verified_weight"], completion_mask
    )
    gather_index = token_route["teacher_index"].clamp_min(0)[:, None, :]
    chosen_teacher_logps = teacher_logps.gather(1, gather_index).squeeze(1)
    native_signed_logratio = compute_teacher_logratio(
        chosen_teacher_logps,
        old_student_logps,
        completion_mask,
    ).float()
    # Keep the native value only on tokens licensed and selected by routing.
    # Inactive positions gather Teacher 0 merely for tensor shape alignment and
    # must not expose an apparent distillation signal in audits.
    signed_logratio = (
        native_signed_logratio * token_route["any_candidate"].float()
    )
    k3 = sampled_reverse_kl_k3(
        chosen_teacher_logps, old_student_logps, config.k3_monitor_clip
    )
    kd_advantage = (
        config.kd_coef
        * token_route["token_weight"]
        * signed_logratio
        * completion_mask.float()
    ).detach()
    if not 0.0 <= task_reward_coef <= 1.0:
        raise ValueError("task_reward_coef must be finite and between zero and one")
    # Zero removes *all* verifier-derived policy-gradient advantages. The
    # verifier can still license Teachers, but it must not optimize the pure
    # OPD control through an accidental GRPO reward path.
    combined = task_reward_coef * base_advantages + kd_advantage
    denominator = completion_mask.float().sum().clamp_min(1)
    active = token_route["any_candidate"]
    active_count = active.sum().clamp_min(1)
    metrics = {
        "kd_advantage_mean": kd_advantage.sum() / denominator,
        "kd_active_rate": active.float().sum() / denominator,
        "signed_logratio_active_mean": (
            (signed_logratio * active).sum() / active_count
        ),
        "signed_logratio_active_abs_mean": (
            (signed_logratio.abs() * active).sum() / active_count
        ),
        "kd_positive_rate": (
            (kd_advantage.gt(0) & active).float().sum() / active_count
        ),
        "kd_negative_rate": (
            (kd_advantage.lt(0) & active).float().sum() / active_count
        ),
        "k3_active_mean": (
            (k3 * active).sum() / active_count
        ),
        "overlap_before_routing_rate": (
            token_route["overlap_before_routing"].float().sum() / denominator
        ),
        "class_license_rate": routing["class_license"].float().mean(),
        "iou_license_rate": routing["iou_license"].float().mean(),
        "any_license_rate": routing["any_licensed"].float().mean(),
    }
    return combined, {
        **token_route,
        "chosen_teacher_logps": chosen_teacher_logps,
        "signed_logratio": signed_logratio,
        "k3": k3,
        "kd_advantage": kd_advantage,
        "metrics": metrics,
    }
