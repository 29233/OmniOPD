"""Sparse full-vocabulary KL for independently authorized privileged teachers."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrivilegedTop10Config:
    fraction: float = .10
    radius: int = 1
    direction: str = 'forward'
    temperature: float = 1.
    chunk_size: int = 16

    def __post_init__(self):
        if not 0 < self.fraction <= 1:
            raise ValueError('fraction must be in (0, 1]')
        if self.radius < 0 or self.chunk_size <= 0 or self.temperature <= 0:
            raise ValueError('radius must be nonnegative; chunk_size and temperature positive')
        if self.direction not in {'forward', 'reverse'}:
            raise ValueError("direction must be 'forward' or 'reverse'")


def select_low_likelihood(logits, token_ids, fraction, radius):
    """Lowest teacher likelihood positions, expanded by neighboring tokens."""
    if logits.ndim != 2 or token_ids.shape != (logits.shape[0],):
        raise ValueError('Expected [tokens,vocab] logits and matching sampled token IDs')
    count = max(1, int(torch.ceil(torch.tensor(len(token_ids) * fraction)).item()))
    with torch.no_grad():
        log_probs = F.log_softmax(logits.detach().float(), dim=-1)
        likelihood = log_probs.gather(-1, token_ids[:, None]).squeeze(-1)
        seed = torch.zeros(len(token_ids), dtype=torch.bool, device=logits.device)
        seed[torch.argsort(likelihood, stable=True)[:count]] = True
        selected = seed.clone()
        for offset in range(1, radius + 1):
            selected[offset:] |= seed[:-offset]
            selected[:-offset] |= seed[offset:]
    return seed, selected, likelihood


def privileged_top10_loss(student, teachers, token_ids, config):
    """Average independent sparse KL terms; teachers never receive gradients."""
    if len(teachers) != 2 or not student.shape[0] or token_ids.shape != (student.shape[0],):
        raise ValueError('Expected two teachers and a nonempty sampled response')
    if any(teacher.shape != student.shape for teacher in teachers):
        raise ValueError('Teacher/student response or vocabulary shapes differ')
    selected = [select_low_likelihood(teacher, token_ids, config.fraction, config.radius)
                for teacher in teachers]
    per_teacher_losses, per_teacher_kls = [], []
    for teacher, (_, mask, _) in zip(teachers, selected):
        total = student.sum() * 0.
        for start in range(0, student.shape[0], config.chunk_size):
            end = min(start + config.chunk_size, student.shape[0])
            log_s = F.log_softmax(student[start:end].float() / config.temperature, dim=-1)
            log_t = F.log_softmax(teacher[start:end].detach().float() / config.temperature, dim=-1)
            if config.direction == 'forward':
                kl = (log_t.exp() * (log_t - log_s)).sum(-1)
            else:
                kl = (log_s.exp() * (log_s - log_t)).sum(-1)
            total = total + (kl.clamp_min(0) * mask[start:end]).sum()
        per_teacher_losses.append(total / mask.sum().clamp_min(1))
        per_teacher_kls.append(per_teacher_losses[-1].detach())
    loss = torch.stack(per_teacher_losses).mean()
    seeds, masks, likelihoods = zip(*selected)
    overlap = masks[0] & masks[1]
    metrics = {
        'kd_loss': loss.detach(), 'audio_kl': per_teacher_kls[0], 'visual_kl': per_teacher_kls[1],
        'audio_seed_rate': seeds[0].float().mean(), 'visual_seed_rate': seeds[1].float().mean(),
        'audio_selected_rate': masks[0].float().mean(), 'visual_selected_rate': masks[1].float().mean(),
        'selected_overlap_rate': overlap.float().mean(),
    }
    return loss, metrics, {
        'token_ids': token_ids.detach(), 'seed_masks': torch.stack(seeds), 'selected_masks': torch.stack(masks),
        'teacher_logprobs': torch.stack(likelihoods),
    }
