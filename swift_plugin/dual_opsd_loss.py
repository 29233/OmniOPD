"""Full-vocabulary, gated KL for current-Student privileged views."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DualGateConfig:
    direction: str = 'forward'
    class_threshold: float = .02
    audio_threshold: float = .02
    class_scale: float = .1
    audio_scale: float = .1
    weight_cap: float = 2.
    base_weight: float = 1.
    gate_weight: float = 1.
    temperature: float = 1.
    chunk_size: int = 32

    def __post_init__(self):
        if self.direction not in {'forward', 'reverse'}:
            raise ValueError("direction must be 'forward' or 'reverse'")
        values = [v for k, v in self.__dict__.items() if k != 'direction']
        if not all(torch.isfinite(torch.tensor(v)) for v in values):
            raise ValueError('Nonfinite dual-gate configuration')
        if min(self.class_threshold, self.audio_threshold, self.base_weight, self.gate_weight) < 0:
            raise ValueError('Thresholds and loss coefficients must be nonnegative')
        if min(self.class_scale, self.audio_scale, self.weight_cap, self.temperature, self.chunk_size) <= 0:
            raise ValueError('Scales, cap, temperature and chunk size must be positive')


def active_response(logits, labels):
    """Swift labels may already be LM-head trimmed; shift exactly once."""
    if logits.shape[:2] != labels.shape:
        raise ValueError(f'Logit/label mismatch: {logits.shape[:2]} != {labels.shape}')
    shifted = torch.roll(labels, shifts=-1, dims=1)
    shifted[:, -1] = -100
    mask = shifted.ne(-100)
    return logits[mask], shifted[mask]


def assert_response_alignment(student_labels, teacher_labels):
    if student_labels.shape[0] != teacher_labels.shape[0]:
        raise ValueError('Response batch sizes differ')
    for a, b in zip(student_labels, teacher_labels):
        if not torch.equal(a[a.ne(-100)], b[b.ne(-100)]):
            raise ValueError('Response token IDs differ between privileged views')


def dual_distillation_loss(student, teachers, token_ids, validity, config):
    """Teachers and all routing weights are detached; only student gets gradients.

    Inputs are active response positions [N,V], and validity is [2,N].
    Base loss averages all valid teacher/token pairs. Gated loss averages
    positive log-ratio weights over those pairs; an empty gate gives zero.
    """
    if len(teachers) != 2 or validity.shape != (2, student.shape[0]):
        raise ValueError('Expected class/audio teachers and [2,N] validity')
    if not student.shape[0] or token_ids.shape != (student.shape[0],):
        raise ValueError('Empty or mismatched response tokens')
    if any(t.shape != student.shape for t in teachers):
        raise ValueError('Teacher and student vocabulary/response shapes differ')
    valid = validity.detach().to(device=student.device, dtype=torch.bool)
    thresholds = student.new_tensor([config.class_threshold, config.audio_threshold]).float()[:, None]
    scales = student.new_tensor([config.class_scale, config.audio_scale]).float()[:, None]
    base_sum, gate_sum = student.sum() * 0., student.sum() * 0.
    all_delta, all_kl, all_weights = [], [], []
    for start in range(0, len(student), config.chunk_size):
        end = min(start + config.chunk_size, len(student))
        log_s = F.log_softmax(student[start:end].float() / config.temperature, dim=-1)
        log_t = torch.stack([F.log_softmax(t[start:end].detach().float() / config.temperature, -1)
                             for t in teachers])
        with torch.no_grad():
            ids = token_ids[start:end, None]
            observed_s = log_s.detach().gather(-1, ids).squeeze(-1)
            observed_t = log_t.gather(-1, ids[None].expand(2, -1, -1)).squeeze(-1)
            delta = observed_t - observed_s[None]
            weights = ((delta - thresholds).clamp_min(0) / scales).clamp_max(config.weight_cap)
            weights = weights * valid[:, start:end]
        if config.direction == 'forward':
            kl = (log_t.exp() * (log_t - log_s[None])).sum(-1)
        else:
            kl = (log_s.exp()[None] * (log_s[None] - log_t)).sum(-1)
        kl = kl * config.temperature ** 2
        # Numerical roundoff can be slightly negative near identical views.
        kl = kl.clamp_min(0)
        base_sum = base_sum + (kl * valid[:, start:end]).sum()
        gate_sum = gate_sum + (kl * weights).sum()
        all_delta.append(delta.detach())
        all_kl.append(kl.detach())
        all_weights.append(weights)
    delta, kl, weights = [torch.cat(parts, dim=1) for parts in (all_delta, all_kl, all_weights)]
    base = base_sum / valid.sum().clamp_min(1)
    gated = gate_sum / weights.sum().clamp_min(1e-8)
    loss = config.base_weight * base + config.gate_weight * gated
    denom = weights.sum(0)
    route = weights / denom.clamp_min(1e-8)
    metrics = {'base_kl': base.detach(), 'gated_kl': gated.detach(), 'kd_loss': loss.detach(),
               'gate_any_rate': (denom > 0).float().mean(),
               'gate_none_rate': (denom == 0).float().mean(),
               'gate_both_rate': (weights > 0).all(0).float().mean()}
    for i, name in enumerate(('class', 'audio')):
        v = valid[i]
        n = v.sum().clamp_min(1)
        metrics.update({f'{name}_valid_rate': v.float().mean(),
                        f'{name}_delta_mean': (delta[i] * v).sum() / n,
                        f'{name}_kl': (kl[i] * v).sum() / n,
                        f'{name}_gate_rate': (weights[i] > 0).float().sum() / n,
                        f'{name}_weight_mean': weights[i].sum() / n,
                        f'{name}_route_mean_active': route[i].sum() / (denom > 0).sum().clamp_min(1)})
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite dual-view distillation loss')
    return loss, metrics, {'delta': delta, 'weights': weights, 'route': route, 'kl': kl,
                           'valid': valid, 'token_ids': token_ids.detach()}
