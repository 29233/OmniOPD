"""Feature alignment for the projected [SEG] prompt consumed by SAM."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def first_seg_hidden(hidden: torch.Tensor, labels: torch.Tensor, token_id: int):
    """Select the first response [SEG] in each row, retaining batch indices.

    Labels must describe the full hidden-state sequence, even when the LM
    head uses logits_to_keep to trim its logits and its loss labels.
    """
    if hidden.ndim != 3 or labels.ndim != 2 or hidden.shape[:2] != labels.shape:
        raise ValueError(f'SEG hidden/label shape mismatch: {tuple(hidden.shape)} vs {tuple(labels.shape)}')
    mask = labels.to(hidden.device).eq(token_id)
    rows = mask.any(-1).nonzero(as_tuple=True)[0]
    positions = mask.int().argmax(-1)[rows]
    return hidden[rows, positions], rows


def validate_response_alignment(student_labels: torch.Tensor, teacher_labels: torch.Tensor):
    """Privileged prompts can differ, but the scored rollout must be identical."""
    if student_labels.shape[0] != teacher_labels.shape[0]:
        raise ValueError('SEG distillation requires the same student/teacher batch size')
    for index, (student, teacher) in enumerate(zip(student_labels, teacher_labels)):
        student = student[student != -100]
        teacher = teacher[teacher != -100].to(student.device)
        if not torch.equal(student, teacher):
            raise ValueError(f'SEG distillation response mismatch in batch row {index}; check truncation/alignment')


def seg_prompt_distillation(student: torch.Tensor, teacher: torch.Tensor,
                           student_rows: torch.Tensor, teacher_rows: torch.Tensor,
                           loss_type: str = 'mse'):
    """Stop-gradient teacher targets; missing-SEG batches keep a zero graph.

    ``student`` includes one dummy prompt when student_rows is empty, to keep
    projector/DDP gradient participation consistent with mask supervision.
    """
    if loss_type not in {'mse', 'cosine'}:
        raise ValueError(f'Unsupported SEG distillation loss: {loss_type}')
    if not torch.equal(student_rows, teacher_rows.to(student_rows.device)):
        raise ValueError('SEG distillation student/teacher sample rows differ')
    student = student.float()
    teacher = teacher.detach().to(device=student.device, dtype=torch.float32)
    count = student_rows.numel()
    if count == 0:
        loss = student.sum() * 0.0
        zero = loss.detach()
        return loss, {key: zero for key in (
            'seg_distill_loss', 'seg_distill_mse', 'seg_distill_cos',
            'seg_distill_student_norm', 'seg_distill_teacher_norm', 'seg_distill_rows')}
    if student.ndim != 2 or student.shape != teacher.shape or student.shape[0] != count:
        raise ValueError(f'SEG prompt shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}')
    mse = F.mse_loss(student, teacher)
    cos = F.cosine_similarity(student, teacher, dim=-1).mean()
    loss = mse if loss_type == 'mse' else 1.0 - cos
    return loss, {
        'seg_distill_loss': loss.detach(),
        'seg_distill_mse': mse.detach(),
        'seg_distill_cos': cos.detach(),
        'seg_distill_student_norm': student.detach().norm(dim=-1).mean(),
        'seg_distill_teacher_norm': teacher.norm(dim=-1).mean(),
        'seg_distill_rows': student.new_tensor(float(count)),
    }
