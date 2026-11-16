"""Pure OPD with audio- and visual-text privileged frozen-teacher views.

This plugin deliberately contains no reward model, GRPO objective, mask BCE/Dice,
or dense background KL.  It is used with ms-swift GKD solely for on-policy
rollout collection and teacher-forced sparse full-vocabulary distillation.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

from swift.rl_core.data import GKDBatch
from swift.rlhf_trainers.utils import encode_sample
from swift.trainers import disable_gradient_checkpointing
from swift.trainers.trainer_factory import TrainerFactory
from swift.utils import to_device
from swift_plugin.aurora_opsd import AuroraGKDTrainer
from swift_plugin.dual_opsd_loss import active_response, assert_response_alignment
from swift_plugin.privileged_top10_loss import PrivilegedTop10Config, privileged_top10_loss


@contextmanager
def temporary_eval(model):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.training = was_training


def inject(user_content, prefix, evidence):
    marker = '<video><audio>'
    if user_content.count(marker) != 1:
        raise ValueError(f'Expected exactly one {marker} marker in privileged OPD prompt')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError(f'Missing {prefix} privilege evidence')
    return user_content.replace(marker, marker + f'{prefix}{evidence.strip()}\n', 1)


def make_views(sample):
    if sample.teacher_prompt or sample.extra.get('teacher_audio_paths'):
        raise ValueError('Privileged-top10 OPD requires audio_privilege_text and visual_privilege_text columns only')
    student, audio, visual = [deepcopy(sample) for _ in range(3)]
    audio_text = sample.extra.get('audio_privilege_text')
    visual_text = sample.extra.get('visual_privilege_text')
    for view in (student, audio, visual):
        # Pure OPD: neither rollout scoring nor teacher scoring runs SAM loss.
        view.extra.pop('sam_frame_paths', None)
        view.extra.pop('mask_paths', None)
    for view, prefix, evidence in (
        (audio, 'Privileged clip-level audio evidence: ', audio_text),
        (visual, 'Hint: Visual evidence about the target: ', visual_text),
    ):
        user = next((message for message in reversed(view.messages) if message['role'] == 'user'), None)
        if user is None or not isinstance(user.get('content'), str):
            raise ValueError('Expected a string user prompt')
        user['content'] = inject(user['content'], prefix, evidence)
    return student, audio, visual


class SparseUpdateAudit(TrainerCallback):
    """Record whether the sparse loss produces a real, finite optimizer update."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.before = {}

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # This hook is after clipping/unscaling, so it observes the gradients
        # the optimizer will actually consume.
        model = kwargs.get('model') or self.trainer.model
        finite, nonfinite, total_elements = 0, 0, 0
        self.before = {}
        selected_groups = set()
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            total_elements += param.grad.numel()
            is_finite = bool(torch.isfinite(param.grad).all())
            finite += int(is_finite)
            nonfinite += int(not is_finite)
            group = ('lora_B' if 'lora_B' in name else 'lora_A' if 'lora_A' in name else
                     'projector' if 'text_hidden_fcs' in name else
                     'lm_head' if 'lm_head' in name else None)
            if group is None or group in selected_groups:
                continue
            selected_groups.add(group)
            count = min(2048, param.numel())
            # Do not use float linspace here: parameters larger than 2^24
            # elements can round its endpoint beyond the valid final index.
            ids = (torch.arange(count, device=param.device, dtype=torch.long)
                   * (param.numel() - 1) // max(count - 1, 1))
            self.before[name] = (param, ids, param.detach().flatten()[ids].clone(), group)
        self.trainer.write_audit({
            'kind': 'pre_optimizer', 'step': state.global_step + 1,
            'gradient_tensors_finite': finite, 'gradient_tensors_nonfinite': nonfinite,
            'gradient_elements': total_elements,
            'representative_groups': sorted(selected_groups),
        })

    def on_optimizer_step(self, args, state, control, **kwargs):
        updates = {}
        for name, (param, ids, previous, group) in self.before.items():
            delta = (param.detach().flatten()[ids] - previous).float()
            updates[group] = {
                'name': name, 'dtype': str(param.dtype), 'sampled_elements': len(ids),
                'changed_elements': int(delta.ne(0).sum()),
                'mean_abs_update': float(delta.abs().mean()),
                'all_finite': bool(torch.isfinite(delta).all()),
            }
        self.trainer.write_audit({
            'kind': 'optimizer_update', 'step': state.global_step + 1,
            'optimizer_step_was_skipped': bool(self.trainer.accelerator.optimizer_step_was_skipped),
            'updates': updates,
        })
        self.before = {}


class AuroraPrivilegedTop10OPDTrainer(AuroraGKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._is_self_distillation or self.use_teacher_api or self._teacher_use_disable_adapter or self.teacher_model is None:
            raise ValueError('Privileged-top10 OPD requires a local independent frozen SFT teacher')
        if self.lmbda != 1 or self.beta != 0 or self.gkd_logits_topk is not None or self.args.sft_alpha != 0:
            raise ValueError('Privileged-top10 OPD requires lmbda=1, beta=0, full vocabulary and sft_alpha=0')
        if float(os.environ.get('AURORA_OPSD_MASK_WEIGHT', '0')) != 0 or float(os.environ.get('AURORA_OPSD_SEG_DISTILL_WEIGHT', '0')) != 0:
            raise ValueError('Privileged-top10 OPD is language-only: mask and SEG feature losses must be zero')
        self.config = PrivilegedTop10Config(
            fraction=float(os.environ.get('AURORA_PRIVILEGED_TOP10_FRACTION', '.10')),
            radius=int(os.environ.get('AURORA_PRIVILEGED_TOP10_RADIUS', '1')),
            direction=os.environ.get('AURORA_PRIVILEGED_TOP10_DIRECTION', 'forward'),
            temperature=float(os.environ.get('AURORA_PRIVILEGED_TOP10_TEMPERATURE', '1')),
            chunk_size=int(os.environ.get('AURORA_PRIVILEGED_TOP10_CHUNK_SIZE', '16')),
        )
        if self.temperature != self.config.temperature:
            raise ValueError('Generation and OPD temperatures must match')
        teacher = self.accelerator.unwrap_model(self.teacher_model)
        if not getattr(teacher, 'peft_config', None):
            raise ValueError('Frozen teacher must load the SFT adapter')
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        teacher.eval()
        # Keep AdamW master parameters in FP32. The A100 runs the forward under
        # BF16 autocast, preventing FP16 scaler overflows in sparse full-vocab KL.
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        self.add_callback(SparseUpdateAudit(self))
        self.write_audit({'kind': 'recipe', 'objective': 'pure sparse OPD, no RL/reward/mask loss',
                          'config': self.config.__dict__, 'teacher_authorization': 'both always enabled',
                          'trainable_storage_dtype': 'float32',
                          'teacher_base': os.environ.get('TEACHER_MODEL'),
                          'teacher_adapters': os.environ.get('TEACHER_ADAPTERS')})

    def write_audit(self, record):
        folder = Path(self.args.output_dir) / 'privileged_top10_audit'
        folder.mkdir(parents=True, exist_ok=True)
        record = {**record, 'rank': self.accelerator.process_index}
        with (folder / f'rank{self.accelerator.process_index}.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _engine_infer(self, *args, **kwargs):
        # Colocated rollout generation bypasses Accelerate's train-forward
        # autocast. The SFT initializer was FP16, but this pure-OPD run uses
        # BF16 compute on A100 to avoid GradScaler-overflow skipped updates.
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            return super()._engine_infer(*args, **kwargs)

    def _prepare_batch_inputs(self, samples):
        batches = []
        for chunk in self.split_by_mini_batches(samples):
            if len(chunk) != 1:
                raise ValueError('Privileged-top10 OPD requires per-device microbatch=1')
            original = chunk[0]
            views = make_views(original)
            encoded = []
            with self._template_context(self.template):
                for view in views:
                    row = encode_sample(view, self.template)
                    row.pop('_extra_kwargs', None)
                    encoded.append(to_device(self.template.data_collator([row]), self.accelerator.device))
            student, audio, visual = encoded
            for teacher in (audio, visual):
                assert_response_alignment(student['labels'], teacher['labels'])
            batches.append({'model_inputs': student, 'privileged_teacher_inputs': [audio, visual],
                            'evidence': {'sample_id': original.extra.get('sample_id'),
                                         'audio_evidence': original.extra.get('audio_privilege_text'),
                                         'visual_evidence': original.extra.get('visual_privilege_text')},
                            'gkd_batch': GKDBatch(data_source=self._data_source)})
        return batches

    def _compute_aurora_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_inputs = dict(inputs['model_inputs'])
        teacher_inputs = [dict(item) for item in inputs['privileged_teacher_inputs']]
        for teacher in teacher_inputs:
            assert_response_alignment(student_inputs['labels'], teacher['labels'])
        if self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1):
            for row in [student_inputs, *teacher_inputs]:
                self.prepare_logits_to_keep(row)
        raw_teacher = self.accelerator.unwrap_model(self.teacher_model)
        teacher_scores, teacher_ids = [], []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False), \
                temporary_eval(raw_teacher), disable_gradient_checkpointing(
                    self.teacher_model, self.args.gradient_checkpointing_kwargs), self.load_teacher_model_context():
            for row in teacher_inputs:
                output = self.teacher_model(**{key: value for key, value in row.items() if key != 'labels'},
                                            aurora_skip_hidden_states=True)
                scores, token_ids = active_response(output.logits, row['labels'])
                teacher_scores.append(scores.detach()); teacher_ids.append(token_ids)
                del output
        outputs = model(**{key: value for key, value in student_inputs.items() if key != 'labels'},
                        aurora_skip_hidden_states=True, output_hidden_states=False)
        scores, token_ids = active_response(outputs.logits, student_inputs['labels'])
        if any(not torch.equal(token_ids, expected) for expected in teacher_ids):
            raise ValueError('Teacher/student sampled response tokens do not align')
        loss, metrics, details = privileged_top10_loss(scores, teacher_scores, token_ids, self.config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite privileged-top10 OPD loss')
        self._aurora_mask_parts = {f'privileged_top10/{key}': self.accelerator.reduce(value, reduction='mean')
                                   for key, value in metrics.items()}
        self.write_audit({'kind': 'loss', 'step': self.state.global_step, 'training': self.model.training,
                          **inputs['evidence'], 'metrics': {key: float(value) for key, value in metrics.items()},
                          'token_ids': details['token_ids'].cpu().tolist(),
                          'seed_masks': details['seed_masks'].cpu().tolist(),
                          'selected_masks': details['selected_masks'].cpu().tolist(),
                          'teacher_logprobs': details['teacher_logprobs'].cpu().tolist(),
                          'teacher_targets_require_grad': any(value.requires_grad for value in teacher_scores)})
        return (loss, outputs) if return_outputs else loss


TrainerFactory.TRAINER_MAPPING['gkd'] = 'swift_plugin.aurora_privileged_top10_opd.AuroraPrivilegedTop10OPDTrainer'
