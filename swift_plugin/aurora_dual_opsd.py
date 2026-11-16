"""ms-swift GKD: frozen-teacher class/audio privileged views and token gates."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from transformers import TrainerCallback

from swift.rl_core.data import GKDBatch
from swift.rlhf_trainers.utils import encode_sample
from swift.trainers import disable_gradient_checkpointing
from swift.trainers.trainer_factory import TrainerFactory
from swift.utils import to_device
from swift_plugin.aurora_opsd import AuroraGKDTrainer
from swift_plugin.dual_opsd_loss import (DualGateConfig, active_response, assert_response_alignment,
                                        dual_distillation_loss)

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from scripts.data.prepare_refavs_opsd import build_teacher_prompt, has_valid_class_privilege


@contextmanager
def temporary_eval(model):
    flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def tensor_digest(tensor):
    if tensor is None:
        return None
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def model_state_digest(model, chunk_elements=4 * 1024 * 1024):
    """Hash every parameter and buffer without materializing the full state dict."""
    digest = hashlib.sha256()
    tensors = [('parameter', name, value) for name, value in model.named_parameters()]
    tensors.extend(('buffer', name, value) for name, value in model.named_buffers())
    for kind, name, value in tensors:
        digest.update(f'{kind}:{name}:{value.dtype}:{tuple(value.shape)}\0'.encode())
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            chunk = flat[start:start + chunk_elements].to(device='cpu').contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_views(sample):
    """Copy before any encoding: Omni mutates audio lists while loading media."""
    if sample.teacher_prompt or sample.extra.get('teacher_audio_paths'):
        raise ValueError('Dual OPSD expects its own metadata, not single-teacher OPSD columns')
    cls = sample.extra.get('class_name')
    valid = has_valid_class_privilege(cls or '')
    if bool(sample.extra.get('dual_class_valid')) != valid:
        raise ValueError('Class privilege validity disagrees with dataset metadata')
    clean = sample.extra.get('dual_clean_audio_paths')
    if not clean or len(clean) != len(sample.audios) or list(clean) == list(sample.audios):
        raise ValueError('Missing or non-distinct clean/degraded audio inputs')
    student, category, audio = [deepcopy(sample) for _ in range(3)]
    for view in (category, audio):
        view.extra.pop('sam_frame_paths', None)
        view.extra.pop('mask_paths', None)
    if valid:
        user = next(m for m in reversed(category.messages) if m['role'] == 'user')
        if not isinstance(user['content'], str):
            raise ValueError('REF-AVS class hint expects a string user prompt')
        user['content'] = build_teacher_prompt(user['content'], cls, 'class')
    audio.audios = list(clean)
    if student.messages != audio.messages or student.audios != category.audios:
        raise AssertionError('Teacher view leaked text or audio privilege')
    if student.videos != category.videos or student.videos != audio.videos:
        raise AssertionError('Teacher views changed videos')
    return (student, category, audio), valid


class UpdateAudit(TrainerCallback):
    """Observe actual FP32 optimizer updates, with small representative samples."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.before = {}

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step >= 4 and (state.global_step + 1) % 100:
            return
        self.before = {}
        chosen = set()
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            group = ('lora_B' if 'lora_B' in name else 'lora_A' if 'lora_A' in name else
                     'projector' if 'text_hidden_fcs' in name else
                     'mask_decoder' if '.mask_decoder.' in name else None)
            if group is None or group in chosen:
                continue
            if group == 'mask_decoder' and 'transformer.layers.0.self_attn.q_proj.weight' not in name:
                continue
            chosen.add(group)
            ids = torch.linspace(0, param.numel() - 1, min(2048, param.numel()),
                                 device=param.device).long()
            self.before[name] = (param, ids, param.detach().flatten()[ids].clone(),
                                 float(param.grad.detach().float().norm()), group)

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.before:
            return
        result = {}
        for name, (param, ids, previous, grad_norm, group) in self.before.items():
            current = param.detach().flatten()[ids]
            delta = (current - previous).float()
            result[group] = {'name': name, 'dtype': str(param.dtype), 'grad_norm_post_clip': grad_norm,
                             'sampled_elements': len(ids), 'changed_elements': int(delta.ne(0).sum()),
                             'mean_abs_update': float(delta.abs().mean()),
                             'sampled_weight_digest': tensor_digest(current)}
        self.trainer.write_audit({'kind': 'optimizer_update', 'step': state.global_step + 1,
                                  'updates': result})
        self.before = {}


class FrozenTeacherAudit(TrainerCallback):
    """Verify the optimizer excludes the teacher and its full state stays fixed."""
    def __init__(self, trainer):
        self.trainer = trainer
        teacher = trainer.accelerator.unwrap_model(trainer.teacher_model)
        trainable = [name for name, param in teacher.named_parameters() if param.requires_grad]
        if trainable:
            raise ValueError(f'Frozen teacher has trainable parameters: {trainable[:8]}')
        self.samples = []
        for name, param in teacher.named_parameters():
            if not param.numel():
                continue
            ids = torch.linspace(0, param.numel() - 1, min(128, param.numel()),
                                 device=param.device).long()
            self.samples.append((name, param, ids, param.detach().flatten()[ids].clone()))
            if len(self.samples) == 4:
                break
        if len(self.samples) != 4:
            raise ValueError('Could not select teacher parameters for freeze audit')
        self.teacher_state_digest = None

    def on_train_begin(self, args, state, control, **kwargs):
        teacher = self.trainer.accelerator.unwrap_model(self.trainer.teacher_model)
        teacher_ids = {id(param) for param in teacher.parameters()}
        optimizer_ids = {id(param) for group in self.trainer.optimizer.param_groups
                         for param in group['params']}
        overlap = teacher_ids & optimizer_ids
        if overlap:
            raise RuntimeError(f'{len(overlap)} frozen teacher parameters appear in optimizer groups')
        self.teacher_state_digest = model_state_digest(teacher)
        self.trainer.write_audit({'kind': 'teacher_state_digest', 'phase': 'train_begin',
                                  'sha256': self.teacher_state_digest,
                                  'parameters': sum(1 for _ in teacher.parameters()),
                                  'buffers': sum(1 for _ in teacher.buffers()),
                                  'optimizer_parameter_overlap': 0})

    def on_optimizer_step(self, args, state, control, **kwargs):
        teacher = self.trainer.accelerator.unwrap_model(self.trainer.teacher_model)
        if any(param.grad is not None for param in teacher.parameters()):
            raise RuntimeError('Frozen teacher received a gradient during training')
        changes = {}
        for name, param, ids, before in self.samples:
            after = param.detach().flatten()[ids]
            changed = int(after.ne(before).sum())
            changes[name] = changed
            if changed:
                raise RuntimeError(f'Frozen teacher parameter changed after optimizer step: {name}')
        if state.global_step < 4 or (state.global_step + 1) % 100 == 0:
            self.trainer.write_audit({'kind': 'teacher_freeze_check', 'step': state.global_step + 1,
                                      'changed_sampled_elements': changes,
                                      'teacher_digest': [tensor_digest(value) for _, _, _, value in self.samples]})

    def on_train_end(self, args, state, control, **kwargs):
        teacher = self.trainer.accelerator.unwrap_model(self.trainer.teacher_model)
        final_digest = model_state_digest(teacher)
        if final_digest != self.teacher_state_digest:
            raise RuntimeError('Frozen teacher full parameter/buffer digest changed during training')
        self.trainer.write_audit({'kind': 'teacher_state_digest', 'phase': 'train_end',
                                  'sha256': final_digest, 'matches_train_begin': True})


class AuroraDualOPSDTrainer(AuroraGKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self._is_self_distillation or self.use_teacher_api or self._teacher_use_disable_adapter
                or self.teacher_model is None):
            raise ValueError('Dual OPSD requires an independent frozen SFT teacher via teacher_model and teacher_adapters')
        if self.lmbda != 1 or self.beta != 0 or self.gkd_logits_topk is not None or self.args.sft_alpha != 0:
            raise ValueError('Dual OPSD requires lmbda=1, beta=0, full vocabulary and sft_alpha=0')
        if float(os.environ.get('AURORA_OPSD_SEG_DISTILL_WEIGHT', '0')) != 0:
            raise ValueError('Feature distillation is not part of this dual-gate experiment')
        self.dual_config = DualGateConfig(**{
            field: (os.environ.get('AURORA_DUAL_KL_DIRECTION', default)
                    if field == 'direction' else
                    int(os.environ.get('AURORA_DUAL_CHUNK_SIZE', default))
                    if field == 'chunk_size' else
                    float(os.environ.get('AURORA_DUAL_' + field.upper(), default)))
            for field, default in asdict(DualGateConfig()).items()
        })
        if self.temperature != self.dual_config.temperature:
            raise ValueError('Generation and KD temperature differ; set both to the same value')
        teacher = self.accelerator.unwrap_model(self.teacher_model)
        if not getattr(teacher, 'peft_config', None):
            raise ValueError('Dual OPSD requires the frozen SFT teacher adapter')
        for param in teacher.parameters():
            param.requires_grad_(False)
            param.grad = None
        teacher.eval()
        # FP32 storage is required for small AdamW steps; autocast handles BF16 compute.
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        self.add_callback(UpdateAudit(self))
        self.add_callback(FrozenTeacherAudit(self))
        self.write_audit({'kind': 'recipe', 'config': asdict(self.dual_config),
                          'current_student_teacher': False, 'frozen_sft_teacher': True,
                          'teacher_base': os.environ.get('TEACHER_MODEL'),
                          'teacher_adapters': [os.environ.get('TEACHER_ADAPTERS')],
                          'teacher_trainable_parameters': sum(p.numel() for p in teacher.parameters()
                                                              if p.requires_grad),
                          'dropout': 0,
                          'trainable_dtypes': sorted({str(p.dtype) for p in self.model.parameters() if p.requires_grad}),
                          'loss': f'base full-vocab {self.dual_config.direction} KL + '
                                  'positive-gain-weighted KL + GT mask BCE/Dice'})

    def write_audit(self, data):
        folder = Path(self.args.output_dir) / 'dual_audit'
        folder.mkdir(parents=True, exist_ok=True)
        data = {**data, 'rank': self.accelerator.process_index}
        with (folder / f'rank{self.accelerator.process_index}.jsonl').open(
                'a', encoding='utf-8') as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _engine_infer(self, *args, **kwargs):
        # HF generate bypasses Accelerate's training-forward autocast. Keep
        # FP32 embedding/master parameters while computing rollout in BF16.
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            return super()._engine_infer(*args, **kwargs)

    def _prepare_batch_inputs(self, samples):
        batches = []
        for chunk in self.split_by_mini_batches(samples):
            if len(chunk) != 1:
                raise ValueError('Dual REF-AVS requires per-device microbatch=1')
            original = chunk[0]
            views, class_valid = make_views(original)
            encoded = []
            with self._template_context(self.template):
                for view in views:
                    row = encode_sample(view, self.template)
                    row.pop('_extra_kwargs', None)
                    encoded.append(to_device(self.template.data_collator([row]), self.accelerator.device))
            student, category, audio = encoded
            for teacher in (category, audio):
                assert_response_alignment(student['labels'], teacher['labels'])
            if not torch.equal(student['input_features'], category['input_features']):
                raise AssertionError('Class teacher received different audio features')
            if not torch.equal(student['input_ids'], audio['input_ids']):
                raise AssertionError('Clean-audio view changed text/video placeholder positions')
            audio_valid = not torch.equal(student['input_features'], audio['input_features'])
            evidence = {'sample_id': original.extra.get('sample_id'), 'class_name': original.extra.get('class_name'),
                        'student_audio': list(original.audios), 'class_audio': list(original.audios),
                        'audio_teacher_audio': list(original.extra['dual_clean_audio_paths']),
                        'class_valid': class_valid, 'audio_valid': audio_valid,
                        'response_ids_match': True,
                        'student_feature_sha256': tensor_digest(student['input_features']),
                        'class_feature_sha256': tensor_digest(category['input_features']),
                        'audio_feature_sha256': tensor_digest(audio['input_features']),
                        'student_user': next(m['content'] for m in reversed(views[0].messages) if m['role'] == 'user'),
                        'class_user': next(m['content'] for m in reversed(views[1].messages) if m['role'] == 'user'),
                        'audio_user': next(m['content'] for m in reversed(views[2].messages) if m['role'] == 'user')}
            # Encoding the copies must never replace original audio paths with waveforms.
            if not all(isinstance(path, str) for path in original.audios):
                raise AssertionError('Encoding mutated rollout audio paths')
            batches.append({'model_inputs': student, 'dual_teacher_inputs': [category, audio],
                            'dual_valid': [class_valid, audio_valid], 'dual_evidence': evidence,
                            'gkd_batch': GKDBatch(data_source=self._data_source)})
        return batches

    def _compute_aurora_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_inputs = dict(inputs['model_inputs'])
        teacher_inputs = [dict(x) for x in inputs['dual_teacher_inputs']]
        seg_labels = student_inputs['labels']
        for teacher in teacher_inputs:
            assert_response_alignment(seg_labels, teacher['labels'])
        if self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1):
            for row in [student_inputs, *teacher_inputs]:
                self.prepare_logits_to_keep(row)
        raw_teacher = self.accelerator.unwrap_model(self.teacher_model)
        teacher_scores = []
        teacher_token_ids = []
        # Score both privileged views with the same immutable SFT checkpoint.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False), \
                temporary_eval(raw_teacher), disable_gradient_checkpointing(
                    self.teacher_model, self.args.gradient_checkpointing_kwargs), \
                self.load_teacher_model_context():
            for row in teacher_inputs:
                forward = {k: v for k, v in row.items() if k != 'labels'}
                output = self.teacher_model(**forward, aurora_skip_hidden_states=True)
                scores, tokens = active_response(output.logits, row['labels'])
                teacher_scores.append(scores.detach())
                teacher_token_ids.append(tokens)
                del output
        outputs = model(**{k: v for k, v in student_inputs.items() if k != 'labels'}, output_hidden_states=True)
        scores, tokens = active_response(outputs.logits, student_inputs['labels'])
        for teacher_tokens in teacher_token_ids:
            if not torch.equal(tokens, teacher_tokens):
                raise ValueError('Shifted target token alignment failed')
        valid = torch.tensor(inputs['dual_valid'], device=scores.device)[:, None].expand(2, len(tokens))
        kd, metrics, details = dual_distillation_loss(
            scores, teacher_scores, tokens, valid, self.dual_config)
        mask, mask_parts = self._mask_supervision(model, outputs, student_inputs, seg_labels)
        loss = kd + float(os.environ.get('AURORA_OPSD_MASK_WEIGHT', '1')) * mask
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite dual OPSD total loss')
        self._aurora_mask_parts = {**mask_parts, **{f'dual/{k}': v for k, v in metrics.items()}}
        # Reduce explicit dual scalars over ranks; parent mask logging retains its local convention.
        self._aurora_mask_parts.update({f'dual/{k}': self.accelerator.reduce(v.detach(), reduction='mean')
                                       for k, v in metrics.items()})
        self.write_audit({'kind': 'loss', 'step': self.state.global_step,
                          'training': self.model.training, **inputs['dual_evidence'],
                          'token_ids': details['token_ids'].cpu().tolist(),
                          'delta': details['delta'].cpu().tolist(),
                          'weights': details['weights'].cpu().tolist(),
                          'route': details['route'].cpu().tolist(),
                          'metrics': {k: float(v) for k, v in metrics.items()},
                          'mask_loss': float(mask.detach()), 'total_loss': float(loss.detach()),
                          'teacher_targets_require_grad': any(x.requires_grad for x in teacher_scores)})
        self._aurora_seg_state = None
        return (loss, outputs) if return_outputs else loss


TrainerFactory.TRAINER_MAPPING['gkd'] = 'swift_plugin.aurora_dual_opsd.AuroraDualOPSDTrainer'
