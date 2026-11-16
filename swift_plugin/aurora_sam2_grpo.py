"""Two-reward GRPO for the trained AURORA SAM2 SFT baseline.

Rewards:

* ``aurora_class_exact``: 1 only when the last ``The target is ...`` category
  prediction matches the REF-AVS ground-truth category.
* ``aurora_sam2_mask_iou``: mean IoU over the ten masks decoded from the
  completion's first ``[SEG]`` hidden state by the trained SAM2 branch.

There is deliberately no format reward and no supervised BCE/Dice loss. The
projector and SAM2 branch are frozen reward decoders; GRPO updates the language
policy through sampled-token log probabilities.
"""

from __future__ import annotations

import functools
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from swift_plugin.sam2_grpo_rewards import class_exact_score
from swift_plugin.segmentation import SEG_TOKEN, load_gt_masks
from swift_plugin.sam2_segmentation import preprocess_sam2_frames

try:
    from swift.rewards import orms
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    from swift.trainers.trainer_factory import TrainerFactory
except ImportError as exc:  # pragma: no cover - exercised in the swift46 runtime
    raise ImportError("aurora_sam2_grpo requires ms-swift >= 4.x") from exc


_REWARD_CONTEXT: Dict[str, Any] = {"trainer": None}
_FAIL_OPEN = os.environ.get("AURORA_GRPO_REWARD_FAIL_OPEN", "0") == "1"


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    for _ in range(8):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        break
    return current


def _patch_rollout_request_videos_type() -> None:
    """Allow REF-AVS' ``videos=[[frame0, ... frame9]]`` representation."""
    from swift.infer_engine.protocol import InferRequest, RolloutInferRequest
    from typing import Any as TypingAny, List as TypingList

    for cls in (RolloutInferRequest, InferRequest):
        annotations = cls.__annotations__
        if "videos" in annotations:
            annotations["videos"] = TypingList[TypingAny]


def _patch_omni_rope_deltas() -> None:
    """Discard rollout-sized Qwen2.5-Omni RoPE caches before reward forwards."""
    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniThinkerForConditionalGeneration as Thinker,
        )
    except ImportError:
        return
    if getattr(Thinker, "_aurora_sam2_grpo_rope_patched", False):
        return
    Thinker._aurora_sam2_grpo_rope_patched = True
    original = Thinker.compute_3d_position_ids

    @functools.wraps(original)
    def safe_compute(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")
        batch_size = None
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        delta = getattr(self, "rope_deltas", None)
        if delta is not None and batch_size is not None and delta.shape[0] > 0:
            if batch_size < delta.shape[0] or batch_size % delta.shape[0] != 0:
                self.rope_deltas = None
        result = original(self, *args, **kwargs)
        if (
            result is not None
            and result.dim() == 3
            and result.shape[1] == 0
            and batch_size
        ):
            self.rope_deltas = None
            result = original(self, *args, **kwargs)
        return result

    Thinker.compute_3d_position_ids = safe_compute


def _as_batch(value: Optional[Sequence[Any]], count: int) -> List[Any]:
    if value is None:
        return [None] * count
    values = list(value)
    if len(values) != count:
        raise ValueError(f"Reward metadata batch has {len(values)} rows, expected {count}")
    return values


class AuroraClassExactORM:
    def __init__(self, args=None, **kwargs):
        self.args = args

    def __call__(self, completions, class_name=None, sample_id=None, **kwargs) -> List[float]:
        count = len(completions)
        names = _as_batch(class_name, count)
        sample_ids = _as_batch(sample_id, count)
        return [
            class_exact_score(completion, name, sid)
            for completion, name, sid in zip(completions, names, sample_ids)
        ]


def _completion_seg_hidden(
    trainer,
    messages: List[dict],
    frame_paths: Sequence[str],
    audios: Optional[Sequence[str]],
) -> Optional[torch.Tensor]:
    """Replay one sampled response and return its first ``[SEG]`` hidden state."""
    template = trainer.template
    model = trainer.model
    device = trainer.accelerator.device
    seg_id = template.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if seg_id is None or seg_id == template.tokenizer.unk_token_id:
        raise ValueError("[SEG] token is missing; pass --new_special_tokens '[SEG]'")

    row = {
        "messages": messages,
        "videos": [list(frame_paths)],
        "audios": list(audios) if audios else None,
    }
    row = {key: value for key, value in row.items() if value is not None}
    original_mode = template.mode
    template.set_mode("train")
    try:
        encoded = template.encode(row)
        encoded.pop("_extra_kwargs", None)
        batch = template.data_collator([encoded])
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        labels = batch["labels"]
        seg_mask = labels.eq(seg_id)
        if not bool(seg_mask.any()):
            return None
        excluded = {
            "gt_masks",
            "sam_pixel_values",
            "sam_resize_sizes",
            "sam_original_sizes",
            "text_position_ids",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        }
        forward_inputs = {key: value for key, value in batch.items() if key not in excluded}
        from swift.trainers import disable_gradient_checkpointing

        with torch.no_grad(), disable_gradient_checkpointing(
            model, trainer.args.gradient_checkpointing_kwargs
        ):
            outputs = model(**forward_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        first_index = seg_mask.int().argmax(dim=-1)
        row_index = torch.nonzero(seg_mask.any(dim=-1), as_tuple=False)[0, 0]
        return hidden[row_index, first_index[row_index]].detach()
    finally:
        template.set_mode(original_mode)


def _mean_mask_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction > 0
    truth = target > 0.5
    intersection = (pred & truth).sum(dim=(-1, -2), dtype=torch.float32)
    union = (pred | truth).sum(dim=(-1, -2), dtype=torch.float32)
    # Match the repository's evaluation convention: empty/empty contributes
    # zero (intersection / (union + eps)), not an artificial perfect score.
    per_frame = intersection / (union + 1e-7)
    return float(per_frame.mean().item())


class AuroraSAM2MaskIoUORM:
    def __init__(self, args=None, **kwargs):
        self.args = args

    def __call__(
        self,
        completions,
        messages=None,
        sam_frame_paths=None,
        mask_paths=None,
        audios=None,
        **kwargs,
    ) -> List[float]:
        trainer = _REWARD_CONTEXT.get("trainer")
        if trainer is None:
            raise RuntimeError("SAM2 mask reward requires AuroraSAM2GRPOTrainer")
        count = len(completions)
        message_rows = _as_batch(messages, count)
        frame_rows = _as_batch(sam_frame_paths, count)
        mask_rows = _as_batch(mask_paths, count)
        audio_rows = _as_batch(audios, count)
        num_frames = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
        base = _unwrap(trainer.model)
        if getattr(base, "sam2_head", None) is None:
            raise RuntimeError("SAM2 mask reward loaded a model without sam2_head")

        rewards = [0.0] * count
        hidden_states: Dict[int, torch.Tensor] = {}
        groups: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[int]] = defaultdict(list)
        was_training = trainer.model.training
        trainer.model.eval()
        try:
            with torch.no_grad():
                for index, (completion, msgs, frames, masks, audio) in enumerate(
                    zip(completions, message_rows, frame_rows, mask_rows, audio_rows)
                ):
                    if SEG_TOKEN not in str(completion):
                        continue
                    if msgs is None or not frames or not masks:
                        raise ValueError(f"Mask reward row {index} lacks messages, frames, or masks")
                    hidden = _completion_seg_hidden(trainer, msgs, frames, audio)
                    if hidden is None:
                        continue
                    hidden_states[index] = hidden
                    key = (tuple(map(str, frames)), tuple(map(str, masks)))
                    groups[key].append(index)

                device = trainer.accelerator.device
                projector_dtype = next(base.text_hidden_fcs.parameters()).dtype
                for (frames, masks), indices in groups.items():
                    images, _, original_sizes = preprocess_sam2_frames(
                        frames, num_frames, image_size=base.sam2_head.image_size
                    )
                    target = load_gt_masks(masks, original_sizes[0], num_frames).to(
                        device=device, dtype=torch.float32
                    )
                    encoded_frames = base.sam2_head.encode_frames(images.to(device))
                    for index in indices:
                        prompt = base.text_hidden_fcs(
                            hidden_states[index].to(device=device, dtype=projector_dtype).unsqueeze(0)
                        )
                        logits = base.sam2_head.decode_encoded_frames(
                            encoded_frames, prompt, tuple(target.shape[-2:])
                        )
                        rewards[index] = _mean_mask_iou(logits, target)
                    del encoded_frames, images, target
        except Exception as exc:
            if not _FAIL_OPEN:
                raise
            print(f"[aurora_sam2_grpo] mask reward failed, scoring batch as zero: {exc!r}", flush=True)
            rewards = [0.0] * count
        finally:
            if was_training:
                trainer.model.train()
        return rewards


class AuroraSAM2GRPOTrainer(GRPOTrainer):
    """Plain GRPO with frozen SAM2/projector reward decoding."""

    @staticmethod
    def _freeze_reward_decoder(model):
        base = _unwrap(model)
        sam2_head = getattr(base, "sam2_head", None)
        projector = getattr(base, "text_hidden_fcs", None)
        if sam2_head is None or projector is None:
            raise RuntimeError("AuroraSAM2GRPOTrainer requires the trained SAM2 AURORA model")
        sam2_head.requires_grad_(False)
        projector.requires_grad_(False)
        if any(parameter.requires_grad for parameter in sam2_head.parameters()):
            raise RuntimeError("SAM2 reward decoder must remain frozen during GRPO")
        if any(parameter.requires_grad for parameter in projector.parameters()):
            raise RuntimeError("SEG projector must remain frozen during GRPO")

    def __init__(self, model=None, *args, **kwargs):
        # Freeze before GRPOTrainer builds rollout parameter groups, then
        # reassert after wrapping/preparation so neither optimizer nor weight
        # synchronization treats the reward decoder as policy parameters.
        self._freeze_reward_decoder(model)
        super().__init__(model=model, *args, **kwargs)
        self._freeze_reward_decoder(self.model)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("GRPO policy has no trainable parameters")
        _REWARD_CONTEXT["trainer"] = self


orms["aurora_class_exact"] = AuroraClassExactORM
orms["aurora_sam2_mask_iou"] = AuroraSAM2MaskIoUORM

_patch_rollout_request_videos_type()
_patch_omni_rope_deltas()

original_trainer = TrainerFactory.TRAINER_MAPPING.get("grpo")
if original_trainer and "AuroraSAM2GRPOTrainer" not in original_trainer:
    TrainerFactory.TRAINER_MAPPING["grpo"] = (
        "swift_plugin.aurora_sam2_grpo.AuroraSAM2GRPOTrainer"
    )
