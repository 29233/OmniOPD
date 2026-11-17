"""AURORA GRPO plugin: original paper's reward design on ms-swift 4.x GRPO.

Reimplements model/LISA_grpo.py's reward set as swift ORM reward functions:

  format  (0/1)   completion ends with the fixed phrase "It is [SEG]."
  class   (0..2)   second-to-last sentence matches "the target is <word>";
                   +1 extra when the word equals the GT class name
  iou     (0..1)   [SEG] hidden -> SAM -> predicted mask vs GT mask IoU
                   (averaged over frames), requiring a model forward per
                   completion - the trainer injects itself into a module-level
                   reward context since ORM instances only receive ``args``.

Total reward = format + iou + class (equal weights, no scaling) exactly as the
original. The GRPO policy loss itself is swift's native trainer; the original
also adds a supervised mask loss (2*BCE + 0.5*Dice) on the rollout's [SEG],
which we attach in AuroraGRPOTrainer.compute_loss reusing the DDP-symmetric
mask branch from aurora_opsd (imported for its patches and helpers).

Original hyperparameters (model/LISA_grpo.py): num_generations=3,
temperature=1.0, top_p=0.9, max_new_tokens=512, beta=0.04,
mask loss weights bce=2.0 / dice=0.5.
"""
from __future__ import annotations

import functools
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

# Reuse the OPSD plugin's infrastructure patches (they are import-time side
# effects we also need on the GRPO path):
#  - RolloutInferRequest.videos annotation -> List[Any] (nested frame lists)
#  - OnPolicySample.to_template_dict forwards sam_frame_paths/mask_paths
#  - the DDP-symmetric mask-loss helpers (_mask_loss_from_seg_hidden etc.)
import swift_plugin.aurora_opsd as _opsd  # noqa: F401
from swift_plugin.aurora_opsd import (
    _mask_loss_from_seg_hidden,
    _unwrap,
)

try:
    from swift.rewards import orms
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    from swift.trainers.trainer_factory import TrainerFactory
except ImportError as exc:  # pragma: no cover
    raise ImportError("aurora_grpo requires ms-swift >= 4.x") from exc

try:
    from .segmentation import SEG_TOKEN, dice_loss, load_gt_masks, preprocess_sam_frames
except ImportError:
    from swift_plugin.segmentation import SEG_TOKEN, dice_loss, load_gt_masks, preprocess_sam_frames

_FORMAT_TAIL = "It is [SEG]."
_CLASS_PATTERN = re.compile(r"the target is (?:the )?(\w+)", re.IGNORECASE)

_MASK_LOSS_ENABLED = os.environ.get("AURORA_GRPO_MASK_LOSS", "1") == "1"
_MASK_WEIGHT = float(os.environ.get("AURORA_GRPO_MASK_WEIGHT", "1.0"))


def _patch_omni_rope_deltas() -> None:
    """Clear stale thinker.rope_deltas before embeds-only forwards.

    Qwen2.5-Omni's thinker caches ``rope_deltas`` on the instance after
    generation. A later forward that arrives without a matching batch size
    (ms-swift binds model.forward to the thinker via use_submodel_func, so
    every forward funnels here) reuses the cached delta in
    compute_3d_position_ids::

        delta = self.rope_deltas.repeat_interleave(bs // self.rope_deltas.shape[0])

    With bs=1 and a rollout batch of 6 the integer division is 0, so ``delta``
    becomes an empty (0, 1) tensor whose broadcast squeezes position_ids'
    batch dim to zero -> (3, 0, seq) -> "mat1 x 0" crashes in o_proj. The
    reuse branch is entered whenever ``input_ids is None`` OR
    ``past_key_values_length > 0`` (decode with KV cache), so guarding on
    input_ids alone is insufficient.

    Reset the cache whenever the cached delta's batch size is not a divisor
    of the incoming batch size, and re-run once if a stale delta still
    squeezed the result's batch dim. After the reset the thinker falls back
    to plain arange positions, matching what the SFT/OPSD forward sees.
    """
    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniThinkerForConditionalGeneration as _Thinker,
        )
    except ImportError:
        return

    if getattr(_Thinker, "_aurora_rope_deltas_patched", False):
        return
    _Thinker._aurora_rope_deltas_patched = True

    _origin_compute = _Thinker.compute_3d_position_ids

    @functools.wraps(_origin_compute)
    def _safe_compute_3d_position_ids(self, *args, **kwargs):
        def _incoming_batch_size():
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None:
                return input_ids.shape[0]
            embeds = kwargs.get("inputs_embeds")
            if embeds is not None:
                return embeds.shape[0]
            return None

        batch_size = _incoming_batch_size()
        delta = getattr(self, "rope_deltas", None)
        if delta is not None and batch_size is not None and delta.shape[0] > 0:
            if batch_size < delta.shape[0] or batch_size % delta.shape[0] != 0:
                self.rope_deltas = None
        result = _origin_compute(self, *args, **kwargs)
        # Belt and braces: a stale delta that slipped through the guard above
        # squeezes the batch dim to zero. Drop the cache and recompute; with
        # rope_deltas=None the thinker returns None and the text model falls
        # back to plain arange positions.
        if (result is not None and result.dim() == 3 and result.shape[1] == 0
                and batch_size is not None and batch_size > 0):
            self.rope_deltas = None
            result = _origin_compute(self, *args, **kwargs)
        return result

    _Thinker.compute_3d_position_ids = _safe_compute_3d_position_ids

# ORM instances are constructed with only ``args``; the IoU reward needs the
# live model/template, so the trainer registers itself here.
_REWARD_CONTEXT: Dict[str, Any] = {"trainer": None}


# ---------------------------------------------------------------------------
# Reward functions (mirroring model/LISA_grpo.py)
# ---------------------------------------------------------------------------
def _format_score(answer: str) -> float:
    answer = answer.replace("<|im_end|>", "")
    return 1.0 if answer.rstrip().endswith(_FORMAT_TAIL) else 0.0


def _class_score(answer: str, class_name: Optional[str]) -> float:
    reward = 0.0
    sentences = [s.strip() for s in re.split(r"[,.!?]+", answer) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    second_last = sentences[-2]
    match = _CLASS_PATTERN.match(second_last)
    if match:
        reward += 1.0
        if class_name and match.group(1).strip().lower() == class_name.strip().lower():
            reward += 1.0
    return reward


class AuroraFormatORM:
    def __init__(self, args=None, **kwargs):
        self.args = args

    def __call__(self, completions, **kwargs) -> List[float]:
        return [_format_score(c) for c in completions]


class AuroraClassORM:
    def __init__(self, args=None, **kwargs):
        self.args = args

    def __call__(self, completions, class_name=None, **kwargs) -> List[float]:
        # class_name arrives batched (one entry per completion) when present.
        names = list(class_name) if class_name is not None else [None] * len(completions)
        return [_class_score(c, n) for c, n in zip(completions, names)]


def _seg_hidden_for_completion(trainer, messages, sam_frame_paths, mask_paths, audios,
                               num_frames: int) -> Optional[List[float]]:
    """Local no-grad LLM forward: run the full conversation (prompt + sampled
    completion) and return the [SEG]-position hidden state as a plain list.

    Only the hidden state crosses the network — the SAM stack (projector,
    encoder, decoder, GT masks) lives on the remote IoU service
    (scripts/grpo/sam_iou_server.py). Returns None when the completion carries no
    [SEG] (nothing to score; the format reward already penalizes that case).
    """
    template = trainer.template
    model = trainer.model
    device = trainer.accelerator.device
    seg_id = trainer.template.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if seg_id is None or seg_id == trainer.template.tokenizer.unk_token_id:
        raise ValueError("[SEG] token missing; pass --new_special_tokens '[SEG]'")

    row = {
        "messages": messages,
        # videos = one "video" per <video> tag; the qwen-omni template accepts
        # a frame-path LIST as a single video entry.
        "videos": [list(sam_frame_paths)],
        # The referring prompt carries an <audio> tag; the audio feature must
        # be present or the audio placeholder scatters an EMPTY tensor and the
        # thinker receives zero-width embeddings (mat1 x0 crash).
        "audios": list(audios) if audios else None,
    }
    row = {k: v for k, v in row.items() if v is not None}
    original_mode = template.mode
    template.set_mode("train")
    try:
        encoded = template.encode(row)
        encoded.pop("_extra_kwargs", None)
        batch = template.data_collator([encoded])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        # Forward with the collator's full output minus supervision-only keys
        # and minus PADDING-FREE keys (cu_seq_lens_q/k, max_length_q/k).
        # NOTE: the historical "mat1 x0" crash was NOT caused by these keys -
        # it was thinker.rope_deltas cached by the rollout generation (batch 6)
        # being replayed on this batch-1 forward
        # (see _patch_omni_rope_deltas above). They are dropped anyway because
        # they are inert under sdpa.
        EXCLUDE = ("labels", "gt_masks", "sam_resize_sizes", "sam_original_sizes",
                   "text_position_ids", "cu_seq_lens_q", "cu_seq_lens_k",
                   "max_length_q", "max_length_k")
        fwd = {k: v for k, v in batch.items() if k not in EXCLUDE}
        from swift.trainers import disable_gradient_checkpointing
        # eval() during the reward forward: training-mode attention masks /
        # dropout paths on the omni thinker produce zero-width attention
        # outputs under the trainer's autocast context (observed as
        # "mat1 Nx0" in o_proj). The rollout itself (generation) already runs
        # in eval-like conditions; the reward forward must match.
        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad(), disable_gradient_checkpointing(
                    model, trainer.args.gradient_checkpointing_kwargs):
                outputs = model(**fwd, output_hidden_states=True)
        finally:
            if was_training:
                model.train()
        labels = batch["labels"]
        hidden = outputs.hidden_states[-1]
        seg_mask = labels.eq(seg_id)
        if not bool(seg_mask.any()):
            return None
        first_idx = seg_mask.int().argmax(dim=-1)
        rows = torch.nonzero(seg_mask.any(dim=-1), as_tuple=False).squeeze(-1)
        seg_hidden = hidden[rows, first_idx[rows]]
        return seg_hidden[0].float().cpu().tolist()
    finally:
        template.set_mode(original_mode)


class _SamIouClient:
    """Minimal HTTP client for scripts/grpo/sam_iou_server.py (no extra deps)."""

    def __init__(self, urls: List[str], timeout: float = 60.0):
        self.urls = [u.rstrip("/") for u in urls]
        self.timeout = timeout

    def iou(self, hidden, sam_frame_paths, mask_paths, num_frames) -> Optional[float]:
        import json
        import urllib.request
        payload = json.dumps({
            "hidden": hidden,
            "sam_frame_paths": list(sam_frame_paths),
            "mask_paths": list(mask_paths),
            "num_frames": int(num_frames),
        }).encode()
        last_error = None
        for url in self.urls:  # try each replica before giving up
            try:
                req = urllib.request.Request(
                    url + "/iou", data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                return float(body["iou"])
            except Exception as exc:  # noqa: BLE001 - try next replica
                last_error = exc
        raise RuntimeError(f"all SAM IoU replicas failed: {last_error!r}")


def _sam_iou_client() -> Optional[_SamIouClient]:
    """Client built from AURORA_SAM_IOU_URLS (comma-separated).

    When unset, the ORM keeps the legacy in-process SAM path (single-node
    debugging). When set (e.g. 'http://10.x.x.x:8877,http://10.x.x.y:8877'),
    [SEG] hidden states are sent to the remote service and the local SAM
    stack is never touched.
    """
    urls = os.environ.get("AURORA_SAM_IOU_URLS", "")
    if not urls:
        return None
    timeout = float(os.environ.get("AURORA_SAM_IOU_TIMEOUT", "60"))
    client = _SamIouClient([u for u in urls.split(",") if u], timeout)
    return client


_SAM_IOU_CLIENT: Optional[_SamIouClient] = None


def _completion_iou(trainer, messages, sam_frame_paths, mask_paths, audios, num_frames: int) -> Optional[float]:
    """[SEG] hidden -> SAM -> IoU vs GT mask, via the remote IoU service.

    The LLM forward (hidden-state extraction) always runs locally — the SAM
    prompt is the [SEG] hidden state, not the token id, so it cannot be
    computed server-side. The SAM stack itself runs on the dedicated service
    when AURORA_SAM_IOU_URLS is set, else falls back to the local in-process
    SAM (single-node debugging; see git history for the removed inline code).
    """
    seg_hidden = _seg_hidden_for_completion(
        trainer, messages, sam_frame_paths, mask_paths, audios, num_frames)
    if seg_hidden is None:
        return None

    global _SAM_IOU_CLIENT
    if _SAM_IOU_CLIENT is None:
        _SAM_IOU_CLIENT = _sam_iou_client()
    if _SAM_IOU_CLIENT is not None:
        return _SAM_IOU_CLIENT.iou(seg_hidden, sam_frame_paths, mask_paths, num_frames)

    # ---- legacy in-process fallback (no AURORA_SAM_IOU_URLS) --------------
    raise RuntimeError(
        "local in-process SAM IoU was removed from the reward path; set "
        "AURORA_SAM_IOU_URLS to the SAM service (scripts/grpo/sam_iou_server.py)")


class AuroraIoUORM:
    def __init__(self, args=None, **kwargs):
        self.args = args

    def __call__(self, completions, messages=None, sam_frame_paths=None,
                 mask_paths=None, audios=None, **kwargs) -> List[float]:
        trainer = _REWARD_CONTEXT.get("trainer")
        if trainer is None:
            raise RuntimeError("AuroraIoUORM requires the AuroraGRPOTrainer reward context")
        num_frames = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))
        # columns arrive batched: one entry per completion
        msgs = list(messages) if messages is not None else [None] * len(completions)
        frames = list(sam_frame_paths) if sam_frame_paths is not None else [None] * len(completions)
        masks = list(mask_paths) if mask_paths is not None else [None] * len(completions)
        auds = list(audios) if audios is not None else [None] * len(completions)
        rewards = []
        for comp, msg, fr, mp, au in zip(completions, msgs, frames, masks, auds):
            if msg is None or fr is None or mp is None:
                rewards.append(0.0)
                continue
            try:
                iou = _completion_iou(trainer, msg, fr, mp, au, num_frames)
                rewards.append(0.0 if iou is None else iou)
            except Exception as exc:
                # An unreachable SAM service is not a zero-IoU example.
                # Fail closed by default, so missing verifiable rewards
                # cannot silently masquerade as a successful GRPO run.
                if os.environ.get("AURORA_GRPO_REWARD_FAIL_OPEN", "0") != "1":
                    raise RuntimeError("SAM IoU reward failed; refusing an invalid GRPO update") from exc
                if os.environ.get("AURORA_GRPO_DEBUG", "0") == "1":
                    import traceback
                    traceback.print_exc()
                print(f"[aurora_iou] reward failed ({exc!r}); scoring 0.0", flush=True)
                rewards.append(0.0)
        return rewards


orms["aurora_format"] = AuroraFormatORM
orms["aurora_class"] = AuroraClassORM
orms["aurora_iou"] = AuroraIoUORM

# Apply the rope_deltas hygiene patch at import time (external_plugins imports
# this module inside the swift rlhf process, before any training forward).
_patch_omni_rope_deltas()


# ---------------------------------------------------------------------------
# Trainer: mask supervision on the GRPO policy-gradient loss
# ---------------------------------------------------------------------------
class AuroraGRPOTrainer(GRPOTrainer):
    """swift GRPO + AURORA mask supervision (2*BCE + 0.5*Dice on rollout [SEG]).

    The three text/mask rewards are handled by the ORM functions above; this
    trainer additionally (a) exposes itself to the IoU reward via the module
    context and (b) adds the supervised mask loss, mirroring the original
    ``loss = loss_r1 + mask_loss``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _REWARD_CONTEXT["trainer"] = self

    def _mask_supervision(self, model, model_inputs):
        """Same contract as aurora_opsd.AuroraGKDTrainer._mask_supervision."""
        if "sam_pixel_values" not in model_inputs:
            return None, None
        # Forward with the collator's full output minus supervision-only and
        # padding-free keys (see _completion_iou for the rationale).
        EXCLUDE = ("labels", "gt_masks", "sam_resize_sizes", "sam_original_sizes",
                   "text_position_ids", "cu_seq_lens_q", "cu_seq_lens_k",
                   "max_length_q", "max_length_k")
        fwd = {k: v for k, v in model_inputs.items() if k not in EXCLUDE}
        outputs = model(**fwd, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        base = _unwrap(model)
        # NOTE: swift's GRPO trainer pops 'labels' from model_inputs before
        # compute_loss (rlhf needs logprobs, not labels), so locate [SEG] via
        # input_ids instead - the rollout's completion carries the token.
        seg_mask = model_inputs["input_ids"].eq(
            base.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN))
        has_seg = seg_mask.any(dim=-1)
        rows = torch.nonzero(has_seg, as_tuple=False).squeeze(-1)
        if len(rows) > 0:
            first_idx = seg_mask.int().argmax(dim=-1)
            seg_hidden = hidden[rows, first_idx[rows]]
            gate = None
        else:
            seg_hidden = hidden[0, 0:1, :] * 1.0
            gate = seg_hidden.new_zeros((1, 1))
            rows = torch.zeros(1, dtype=torch.long, device=hidden.device)
        loss, parts = _mask_loss_from_seg_hidden(model, seg_hidden, rows, model_inputs, gate=gate)
        parts["seg_rows"] = torch.tensor(float(int(has_seg.sum().item())))
        return loss, parts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if not _MASK_LOSS_ENABLED:
            return loss
        if isinstance(inputs, list):
            inputs = inputs[0]
        model_inputs = inputs["model_inputs"]
        try:
            mask_loss, parts = self._mask_supervision(model, model_inputs)
        except Exception as exc:
            print(f"[aurora_grpo] mask supervision failed ({exc!r}); continuing", flush=True)
            return loss
        if mask_loss is None:
            return loss
        self._aurora_mask_parts = parts
        return loss + _MASK_WEIGHT * mask_loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        parts = getattr(self, "_aurora_mask_parts", None)
        if parts:
            for k, v in parts.items():
                try:
                    logs[f"aurora/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        return super().log(logs, *args, **kwargs)


# --- trainer registration (import-time side effect) ------------------------
if _MASK_LOSS_ENABLED or True:
    _orig = TrainerFactory.TRAINER_MAPPING.get("grpo")
    if _orig and "AuroraGRPOTrainer" not in _orig:
        TrainerFactory.TRAINER_MAPPING["grpo"] = "swift_plugin.aurora_grpo.AuroraGRPOTrainer"
