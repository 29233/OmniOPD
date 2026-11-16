"""AURORA OPSD trainer: ms-swift 4.x GKD trainer + AURORA mask supervision.

Load this file via ``--external_plugins`` when running ``swift rlhf
--rlhf_type gkd``. It swaps the GKD trainer for a subclass whose
``compute_loss`` additionally runs the AURORA SAM mask branch (BCE + Dice) on
the student's on-policy rollout, so language distillation (JSD against the
privileged teacher view) and mask grounding are trained jointly — mirroring
the Stage-1 SFT contract (CE + 2.0*BCE + 0.5*Dice) with JSD replacing CE.

Design decisions (see the load.md conversation history for the analysis):

- ``lmbda=1`` (pure on-policy) is assumed: every supervised row carries the
  student's own rollout, and the [SEG] token(s) inside that rollout drive the
  mask branch. GT-label rows (``lmbda<1``) still work — the same forward
  finds [SEG] in the GT response instead.
- Rollouts WITHOUT [SEG] contribute the JSD loss only (skip the mask term);
  rollouts with MULTIPLE [SEG] use the FIRST occurrence (the probe showed the
  model emits [SEG] once; the tolerance here just avoids crashes on rare
  degenerate generations). This is the selection-bias trade-off we agreed on:
  the text signal still covers [SEG]-less failures.
- The teacher forward never runs the mask branch: swift's GKD path excludes
  ``labels`` from teacher inputs (``sft_alpha=0`` default), and the AURORA
  wrapper bypasses supervision when ``labels is None``.
- Trainer swap: ``TrainerFactory.TRAINER_MAPPING['gkd']`` is patched at plugin
  import time. Plugins are imported during argument parsing
  (``_import_external_plugins``), strictly before ``TrainerFactory.get_trainer_cls``
  runs in the training pipeline, so the patched entry wins. Guarded by
  ``AURORA_OPSD_MASK_LOSS`` (default on when this plugin is loaded) and only
  active for swift 4.x.

Loss weighting: the JSD term keeps the trainer's own scale; the mask term is
``AURORA_MASK_WEIGHT`` (default 1.0) * (2.0*BCE + 0.5*Dice) — the same BCE/Dice
weights as Stage-1 SFT (env-overridable via ``AURORA_BCE_WEIGHT`` /
``AURORA_DICE_WEIGHT``, read by ``attach_aurora_segmentation``). Initial run
should log both components; if JSD dominates (probe measured unclipped
KL~12.7), raise ``AURORA_MASK_WEIGHT``.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch

from swift_plugin.seg_distillation import first_seg_hidden, seg_prompt_distillation, validate_response_alignment

try:
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    from swift.trainers import disable_gradient_checkpointing
    from swift.trainers.trainer_factory import TrainerFactory
except ImportError as exc:  # pragma: no cover - swift 3.x has no gkd trainer
    raise ImportError(
        "aurora_opsd requires ms-swift >= 4.x (GKD trainer with OPSD "
        "teacher_prompt support). The base env (swift 3.10.1) is not supported."
    ) from exc

# Local imports (plugin may be imported as part of the swift_plugin package or
# standalone via --external_plugins, so try both).
try:
    from .segmentation import SEG_TOKEN
except ImportError:
    from swift_plugin.segmentation import SEG_TOKEN

_MASK_LOSS_ENABLED = os.environ.get("AURORA_OPSD_MASK_LOSS", "1") == "1"
_MASK_WEIGHT = float(os.environ.get("AURORA_OPSD_MASK_WEIGHT", "1.0"))
# λ2: [SEG] embedding feature distillation. Aligns the student's [SEG] prompt
# (text_hidden_fcs output, the 256-d tensor the SAM branch consumes) with the
# TEACHER's [SEG] prompt computed under the privileged view. This aligns the
# representation consumed by SAM in addition to next-token distributions.
# 0.0 disables it. Current OPSD shares the projector, with teacher targets
# detached; independent local teachers use their own projector.
#
# Teacher and student share response_token_ids, so [SEG] presence is identical
# on both sides. The loss is zero on [SEG]-less rollouts; response and batch
# row alignment are checked explicitly. Requires the mask supervision branch (it
# guarantees text_hidden_fcs participates in the DDP graph every step, see
# _mask_loss_from_seg_hidden) and a LOCAL teacher (no --teacher_model_server).
_SEG_DISTILL_WEIGHT = float(os.environ.get("AURORA_OPSD_SEG_DISTILL_WEIGHT", "0"))
_SEG_DISTILL_LOSS = os.environ.get("AURORA_OPSD_SEG_DISTILL_LOSS", "mse")
if not torch.isfinite(torch.tensor(_SEG_DISTILL_WEIGHT)) or _SEG_DISTILL_WEIGHT < 0:
    raise ValueError('AURORA_OPSD_SEG_DISTILL_WEIGHT must be finite and non-negative')
if _SEG_DISTILL_LOSS not in {'mse', 'cosine'}:
    raise ValueError('AURORA_OPSD_SEG_DISTILL_LOSS must be mse or cosine')
if _SEG_DISTILL_WEIGHT > 0 and not _MASK_LOSS_ENABLED:
    raise ValueError('SEG distillation requires AURORA_OPSD_MASK_LOSS=1')
# Audio-degrade privilege: the student hears a noise-degraded copy (the row's
# audios already points at it — the dataset builder rewrote it), the teacher
# hears the ORIGINAL audio via the teacher_audio_paths dataset column. Empty
# string (default) disables the swap; "original" enables it. The dataset-side
# marker is the teacher_audio_paths column, whose presence must match this
# env (bidirectional guard in _patch_teacher_audio).
_TEACHER_AUDIO = os.environ.get("AURORA_OPSD_TEACHER_AUDIO", "")
_DEBUG_LOG = os.environ.get("AURORA_OPSD_DEBUG_LOG", "0") == "1"
_CURRENT_TEACHER = os.environ.get("AURORA_OPSD_TEACHER_MODE", "frozen") == "current"
# Optional OPSD privilege: append the first-frame GT mask as a teacher-only
# image.  Keep this disabled by default so existing class-privilege runs are
# bit-for-bit unchanged.
_TEACHER_MASK_IMAGE = os.environ.get("AURORA_OPSD_TEACHER_MASK_IMAGE", "0") == "1"


def _patch_frozen_teacher_adapters() -> None:
    """Bridge older Swift loaders which read reward_adapters for teachers.

    Scoped to explicitly guarded local SFT-teacher runs. New Swift versions
    already read teacher_adapters; mirroring the legacy argument is harmless
    there. Restore the reward configuration even if model loading fails.
    """
    if os.environ.get('AURORA_OPSD_REQUIRE_FROZEN_TEACHER', '0') != '1':
        return
    from swift.pipelines.train.rlhf import SwiftRLHF

    original = SwiftRLHF._prepare_single_model
    if getattr(original, '_aurora_teacher_adapters', False):
        return

    def prepare_with_teacher_adapters(self, key, origin_key, model_type, model_revision):
        if key != 'teacher':
            return original(self, key, origin_key, model_type, model_revision)
        if not self.args.teacher_adapters:
            raise ValueError('Frozen SFT teacher requires explicit teacher_adapters')
        previous = self.args.reward_adapters
        self.args.reward_adapters = self.args.teacher_adapters
        try:
            return original(self, key, origin_key, model_type, model_revision)
        finally:
            self.args.reward_adapters = previous

    prepare_with_teacher_adapters._aurora_teacher_adapters = True
    SwiftRLHF._prepare_single_model = prepare_with_teacher_adapters


_patch_frozen_teacher_adapters()


@contextmanager
def _current_teacher_eval(model):
    """Evaluate the same active student parameters for the privileged view.

    GKD's self-distillation path otherwise inherits the model's train mode;
    dropout would make the privileged target noisy and unlike the rollout.
    Restore the caller's mode after the no-grad teacher forward.
    """
    if not _CURRENT_TEACHER:
        yield
        return
    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in training_flags:
            module.training = training


def _allow_http_teacher_without_local_vllm() -> None:
    """Allow ms-swift's requests-only teacher client without importing vLLM.

    ``VLLMInferClient`` is an HTTP client, but upstream currently rejects its
    construction when the training environment does not also have the vLLM
    Python package installed. A remote/custom teacher server does not require
    any local vLLM runtime. Limit the compatibility patch to the explicit
    split-teacher launcher so colocated-vLLM configurations keep their normal
    dependency check.
    """
    if os.environ.get("AURORA_OPSD_HTTP_TEACHER", "0") != "1":
        return
    from swift.rlhf_trainers import vllm_client

    vllm_client.is_vllm_available = lambda: True


_allow_http_teacher_without_local_vllm()


def _patch_rollout_request_videos_type() -> None:
    """Relax ``RolloutInferRequest.videos`` typing to accept frame lists.

    AURORA rows represent each video as a list of frame image paths
    (``videos=[[f0..f9]]``) — the qwen-omni template deliberately supports
    this ("image list as video", template/templates/qwen.py handles
    ``isinstance(video, list)``), and the SFT path encodes it fine. The RL
    rollout path, however, serializes each sample with dacite into
    ``RolloutInferRequest``, whose ``videos: List[str]`` annotation rejects
    the nested structure before the template ever sees it.

    Widening the annotation to ``List[Any]`` lets the nested frame list flow
    through to the template unchanged. dacite reads types via
    ``get_type_hints`` (i.e. the class ``__annotations__``), so patching the
    dataclass ``Field.type`` attribute has no effect — the annotation itself
    must change. The ``videos`` field is declared on the PARENT class
    ``InferRequest`` (RolloutInferRequest only adds uuid etc.), and
    ``get_type_hints`` resolves inherited annotations from the defining
    class, so the patch must target the parent.
    """
    from swift.infer_engine.protocol import InferRequest, RolloutInferRequest

    from typing import Any, List

    for cls in (RolloutInferRequest, InferRequest):
        ann = cls.__annotations__
        if "videos" in ann and str(ann["videos"]) != "typing.List[typing.Any]":
            ann["videos"] = List[Any]


_patch_rollout_request_videos_type()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP/DeepSpeed/PEFT wrappers to reach the AURORA-annotated model."""
    current = model
    for _ in range(6):
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "get_base_model"):
            current = current.get_base_model()
            continue
        break
    return current


def _extract_seg_hidden(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], int]:
    """Gather [SEG]-position hidden states from a (possibly [SEG]-less) batch.

    Returns ``(seg_hidden, n_seg)`` where ``seg_hidden`` is ``[n_seg, D]``
    (``None`` when no sample emitted [SEG]) and ``n_seg`` counts total
    occurrences. Only the FIRST [SEG] per row is kept (rows with several
    occurrences are rare degenerate generations; crashing on them — as the
    SFT path deliberately does — would kill an on-policy run).
    """
    base = _unwrap(model)
    seg_token_id = base.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    if seg_token_id is None or seg_token_id == base.aurora_tokenizer.unk_token_id:
        raise ValueError("[SEG] is missing; pass --new_special_tokens '[SEG]'")
    seg_mask = labels.eq(seg_token_id)
    total = int(seg_mask.sum().item())
    if total == 0:
        return None, 0
    # first [SEG] per row: argmax finds the first True index
    first_idx = seg_mask.int().argmax(dim=-1)  # [B]
    has_seg = seg_mask.any(dim=-1)  # [B]
    rows = torch.nonzero(has_seg, as_tuple=False).squeeze(-1)
    positions = first_idx[rows]
    seg_hidden = hidden_states[rows, positions]  # [n_seg, D]
    return seg_hidden, total


def _mask_loss_from_seg_hidden(
    model: torch.nn.Module,
    seg_hidden: torch.Tensor,
    seg_rows: torch.Tensor,
    batch: Dict[str, Any],
    gate: Optional[torch.Tensor] = None,
    text_prompts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run text_hidden_fcs -> SAM on [SEG] hidden states; return (loss, parts).

    ``seg_rows`` maps each seg-hidden row back to its batch index so the right
    frames/GT masks are used. Frames are shared per row (AURORA contract), so
    only the rows that emitted [SEG] contribute.

    ``gate`` (shape ``[n_seg, 1]``, values 0/1) implements the DDP-symmetric
    dummy path: when a batch has NO [SEG] at all we still run the SAME module
    sequence (text_hidden_fcs -> prompt_encoder -> mask_decoder) on a zero
    vector against the first row's frames, with gate=0 so the loss
    contribution is exactly zero. This keeps the autograd graph — and
    therefore the DDP gradient-hook firing order — IDENTICAL across ranks
    regardless of their (independently sampled) rollouts. Without this,
    per_device_train_batch_size=1 multi-GPU runs deadlock: ranks with [SEG]
    fire hooks for sam modules that [SEG]-less ranks never touch, the NCCL
    collective sequences diverge and both ranks block forever.
    """
    base = _unwrap(model)
    backend = os.environ.get("AURORA_SEG_BACKEND", "sam").lower()
    if text_prompts is None:
        projector_dtype = next(base.text_hidden_fcs.parameters()).dtype
        text_prompts = base.text_hidden_fcs(seg_hidden.to(dtype=projector_dtype))
    if gate is not None:
        text_prompts = text_prompts * gate.to(dtype=text_prompts.dtype)

    if backend == "sam2":
        if getattr(base, "sam2_head", None) is None:
            raise RuntimeError("AURORA_SEG_BACKEND=sam2 but the model has no sam2_head")
        images = batch["sam_pixel_values"].to(device=seg_hidden.device)
    elif backend == "sam":
        images = batch["sam_pixel_values"].to(
            device=seg_hidden.device, dtype=next(base.sam.image_encoder.parameters()).dtype
        )
        batch_size, frame_count = images.shape[:2]
        with torch.no_grad():
            image_embeddings = base.sam.image_encoder(images.flatten(0, 1))
        decoder_dtype = next(base.sam.mask_decoder.parameters()).dtype
        image_embeddings = image_embeddings.view(
            batch_size, frame_count, *image_embeddings.shape[1:]
        ).to(decoder_dtype)
        prompt_encoder_dtype = next(base.sam.prompt_encoder.parameters()).dtype
    else:
        raise ValueError(f"Unsupported AURORA_SEG_BACKEND={backend!r}")

    # DDP symmetry: iterate over a fixed row count. With no [SEG] in the batch
    # we still process ONE dummy row (index 0) gated to zero.
    if seg_rows is not None and len(seg_rows) > 0:
        iter_rows = seg_rows.tolist()
    else:
        iter_rows = [0]

    bce = seg_hidden.new_zeros((), dtype=torch.float32)
    dice = seg_hidden.new_zeros((), dtype=torch.float32)
    for i, batch_index in enumerate(iter_rows):
        target = batch["gt_masks"][batch_index].to(device=seg_hidden.device, dtype=torch.float32)
        if backend == "sam2":
            pred_mask = base.sam2_head(
                images[batch_index], text_prompts[i], tuple(target.shape[-2:])
            )
        else:
            prompt = text_prompts[i].float().to(prompt_encoder_dtype).view(1, 1, -1)
            sparse, dense = base.sam.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=prompt)
            low_res_masks, _ = base.sam.mask_decoder(
                image_embeddings[batch_index],
                image_pe=base.sam.prompt_encoder.get_dense_pe().to(decoder_dtype),
                sparse_prompt_embeddings=sparse.to(decoder_dtype),
                dense_prompt_embeddings=dense.to(decoder_dtype),
                multimask_output=False,
            )
            pred_mask = base.sam.postprocess_masks(
                low_res_masks,
                input_size=tuple(batch["sam_resize_sizes"][batch_index][0]),
                original_size=tuple(batch["sam_original_sizes"][batch_index][0]),
            )[:, 0]
        target = target.to(device=pred_mask.device, dtype=pred_mask.dtype)
        if target.shape != pred_mask.shape:
            raise ValueError(f"GT/predicted mask shape mismatch: {target.shape} != {pred_mask.shape}")
        from swift_plugin.segmentation import dice_loss

        bce = bce + torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, target)
        dice = dice + dice_loss(pred_mask, target)

    n = len(iter_rows)
    bce = bce / n
    dice = dice / n
    # Detach metrics BEFORE the gate multiplication so logged numbers stay
    # interpretable (dummy path reports ~the ungated magnitude but its loss
    # term below is exactly zero).
    parts = {"mask_bce_loss": bce.detach(), "mask_dice_loss": dice.detach()}
    loss = base.aurora_bce_weight * bce + base.aurora_dice_weight * dice
    if gate is not None:
        loss = loss * gate.float().sum()
    parts["mask_loss"] = loss.detach()
    return loss, parts


class AuroraGKDTrainer(GKDTrainer):
    """GKD trainer with AURORA SAM mask supervision on the student rollout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('AURORA_OPSD_REQUIRE_FROZEN_TEACHER', '0') == '1':
            if (self._is_self_distillation or self.use_teacher_api
                    or self._teacher_use_disable_adapter or self.teacher_model is None):
                raise ValueError('This run requires an independent frozen SFT teacher; pass teacher_model and teacher_adapters')
            if any(parameter.requires_grad for parameter in self.teacher_model.parameters()):
                raise ValueError('Frozen teacher still contains trainable parameters')
            teacher = self.accelerator.unwrap_model(self.teacher_model)
            if not getattr(teacher, 'peft_config', None):
                raise ValueError('Frozen SFT teacher has no loaded PEFT adapter')
            print('[opsd] verified independent frozen teacher with loaded SFT adapter; trainable teacher parameters=0', flush=True)
        if _CURRENT_TEACHER and (
            not self._is_self_distillation or self.use_teacher_api or self._teacher_use_disable_adapter
        ):
            raise ValueError(
                "current OPSD must reuse the active student and adapter; omit "
                "teacher_model, teacher_adapters and teacher_model_server."
            )
        if _SEG_DISTILL_WEIGHT > 0:
            # λ2 reuses the mask branch's DDP-symmetric graph and its stashed
            # [SEG] projected prompts; without it the student projector might not
            # participate in the graph every step (deadlock risk), and the
            # teacher-side hidden states come from the local teacher forward
            # re-implemented in compute_loss.
            if not _MASK_LOSS_ENABLED:
                raise ValueError(
                    "AURORA_OPSD_SEG_DISTILL_WEIGHT > 0 requires the mask "
                    "supervision branch (AURORA_OPSD_MASK_LOSS=1, the default)."
                )
            if self.use_teacher_api:
                raise ValueError(
                    "AURORA_OPSD_SEG_DISTILL_WEIGHT > 0 requires a local teacher; "
                    "--teacher_model_server cannot provide hidden states."
                )
        if _TEACHER_AUDIO == "original" and self.use_teacher_api:
            # The audio swap lives in to_teacher_template_dict; the teacher API
            # path builds requests via to_infer_request instead, so the swap
            # would silently not apply. Fail loudly.
            raise ValueError(
                "AURORA_OPSD_TEACHER_AUDIO=original requires a local teacher; "
                "--teacher_model_server cannot receive swapped audios."
            )

    def _try_static_graph(self) -> None:
        """No-op: static graph is requested via ``--ddp_static_graph true``.

        The mask branch calls sam.mask_decoder / text_hidden_fcs directly
        (outside the DDP-wrapped forward), so those parameters fire their DDP
        gradient hooks a second time during the single backward pass.
        ``DistributedDataParallel(static_graph=True)`` makes DDP tolerate
        parameters that contribute to the loss graph outside forward(). The
        flag must be set when DDP is CONSTRUCTED (before any forward), so a
        lazy runtime call is too late — the training script passes
        ``--ddp_static_graph true`` instead.
        """

    def _mask_supervision(self, model, outputs, model_inputs, seg_labels=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the AURORA mask loss from the student forward outputs.

        Requires ``output_hidden_states=True`` on the student forward (set in
        ``compute_loss``) and the AURORA supervision tensors to be present in
        ``model_inputs`` (our template's collator puts them there for
        segmentation rows).

        DDP-symmetric contract: the module call sequence is IDENTICAL whether
        or not the batch emitted [SEG] — the no-[SEG] case runs one dummy row
        (zero hidden state, gate=0) through the same text_hidden_fcs ->
        prompt_encoder -> mask_decoder path. See _mask_loss_from_seg_hidden.
        """
        labels = model_inputs["labels"] if seg_labels is None else seg_labels
        if "sam_pixel_values" not in model_inputs:
            raise ValueError(
                "AURORA OPSD mask supervision enabled but model_inputs lacks "
                "sam_pixel_values; check that rows carry sam_frame_paths/mask_paths "
                "and the aurora template is in use."
            )
        hidden = outputs.hidden_states[-1]
        base = _unwrap(model)
        seg_hidden, rows = first_seg_hidden(hidden, labels, base.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN))
        n_seg = len(rows)

        gate = None
        if n_seg == 0:
            # Dummy row: zero vector from the FIRST sequence position, gated
            # to zero. Same modules fire, loss contribution is exactly 0.
            seg_hidden = hidden[0, 0:1, :] * 1.0  # [1, D], keeps autograd
            gate = seg_hidden.new_zeros((1, 1))
            rows = torch.zeros(1, dtype=torch.long, device=hidden.device)

        prompts = None
        if _SEG_DISTILL_WEIGHT > 0:
            dtype = next(base.text_hidden_fcs.parameters()).dtype
            prompts = base.text_hidden_fcs(seg_hidden.to(dtype=dtype)).float()
        loss, parts = _mask_loss_from_seg_hidden(
            model, seg_hidden, rows, model_inputs, gate=gate, text_prompts=prompts
        )
        parts["seg_rows"] = torch.tensor(float(n_seg))
        # Reuse the EXACT projected prompt used for mask decoding. The empty
        # row list marks the dummy path; its projector graph remains attached.
        self._aurora_seg_state = (prompts, rows[:n_seg], labels)
        return loss, parts

    def _teacher_seg_targets(self, teacher_model, outputs, labels):
        """Project inside the teacher's no_grad/eval/adapter/offload contexts."""
        base = _unwrap(teacher_model)
        if not outputs.hidden_states:
            raise RuntimeError('SEG distillation requires teacher hidden states')
        hidden, rows = first_seg_hidden(
            outputs.hidden_states[-1], labels, base.aurora_tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        )
        with torch.no_grad():
            dtype = next(base.text_hidden_fcs.parameters()).dtype
            prompts = base.text_hidden_fcs(hidden.to(dtype=dtype)).float().detach()
        # Retain only the tiny projected target, not every teacher layer.
        outputs.hidden_states = None
        return prompts, rows

    def _seg_distill_loss(self, teacher_targets, teacher_labels):
        """Align actual mask prompts; current OPSD shares the active projector.

        Frozen-teacher runs use that teacher's own frozen projector. Current
        OPSD uses the student's current projector with stop-gradient targets;
        GT mask supervision remains necessary to anchor the representation.
        """
        seg_state = getattr(self, "_aurora_seg_state", None)
        self._aurora_seg_state = None
        if seg_state is None or seg_state[0] is None or teacher_targets is None:
            raise RuntimeError("λ2 requires mask supervision to have run this step")
        s_prompt, s_rows, s_labels = seg_state
        validate_response_alignment(s_labels, teacher_labels)
        t_prompt, t_rows = teacher_targets
        loss, parts = seg_prompt_distillation(s_prompt, t_prompt, s_rows, t_rows, _SEG_DISTILL_LOSS)
        parts['seg_distill_weighted_loss'] = loss.detach() * _SEG_DISTILL_WEIGHT
        return loss, parts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute OPSD loss while keeping teacher/AdamW residency disjoint.

        With a 7B independent teacher, step 1 fits because AdamW has not
        created its moment tensors yet. From step 2 onward those states plus
        the temporarily loaded teacher exceed an 80GB GPU. ms-swift's normal
        ``offload_context`` only covers vLLM generation, not this custom local
        teacher forward, so explicitly offload optimizer state for the whole
        loss computation and restore it before Trainer performs optimizer.step.
        """
        should_offload_optimizer = (
            self.args.offload_teacher_model
            and self.args.offload_optimizer
            and getattr(self, "optimizer", None) is not None
            and bool(self.optimizer.state)
        )
        if should_offload_optimizer:
            self.offload_optimizer()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        try:
            return self._compute_aurora_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            if should_offload_optimizer:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.load_optimizer()

    def _compute_aurora_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """JSD distillation loss (parent) + AURORA mask loss (this class).

        The parent's non-liger path forwards the student WITHOUT labels, so
        the AURORA wrapper (which keys mask supervision on ``labels``) stays
        inert and the parent's JSD loss sees clean logits. We re-run the mask
        branch here against the SAME forward's hidden states — one extra SAM
        pass, no extra LLM forward.
        """
        if not _MASK_LOSS_ENABLED:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        self._try_static_graph()
        model_inputs = inputs["model_inputs"]
        gkd_batch = inputs["gkd_batch"]
        self._aurora_seg_state = None
        # LM-head trimming changes loss labels but not the last hidden-state
        # sequence. Preserve original labels for locating the actual [SEG].
        student_seg_labels = model_inputs["labels"]
        teacher_seg_labels = inputs["teacher_model_inputs"]["labels"]

        # --- replicate the parent's student forward, but ask for hidden states
        # NOTE: kept in sync with GKDTrainer.compute_loss (non-liger, non-API
        # branch). The teacher side is delegated to the parent by NOT calling
        # it here — instead we call the parent for the JSD part only after
        # re-running the student forward would double the cost. To keep a
        # single student forward we reimplement the parent's loss path below.
        use_logits_to_keep = self.get_use_logits_to_keep(self.template.sequence_parallel_size == 1)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(model_inputs)
            if not self.use_teacher_api:
                self.prepare_logits_to_keep(inputs["teacher_model_inputs"])
        teacher_labels = inputs["teacher_model_inputs"]["labels"]

        forward_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
        if self.use_liger_gkd_loss:
            raise ValueError("AURORA mask supervision is incompatible with --use_liger_kernel GKD loss")

        outputs_student = model(**forward_inputs, output_hidden_states=True)
        # The AURORA forward wrapper may or may not be active on this model
        # (it is, via attach_aurora_segmentation at load time): with
        # sam_pixel_values present but labels absent it returns early, so
        # hidden_states survive for our mask branch.

        # --- teacher output (mirrors parent logic)
        outputs_teacher = None
        teacher_targets = None
        if self.use_teacher_api:
            teacher_topk_logprobs = gkd_batch.teacher_topk_logprobs
            teacher_topk_indices = gkd_batch.teacher_topk_indices
            teacher_out = type("TeacherOutputHolder", (), {})()  # placeholder, replaced below
            from swift.rlhf_trainers.gkd_helpers import TeacherOutput

            teacher_out = TeacherOutput(
                topk_logprobs=teacher_topk_logprobs, topk_indices=teacher_topk_indices, labels=teacher_labels
            )
        else:
            t_fwd = {k: v for k, v in inputs["teacher_model_inputs"].items() if k != "labels"}
            # The teacher only needs its last hidden state when the optional
            # [SEG] feature-distillation term is enabled. Otherwise suppress
            # all hidden states before Accelerate's FP32 output conversion.
            if _SEG_DISTILL_WEIGHT <= 0:
                t_fwd["aurora_skip_hidden_states"] = True
            if self._is_self_distillation:
                adapter_ctx = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self._teacher_use_disable_adapter
                    else _null_ctx()
                )
                with torch.no_grad(), adapter_ctx, _current_teacher_eval(model), disable_gradient_checkpointing(
                    model, self.args.gradient_checkpointing_kwargs
                ):
                    # The AURORA forward wrapper always requests hidden states
                    # internally, so outputs_teacher.hidden_states is available
                    # for the λ2 branch even without an explicit flag.
                    outputs_teacher = model(**t_fwd)
                    if _SEG_DISTILL_WEIGHT > 0:
                        teacher_targets = self._teacher_seg_targets(model, outputs_teacher, teacher_seg_labels)
            else:
                load_ctx = (
                    self.load_teacher_model_context()
                    if self.args.offload_teacher_model
                    else _null_ctx()
                )
                with load_ctx, torch.no_grad(), disable_gradient_checkpointing(
                    self.teacher_model, self.args.gradient_checkpointing_kwargs
                ):
                    outputs_teacher = self.teacher_model(**t_fwd)
                    if _SEG_DISTILL_WEIGHT > 0:
                        teacher_targets = self._teacher_seg_targets(
                            self.teacher_model, outputs_teacher, teacher_seg_labels
                        )
            from swift.rlhf_trainers.gkd_helpers import TeacherOutput

            teacher_out = TeacherOutput(full_logits=outputs_teacher.logits, labels=teacher_labels)

        jsd_loss = self._compute_jsd_loss(outputs_student.logits, teacher_out, model_inputs["labels"])

        # Keep the privileged-teacher distillation term visible in the regular
        # trainer log.  ``loss`` is globally reduced by Trainer, whereas the
        # mask diagnostics below are computed on the local rank, so callers
        # must not infer this value by subtracting ``mask_loss`` from ``loss``.
        opsd_jsd_metric = jsd_loss.detach()
        try:
            opsd_jsd_metric = self.accelerator.reduce(opsd_jsd_metric, reduction="mean")
        except (AttributeError, TypeError):
            # Keep single-process/debug runs independent of Accelerator's
            # distributed reduction API.
            pass

        # --- AURORA mask loss on the same student forward
        if "sam_pixel_values" in model_inputs:
            mask_loss, parts = self._mask_supervision(model, outputs_student, model_inputs, student_seg_labels)
            loss = jsd_loss + _MASK_WEIGHT * mask_loss
            # λ2: [SEG] prompt distillation from the privileged teacher
            if _SEG_DISTILL_WEIGHT > 0:
                seg_distill, seg_parts = self._seg_distill_loss(teacher_targets, teacher_seg_labels)
                loss = loss + _SEG_DISTILL_WEIGHT * seg_distill
                parts.update(seg_parts)
            parts["opsd_jsd_loss"] = opsd_jsd_metric
            self._aurora_mask_parts = parts
        else:
            if _SEG_DISTILL_WEIGHT > 0:
                raise ValueError('SEG distillation requires sam_pixel_values and GT mask supervision')
            loss = jsd_loss
            self._aurora_mask_parts = {"opsd_jsd_loss": opsd_jsd_metric}

        if return_outputs:
            return (loss, outputs_student)
        return loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        # surface the mask-loss components in the trainer's regular logging
        # (transformers Trainer.log is the canonical entry; GKDTrainer
        # overrides it, so we hook here rather than in _log)
        parts = getattr(self, "_aurora_mask_parts", None)
        if parts:
            for k, v in parts.items():
                try:
                    logs[f"aurora/{k}"] = float(v)
                except (TypeError, ValueError):
                    pass
        return super().log(logs, *args, **kwargs)


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --- make AURORA supervision fields survive the rollout round-trip -------------
# ``OnPolicySample.to_template_dict`` forwards only StandardKeys (messages/
# images/videos/audios/tools/objects) to ``template.encode`` — our template's
# ``_encode`` reads ``sam_frame_paths``/``mask_paths`` from the row's extra
# kwargs, which would otherwise be dropped. Patch ``to_template_dict`` to
# forward the AURORA supervision columns from ``extra`` so the training
# forward receives sam_pixel_values/gt_masks.
_AURORA_ROW_KEYS = ("sam_frame_paths", "mask_paths", "class_name", "sample_id")


def _patch_sample_to_template_dict() -> None:
    from swift.rl_core.data import OnPolicySample

    if getattr(OnPolicySample.to_template_dict, "_aurora_patched", False):
        return
    original = OnPolicySample.to_template_dict

    def aurora_to_template_dict(self):
        d = original(self)
        for key in _AURORA_ROW_KEYS:
            if key not in d and key in self.extra:
                d[key] = self.extra[key]
        return d

    aurora_to_template_dict._aurora_patched = True
    OnPolicySample.to_template_dict = aurora_to_template_dict


def _patch_teacher_mask_image() -> None:
    """Guard + inject the teacher-only first-frame mask image for OPSD.

    ``OnPolicySample.to_teacher_template_dict`` starts from the standard
    fields, so adding ``images`` here affects only the teacher encoding.  The
    student path never sees this field.  The corresponding ``<image>`` tag
    comes from the offline dataset builder (``--privilege-kind
    first-mask-image``); the image injection comes from
    ``AURORA_OPSD_TEACHER_MASK_IMAGE=1`` at train time.

    The two switches live in different files and are flipped in different
    sessions, so this patch ALWAYS installs a consistency guard (even when the
    image is disabled): the raw dataset ``teacher_prompt`` column must contain
    ``<image>`` exactly when ``AURORA_OPSD_TEACHER_MASK_IMAGE=1``.  A mismatch
    is otherwise silent — a literal ``"<image>"`` string stays in the teacher
    prompt (dataset mask / env 0), or an unexplained image arrives without its
    describing sentence (dataset class / env 1) — and the only symptom is a
    degraded distillation signal several GPU-days later.
    """
    from swift.rl_core.data import OnPolicySample

    if getattr(OnPolicySample.to_teacher_template_dict, "_aurora_mask_patched", False):
        return

    original_build = OnPolicySample.build_teacher_view
    original_to_teacher = OnPolicySample.to_teacher_template_dict

    _mismatch_help = (
        "AURORA OPSD teacher-mask-image configuration mismatch: the dataset's "
        "teacher_prompt column and AURORA_OPSD_TEACHER_MASK_IMAGE disagree. "
        "Rebuild the dataset with `scripts/data/prepare_refavs_opsd.py "
        "--privilege-kind {wanted}` and/or set "
        "AURORA_OPSD_TEACHER_MASK_IMAGE={wanted_env} in "
        "scripts/opd/train_opsd_swift.sh."

    )

    def aurora_build_teacher_view(self):
        result = original_build(self)
        if result and self.teacher_messages is not None:
            for message in reversed(self.teacher_messages):
                if message.get("role") == "user":
                    content = message.get("content") or ""
                    if _TEACHER_MASK_IMAGE and "<image>" not in content:
                        # Keep video/audio first, then expose the mask as a
                        # separate visual input before the natural-language
                        # privilege block.
                        marker = "<video><audio>"
                        if marker in content:
                            content = content.replace(marker, marker + "<image>", 1)
                        else:
                            content = "<image>" + content
                        message["content"] = content
                    break
        return result

    def aurora_to_teacher_template_dict(self):
        d = original_to_teacher(self)
        # Judge from the RAW dataset teacher_prompt (not teacher_messages —
        # the build wrapper above may have inserted a tag itself).
        prompt_has_tag = isinstance(self.teacher_prompt, str) and "<image>" in self.teacher_prompt
        if prompt_has_tag != _TEACHER_MASK_IMAGE:
            raise ValueError(_mismatch_help.format(
                wanted="first-mask-image" if _TEACHER_MASK_IMAGE else "class",
                wanted_env="1" if _TEACHER_MASK_IMAGE else "0",
            ))
        if not _TEACHER_MASK_IMAGE:
            return d
        # Order guard: the <image> tag sits EARLY in the message while we
        # append to the END of the images list. That pairing is only correct
        # when the row carries no other images (AURORA rows are video/audio
        # only) — otherwise the mask would bind to the wrong tag.
        if self.images:
            raise ValueError(
                "AURORA_OPSD_TEACHER_MASK_IMAGE=1 requires teacher rows without "
                f"pre-existing images; got {len(self.images)} image(s) on sample "
                f"{self.request_id!r}."
            )
        mask_paths = self.extra.get("mask_paths") or []
        if not mask_paths:
            raise ValueError(
                "AURORA_OPSD_TEACHER_MASK_IMAGE=1 requires non-empty mask_paths "
                "on every OPSD sample"
            )
        first_mask = str(mask_paths[0])
        if not os.path.isfile(first_mask):
            raise FileNotFoundError(f"teacher first-frame mask does not exist: {first_mask}")
        d["images"] = list(d.get("images") or []) + [first_mask]
        return d

    aurora_build_teacher_view._aurora_mask_patched = True
    aurora_to_teacher_template_dict._aurora_mask_patched = True
    OnPolicySample.build_teacher_view = aurora_build_teacher_view
    OnPolicySample.to_teacher_template_dict = aurora_to_teacher_template_dict


def _patch_teacher_audio() -> None:
    """Audio-degrade privilege: teacher hears the ORIGINAL audio.

    The dataset builder (``prepare_refavs_opsd.py --privilege-kind
    audio-degrade``) rewrote the row's ``audios`` to the degraded copy — the
    student (rollout + training encode + eval) hears it everywhere via the
    standard path — and stored the original in a ``teacher_audio_paths``
    column (lands in ``OnPolicySample.extra``). This patch swaps the original
    into the TEACHER encoding only, in ``to_teacher_template_dict``.

    Unlike the mask-image privilege there is no tag insertion and no
    ``build_teacher_view`` change: both sides carry exactly one ``<audio>``
    tag bound to ``audios[0]``, and ``teacher_prompt`` is byte-identical to
    the student user content. The degraded and original waveforms have the
    same length (noise/muting preserves sample count), so the two sequences align
    token-for-token.

    The bidirectional guard mirrors _patch_teacher_mask_image: the
    ``teacher_audio_paths`` column's presence must match
    ``AURORA_OPSD_TEACHER_AUDIO == 'original'``. A mismatch is silent — the
    teacher would simply hear the degraded audio (no privilege) or an
    unexplained clean audio — and only surfaces days later in eval.
    """
    from swift.rl_core.data import OnPolicySample

    if getattr(OnPolicySample.to_teacher_template_dict, "_aurora_audio_patched", False):
        return

    original_to_teacher = OnPolicySample.to_teacher_template_dict

    def aurora_audio_to_teacher_template_dict(self):
        d = original_to_teacher(self)
        has_col = bool(self.extra.get("teacher_audio_paths"))
        if has_col != (_TEACHER_AUDIO == "original"):
            wanted_env = "original" if has_col else "(unset)"
            raise ValueError(
                "AURORA OPSD teacher-audio configuration mismatch: the dataset's "
                "teacher_audio_paths column and AURORA_OPSD_TEACHER_AUDIO disagree "
                f"(column={'present' if has_col else 'absent'}, "
                f"env={_TEACHER_AUDIO!r}). Rebuild the dataset with "
                "`scripts/data/prepare_refavs_opsd.py --privilege-kind audio-degrade` "
                f"and/or set AURORA_OPSD_TEACHER_AUDIO={wanted_env} in the "
                "training script."
            )
        if not _TEACHER_AUDIO:
            return d
        orig_paths = [str(p) for p in self.extra["teacher_audio_paths"]]
        student_paths = [str(p) for p in (d.get("audios") or [])]
        if len(orig_paths) != len(student_paths):
            raise ValueError(
                "AURORA_OPSD_TEACHER_AUDIO=original: teacher_audio_paths has "
                f"{len(orig_paths)} entries but the row carries {len(student_paths)} "
                "audio(s); they must pair 1:1."
            )
        for p in orig_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"teacher original audio does not exist: {p}")
        _audio_debug_count = getattr(self.__class__, "_aurora_audio_debug_n", 0)
        if _DEBUG_LOG and _audio_debug_count < 5:
            self.__class__._aurora_audio_debug_n = _audio_debug_count + 1
            print(
                f"[aurora-audio] request={self.request_id!r} "
                f"student={student_paths} teacher={orig_paths}",
                flush=True,
            )
        # CRITICAL: build a NEW list. d["audios"] (from _standard_fields) is a
        # reference to self.audios, and the omni template's replace_tag does
        # `inputs.audios[index] = load_audio(...)` IN PLACE — sharing the list
        # would write the teacher's clean waveform back onto the sample and
        # leak it into the student rollout via to_infer_request.
        d["audios"] = orig_paths
        return d

    aurora_audio_to_teacher_template_dict._aurora_audio_patched = True
    OnPolicySample.to_teacher_template_dict = aurora_audio_to_teacher_template_dict


_patch_sample_to_template_dict()
_patch_teacher_mask_image()
_patch_teacher_audio()
if _MASK_LOSS_ENABLED:
    _orig_entry = TrainerFactory.TRAINER_MAPPING.get("gkd")
    if _orig_entry and "AuroraGKDTrainer" not in _orig_entry:
        TrainerFactory.TRAINER_MAPPING["gkd"] = (
            "swift_plugin.aurora_opsd.AuroraGKDTrainer"
            if _orig_entry.startswith("swift.")
            else _orig_entry  # already patched by another plugin; don't stack
        )
