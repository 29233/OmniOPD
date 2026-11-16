# AURORA SFT on ms-swift

This first migration milestone supports Qwen2.5-Omni-3B with the native
ms-swift multimodal processor and an in-process AURORA SAM head. It is limited
to SFT; VideoLLaMA2 registration is intentionally a later milestone.

The same registration now accepts the local Qwen2.5-Omni-7B checkpoint. The
current 7B memory boundary is documented below; the 3B smoke command remains
the default because it is faster to iterate.

## What is trained

`[SEG]` is added through ms-swift's `--new_special_tokens` API. The SFT
objective is:

```text
language CE + 2.0 * mask BCE + 0.5 * mask Dice
```

The language model uses LoRA on `thinker.model` Q/V projections. The complete
`embed_tokens`, `lm_head`, `text_hidden_fcs`, and SAM `mask_decoder` modules are
saved. The SAM image encoder, Qwen vision tower, Qwen audio tower, and
multimodal projector remain frozen.

## REF-AVS prompt contract

The Qwen2.5-Omni Stage-1 data path intentionally has no `system` turn. Each
user message is rendered as:

```text
<video><audio>The reference is: {exp} Please segment the corresponding object in the video.
```

`{exp}` is copied verbatim from `REFAVS/metadata.csv`. It can describe visual,
temporal, spatial, or audio-related properties; the instruction therefore says
`corresponding object`, not `corresponding sounding object`. The older
VideoLLaMA2/LLaVA dataset helpers use a different safety `system_message` and
the wording `...in the images.`; those templates are not used by this Omni
migration.

## Precision and loss scaling

The smoke command loads the model with `--torch_dtype float16` and enables
ms-swift/Transformers FP16 mixed precision (`fp16=true`, `bf16=false`). The
test machine uses Tesla V100 GPUs, which do not provide native BF16 support.
Attention uses `sdpa`; Flash Attention is not enabled. Trainable parameters are
converted to FP32 by ms-swift/PEFT where applicable, while the frozen Qwen
media towers and SAM image encoder run in their loaded FP16 dtype. SAM mask
losses are accumulated in FP32 for numerical stability.

DeepSpeed ZeRO-2 uses the repository configuration
`configs/deepspeed_zero2_fp16_v100.json`, with dynamic FP16 loss scaling and an
initial scale of `2^8` (256). The default DeepSpeed initial scale (`2^16`) was
observed to overflow on this joint loss: five steps were skipped, with
`overflow=true` and no parameter changes. The reduced initial scale produces
non-zero gradient norms and optimizer updates on V100.

## Data conversion and training

The data converter (`scripts/data/prepare_refavs_swift_sft.py`) writes standard
`messages/videos/audios` fields plus the custom `sam_frame_paths/mask_paths`
supervision fields (`AURORA_NUM_SAM_FRAMES=10` by default). `images` is deliberately
not used for ground-truth masks.

The converter accepts `--split train`, `--split val`, and the REF-AVS test
splits (`test_s`, `test_u`, `test_n`). Training reasoning is used only for the
train split. Validation and test records use the deterministic teacher-forced
answer `It is [SEG].`, while retaining split-specific reference text and SAM
mask paths.

## Full training entry point and checkpoints

Use `scripts/sft/train_qwen2_5_omni_swift_sft.sh` for a full-data run. It defaults
to `MAX_SAMPLES=-1`, `MAX_STEPS=-1` (the configured epoch count), 10 SAM frames,
and saves/evaluates every 500 optimizer steps:

```bash
NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/sft/train_qwen2_5_omni_swift_sft.sh
```

Override `SAVE_STEPS` to change the interval.

The full entry point generates `outputs/data/refavs/refavs_val.jsonl` from the REF-AVS
`val` split and passes it as `--val_dataset`. `eval_strategy=steps` and
`eval_steps` follow `SAVE_STEPS` by default, so ms-swift evaluates at each
interval before saving the matching checkpoint. With
`predict_with_generate=True`, the validation pass is generation-based and the
generated collection is written to `predict.jsonl`; set `EVAL_STEPS` to use a
different evaluation cadence.

TensorBoard logging is enabled by default with `--report_to tensorboard`.
Logs are written below the run's `runs/` directory. Use `REPORT_TO=none` to
disable it, or run `tensorboard --logdir /path/to/output/runs`.

Evaluation uses ms-swift's native `predict_with_generate=True` path. The
`Seq2SeqTrainer` uses the official `PtEngine` generation implementation,
partitions evaluation across all ranks, gathers the generated predictions, and
appends them to `predict.jsonl` in the run directory. Set
`AURORA_FREE_EVAL_MAX_TOKENS` to change the generation cap (the default is
128). This online validation pass is LLM-only: it records the generated CoT,
whether `[SEG]` was emitted, and the reference labels. Full segmentation testing
on `test_s` and `test_u` is run via `scripts/eval/test_qwen2_5_omni_refavs.py` or
`scripts/eval/eval_omni_two_stage.py`.

Every ms-swift checkpoint's `adapter_model.safetensors` includes the
trainable SAM mask decoder because `mask_decoder` is listed in
`--modules_to_save`. The external callback additionally writes a standalone
`sam_mask_decoder.safetensors` (120 tensors) into the same checkpoint directory.
The SAM image encoder and prompt encoder are frozen and are intentionally not
duplicated in this sidecar; the original full SAM checkpoint remains the source
for those frozen components.
