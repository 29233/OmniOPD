# SAM2 SFT baseline GRPO

This experiment starts from a completed three-epoch 7B SAM2 SFT checkpoint
(specified via `SFT_CHECKPOINT`).

It uses two sequence rewards with no separate format reward:

1. `aurora_class_exact` is 1 only when the last `The target is ...` prediction
   exactly matches the normalized REF-AVS category; otherwise it is 0.
2. `aurora_sam2_mask_iou` is the mean binary IoU of all ten masks decoded from
   the first generated `[SEG]` hidden state by the trained projector and SAM2
   video-memory branch. Missing `[SEG]` gives 0.

The projector, Hiera encoder, SAM2 mask decoder, and SAM2 memory modules are
frozen reward decoders. There is no supervised BCE/Dice auxiliary loss. The
policy updates q/v LoRA, `embed_tokens`, and `lm_head` through native ms-swift
GRPO. The total scalar reward is the weighted sum of class correctness and
mask mIoU; both weights default to 1.

## Data and media settings

The launcher derives prompt-only GRPO rows from the same structured prompt
source used by SAM2 SFT and strips every assistant response. All 10,201 train
rows have a valid category. Validation contains 64 REF-AVS `null` category
rows; these are excluded because no class prediction can match them, leaving
1,277 validation rows over the same 38 valid classes.

The audited dataset has ten stored frames over each approximately ten-second
clip, so nominal data coverage is 1 FPS. Exact sampling instants are not
available. The checkpoint also used a 1 FPS model timeline, ten SAM2 frames,
`max_pixels=200704`, `max_length=5120`, FP16, and SDPA; the launcher rejects a
checkpoint or dataset manifest that does not match these settings.

## Launch and defaults

Run the complete preflight without loading models:

```bash
DRY_RUN=1 bash scripts/grpo/train_qwen2_5_omni_7b_sam2_grpo_8gpu.sh
```

Launch on eight GPUs after approving the experiment parameters:

```bash
bash scripts/grpo/train_qwen2_5_omni_7b_sam2_grpo_8gpu.sh
```

Defaults are `num_generations=8`, `generation_batch_size=8`, one epoch,
`lr=1e-6`, `beta=0.04`, `temperature=1.0`, `top_p=0.9`, completion length 512,
and equal reward weights. Every value can be overridden by an environment
variable with the corresponding uppercase name.

The mask reward currently fails closed: any unexpected SAM2 reward exception
terminates training instead of silently assigning zero. Set
`AURORA_GRPO_REWARD_FAIL_OPEN=1` only for explicit diagnostics.

