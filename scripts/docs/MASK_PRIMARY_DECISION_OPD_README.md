# Mask-Primary GRPO / Outcome-Licensed Decision OPD

No LM-as-judge is used. This recipe does **not** claim that a correct mask
verifies free-text reasoning or audible evidence.

## Experimental Arms

- **G1 (`ARM=grpo`)**: Task reward `valid_SEG * (0.2*C + 0.8*IoU)`, no Teacher.
- **G2 (`ARM=decision`)**: Exactly G1's task reward plus native sampled reverse-KL Decision-category OPD.

G2 vs G1 within the same seed is the matched OPD comparison.

## Shared Protocol

- 8 GPUs, group size 8, 1000 optimizer updates, save every 250 steps.
- q/v LoRA only, LR 1e-7, cosine scheduler, reference KL beta 0.01,
  generation temperature 1.0 / top-p 0.9, BF16/SDPA, completion cap 256.
- Dataset: 10 stored frames over approximately 10 seconds, nominal 1 FPS;
  model timeline 1 FPS, SAM 10 frames.
  `FPS_MAX_FRAMES=768`, `VIDEO_MAX_TOKEN_NUM=768`, `IMAGE_MAX_TOKEN_NUM=16384`,
  `max_pixels=200704`, `max_length=5120`.
- Embeddings/head/projector/SAM2 are frozen during post-training and loaded
  from the trained SFT checkpoint.

Native `reward_weights` remain `[0.5, 0.5]`. The opt-in plugin (`swift_plugin/aurora_mask_primary.py`)
scales the raw columns to `[0.4*valid*C, 1.6*valid*IoU]`, producing the exact stated sum.
Invalid/multiple `[SEG]` completions receive zero reward. The reward decoder is fail-closed.

## G2 License and Token Scope

Current Student with adapters enabled, stop-gradient, audio and visual
privileged prompt views, one fresh trajectory per enabled view at every step.
Audio Teacher is disabled on Visual/Semantic rows.

The teacher must:

1. correct a Student category-exact error;
2. have one valid `[SEG]` and GT-correct category;
3. have mean 10-frame IoU >= 0.5;
4. not regress Student IoU by more than 0.01.

Same-class mask gains do not license Decision-category distillation. All BPE
tokens overlapping the single `Decision: The target is <category>` category
span are eligible, with no bottom-likelihood truncation or radius expansion.
Evidence, boilerplate, `[SEG]` and EOS are excluded. Two eligible Teachers are
routed by verified weighting, one Teacher per token, `kd_coef=0.25`.

## Launch

```bash
ARM=decision \
SFT_CHECKPOINT=/path/to/sam2_sft_checkpoint \
DATA_ROOT=/path/to/student_only_opd_data \
OUTPUT_DIR=/path/to/output_dir \
bash scripts/opd/run_mask_primary_train_8gpu.sh
```
