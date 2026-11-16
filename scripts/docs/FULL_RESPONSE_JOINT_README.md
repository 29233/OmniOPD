# Full-Response Joint OPD

Token-scope ablation against the stage-localized, low-likelihood Online Verified Joint OPD recipe.

## Protocol

- 8 GPUs, group size 8, 1,000 optimizer steps, save every 250 steps.
- LR 1e-7, reference KL 0.01, BF16 SDPA, response cap 256, `max_length=5120`,
  `max_pixels=200704`; q/v LoRA only; trained SFT projector/`[SEG]`/SAM2 preserved.
- `0.5*C + 0.5*C*IoU` GRPO reward (`task_coef=1.0`), KD coefficient `0.25`.
- Online category-rescue OR both-correct IoU gain > 0.01 license,
  verified soft weights and hard routing.
- Audio Teacher remains disabled for Visual/Semantic; both views receive
  current Student adapters, refreshed every rollout/update, detached.

## Algorithmic Scope

`AuroraFullResponseOPDTrainer` (`swift_plugin/aurora_full_response_opd.py`) replaces
stage/low-likelihood selection with the entire native `completion_mask`, for every
applicable Teacher (`fraction=1.0`, `radius=0`, no stage restriction and no
lowest-likelihood selection). This includes reference, all evidence, reasoning,
decision, segmentation, `[SEG]`, EOS and valid template-added suffix positions;
only prompt/media/padding are excluded. License still limits which trajectories receive OPD.

The inherited sampled native reverse-KL advantage is unchanged:
`0.25 * verified_weight * (teacher_logp - old_student_logp)`.
Negative values are kept; `k3` remains monitoring only.

## Launch

```bash
MODEL=/path/to/Qwen2___5-Omni-7B \
AURORA_SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt \
SFT_CHECKPOINT=/path/to/sam2_sft_checkpoint \
DATA_ROOT=/path/to/student_only_opd_data \
OUTPUT_DIR=/path/to/output_dir \
bash scripts/opd/run_full_response_joint_8gpu.sh
```
