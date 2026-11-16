# Clean Omni reasoning-only MOPD-Router migration

## Scope / 不与旧迁移混用

- Fresh clone: `https://github.com/TURLEing/MOPD-Router.git`
- Upstream baseline: `a630dc3891c5b1c92cdc2815193fdcbd8fa4291b`
- Branch: `codex/omni-reasoning-only`
- Repository:
  `third_party/MOPD-Router-Omni`
- Existing `third_party/MOPD-Router`
  and running experiments were not edited.
- No imports from `swift_plugin`, `refavs_migration`, SAM or the historical
  Joint trainer. A native HF **Thinker** is loaded; no segmentation head is
  instantiated. The old trained segmentation checkpoint remains separate and
  untouched.

This is **pure native MOPD**, not GRPO+OPD Joint:
`only_reverse_kl_advantages=true`, `lambda_vals=1`, task reward skipped,
reference KL disabled. Entropy / accessible novelty / ExpertAlign are selectable
alternatives, not three simultaneously multiplied gates. Default ExpertAlign.
No category/IoU license, stage mask, low-likelihood filter, radius or forced
single-teacher fallback is added.

`run.sh`, `dp_actor.py`, `teacher_routing.py`, `ray_trainer.py` and
`core_algos.py` are **byte-identical upstream**, enforced by preflight/tests.
This includes native signed reverse-KL and PPO update logic.

## What is adapted

| File | Purpose |
|---|---|
| `omni_mopd/data.py` | Raw REF-AVS JSONL, real ten-frame video + original WAV, native processor and mRoPE |
| `omni_mopd/modeling.py` | Native HF mRoPE helper; language-only parameter freeze; vocabulary checks |
| `omni_mopd/rollout.py` | Audio/video-aware synchronous HF generation sharing the FSDP actor |
| `omni_mopd/export_thinker.py` | Explicitly import language LoRA/embed/head, merge, export standalone Thinker; reject missing trained embeddings |
| `omni_mopd/preflight.py` | Verify upstream core, independent teacher paths, shared vocabulary/lineage, audit media timing and write manifest |
| `verl/verl/workers/fsdp_workers.py` | Select native Thinker, freeze non-language modules, connect HF rollout in the **original Ray/FSDP worker** |
| `verl/verl/trainer/main_ppo.py` | Verify common token IDs; avoid text-only re-tokenization of media-expanded Omni prompts |
| `verl/verl/utils/model.py` | Transformers 5 AutoModel class-name compatibility only |

Native vLLM, remove-padding, fused kernels, Ulysses SP>1, FSDP2 and adapter-as-ref
paths are not claimed compatible: this adapter rejects those combinations.
HF rollout is an implementation difference from the paper's vLLM backend;
RayPPOTrainer, native FSDP worker orchestration and router/loss are retained.

### Optimized parameters and tokens

- Train native `model.layers.*` and `model.norm.*` (full language decoder,
  **not** the old q/v-only LoRA recipe).
- Freeze visual/audio towers, embedding and LM head. No SAM/projector exists.
- Keep the native **whole response** completion mask, including `[SEG]`/EOS if
  generated. “Reasoning-only” describes the language model path/parameters;
  it does **not** introduce another per-CoT-stage token gate.
- Perception features are still provided: this is not text-only/no-audio input.
- No guarantee that frozen-head segmentation improves: changing the Thinker's
  hidden states can change future `[SEG]` features even if SAM is unchanged.
  A later full segmentation evaluation must reattach the *trained* saved
  projector/SAM2 and save all masks. No segmentation test is run here.

### Teacher contract

The launcher requires **three independent frozen merged Thinker checkpoints**
and an explicit shared pre-specialization base, all sharing token IDs.
They score the same Student prompt/media/response prefixes. They do not sample
their own trajectories, and do not change with Student policy updates.

Names `MATH_TEACHER_MODEL`, `CODE_TEACHER_MODEL`,
`INSTRUCT_FOLLOWING_TEACHER_MODEL` are the upstream channel names, **not a claim
that REF-AVS has math/code experts**.

Existing Audio/Visual privileged *current-policy views* are NOT silently used
as these independent teachers. That would be a separate algorithmic adaptation,
not “only replacing the model” in the original method.
No arbitrary historical checkpoints have been selected as a scientifically
justified expert pool. They must be supplied explicitly before real training.
The exported SFT initializer below is not itself three specialized teachers.

## Media / optimizer settings

Sources: initializer
`outputs/train/sam2_merged_sft_3epoch_8gpu_20261102_merged_v4_full/checkpoint-4809/args.json`,
its parent `training_manifest.json`, and
`outputs/artifacts/refavs_fps_audit_20261025/audit.json`.

- Ten stored frames per approximately ten seconds: nominal **1 FPS**, exact
  sample instants unverified. Model timeline fixed to 1 FPS.
- max_pixels=200704; frame/token settings 768/768/16384; native SDPA.
- Prompt<=4864 + response<=256 = sequence<=5120. No truncation or silent
  overlength filtering; an invalid row raises an error.
- BF16 compute/reference weights; FP32 actor parameters/optimizer (native
  mixed-precision training, not BF16-only optimizer state).
- Merged exports use BF16. Original SFT was FP16 and used unmerged LoRA;
  merging/casting is an explicit numerical difference, not an equivalence claim.
- LR=1e-5 constant, clip=0.2, grad clip=1, no reference KL; native defaults.
- Rollout temperature=1, **top_p=1**, top_k=0, n=1. No top-p/full-softmax mismatch.
- Default **2 GPUs / batch2 / mini-batch2 / 3-step smoke**, not the paper's
  batch1024, three-epoch budget, and not the old group8 GRPO recipe.
- Language-only run: mask saving is explicitly inapplicable; preflight records
  this exception. Debug validation uses training rows and is disabled; official
  `test_s`/`test_u` are not used for training.

## Environment

```bash
cd third_party/MOPD-Router-Omni
bash bootstrap_omni_env.sh
export PYTHON_BIN=/path/to/mopd-router-omni-v1/bin/python
export PYTHONPATH="$PWD:$PWD/verl"
```

## Export

```bash
"$PYTHON_BIN" -m omni_mopd.export_thinker \
  --base models/Qwen2___5-Omni-7B \
  --adapter /path/to/sam2_sft_checkpoint \
  --out /absolute/NEW/merged_thinker_export \
  --device cpu
```

Export teachers in the same explicit way using their own adapter sources.
For a raw native base, omit `--adapter`; do not call an arbitrary tuned
checkpoint “pre-specialization base” merely to pass a path check.

## Launch (only after independent teachers are available)

```bash
export STUDENT_MODEL=/absolute/merged_student_thinker
export TEACHER_BASE_MODEL=/absolute/shared_pre_specialization_thinker
export MATH_TEACHER_MODEL=/absolute/merged_frozen_teacher_1
export CODE_TEACHER_MODEL=/absolute/merged_frozen_teacher_2
export INSTRUCT_FOLLOWING_TEACHER_MODEL=/absolute/merged_frozen_teacher_3
export CKPT_DIR=/absolute/NEW/output_directory
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS_PER_NODE=2
export TEACHER_SIGNAL_AGGREGATION=delta
export TOTAL_TRAINING_STEPS=3
bash run_omni.sh
```

`DRY_RUN=1 bash run_omni.sh` validates real model/data manifests without Ray/model
loading; it still creates a **new** manifest directory. Use a fresh output path.
The wrapper intentionally refuses existing output directories.

## Verification

```bash
"$PYTHON_BIN" -m unittest omni_mopd.test_migration -v
```

Tests cover native-core hashes, no assistant/privilege leakage, mask-free
exports, exact vocabulary, actual tiny HF Omni audio/video+mRoPE forward vs
native actor logp, native reverse-KL advantage replacement/direction, real
optimizer update and frozen non-language tensors.

All model-benefit claims require matched official `test_s`/`test_u`
comparisons under the root AGENTS.md protocol.
