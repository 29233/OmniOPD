# Contributing to OmniOPD

This is research code with external media and model artifacts. A green
source-only CI job means the checkout is structurally sound; it **does not**
demonstrate working eight-GPU training or a gain on REF-AVS.

## Before you change an algorithm

1. Identify the recipe and write down its initial checkpoint, training
   split/hash, model and trained SAM components, FPS/visual limits, optimizer,
   sample mixture, teacher roles, output budget and proposed control.
   Prefer a small PR for one algorithmic change and its tests over editing
   many historical launchers simultaneously.
2. For 7B SFT/GRPO/OPD, follow `AGENTS.md` and
   `COLLABORATION_HANDOFF.md`. **Never** use `test_s`/`test_u` as training
   data or use a validation/smoke metric as final improvement evidence.
3. Before switching a checkpoint's media preprocessing or CoT prompt,
   compare its saved `args.json`, training/launch manifest and trained
   projector/SAM sidecar. Historical native MOPD step500 must retain its
   own capped-video evaluation path; changing an evaluator does not retrain
   that checkpoint.

## Changes and checks

```bash
python scripts/setup/repo_doctor.py --strict
python -m unittest discover -s tests -p 'test_repository_contract.py' -v
```

These are dependency-free checkout checks. With an explicitly provisioned
Python/CUDA environment and `PYTHONPATH="$PWD/third_party/sam2:$PWD"`:

```bash
python -m unittest discover -s tests -p 'test_independent_core.py' -v
cd third_party/MOPD-Router-Omni
PYTHONPATH="$PWD:$PWD/verl" python -m unittest \
  omni_mopd.test_migration omni_mopd.test_train_pipeline -q
```

For a training or reward change, include a bounded *real* optimizer-step
test: nonzero reward/KD coverage (when claimed), finite loss/grad, a
loadable checkpoint, and a diagnostic inference that actually emits `[SEG]`
before claiming segmentation works. Full benefit evidence requires the
official test splits and a matched no-OPD control for OPD attribution.
For SAM/SAM2 evaluation, save every valid predicted frame mask and record
per-frame IoU/Dice in the case JSONL and manifest. CI does not download
weights, data or CUDA wheels.

## Pull request information

- What recipe/algorithm is changed and why? Which files are upstream versus
  local modifications?
- Exact code commit, checkpoint provenance, data manifest SHA256 and
  non-secret launcher/environment differences.
- Expected teacher routing and active token selection, verifiable reward
  availability, `[SEG]` rate and relevant class/mask diagnostics.
- Which CPU tests, single-step tests and official `test_s`/`test_u` comparisons
  were run? What is *not yet* verified?
- Were any prompt format, media path, FPS, dtype, trained SAM components or
  token/response-length limits changed? If so, how are historical
  checkpoints protected from incompatible replay?

Do not commit `.env`, credentials, media, GT masks, pretrained/checkpoint
weights or `outputs/`. Large files and data access must be arranged through
the approved artifact channel. Repository-root licensing and synthetic-data
redistribution decisions remain with the rights holders.
