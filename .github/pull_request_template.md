## Scope

- Recipe/path changed (SFT / GRPO / OPSD / native MOPD / evaluation / data):
- What is *not* claimed by this PR:

## Reproduction contract

- Source and base checkpoint(s), including trained SEG/projector/SAM sidecar:
- Train data/split and manifest SHA256 (not raw dataset contents):
- Model timeline FPS, frame/token limits, dtype and max prompt/response:
- Optimizer budget, seed and expected no-OPD control (if relevant):
- Teacher source, routing/license/token span and reward-failure policy:

## Verification

- [ ] `python scripts/setup/repo_doctor.py --strict` and dependency-free tests pass.
- [ ] Correct checkpoint/template and data paths; no test split in training.
- [ ] Bounded *real* optimizer step/checkpoint checked, or clearly marked as not run.
- [ ] Reward/KD active coverage and non-finite/OOM failures audited.
- [ ] Official `test_s`/`test_u` evaluation and matched control provided for benefit claims.
- [ ] Valid predicted masks and per-frame metrics saved for segmentation evaluation.
- [ ] No private media, checkpoint shards, secrets or third-party material without approval.

Links to manifests, logs and evaluation summaries (access-controlled):
