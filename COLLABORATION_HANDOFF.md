# OmniOPD collaboration handoff

This is a **source checkout**, not a self-contained copy of models and
REF-AVS. The paths below identify assets available on the current training
machine; they are not portable paths or a promise that these assets may be
publicly redistributed. Review rights and access before making a bundle.

## 0. Current release blockers & portability status

1. **Publish the reviewed revision to the default branch (`origin/HEAD`).**
   Confirm that the default branch (`main`) on `origin` includes the reviewed
   portability fixes, `scripts/data/`, vendored SAM/SAM2 and `verl/models`,
   `bootstrap_ms_swift.sh --install-deps`, reward fail-closed checks,
   checkpoint/template parity, and native MOPD media preprocessing. Until the
   collaboration branch is merged to `main`, a default `git clone` gets the
   older initial commit.
2. **Choose the project's own license (still external / open).** There is no
   top-level `LICENSE`. Keep the separate `third_party/MOPD-Router-Omni/LICENSE`,
   `third_party/MOPD-Router-Omni/verl/LICENSE`,
   `third_party/sam2/LICENSE`, and `model/segment_anything/LICENSE` with the
   included code. Do not assume that a dependency's license licenses this
   project's code or its training data.
3. **Control data/model access separately (external by design).** REF-AVS
   dataset terms, pretrained model-card terms, and rights to redistribute
   synthetic CoT / audio captions must be checked independently. Supply media
   and weights via an access-controlled store rather than committing them to Git.
4. **Data synthesis scripts (external / unbundled).** `scripts/README.md`
   documents 14 historical optional synthesis scripts absent from this trimmed
   checkout (including merged-SFT generation and several privilege synthesis
   steps). Reviewed JSONL manifests can be relocated with
   `scripts/data/relocate_media_paths.py`, while *from-scratch regeneration*
   still requires supplying those missing scripts, their prompts, model/API
   provenance, and validation tests separately.
5. **Portability without bypassing safeguards (code fixes applied).**
   - Launchers auto-detect `.venv` / `VIRTUAL_ENV`, support `.env`
     (`OMNIOPD_LOAD_ENV=1`), export `PYTHONPATH`, and accept overrides for
     `SFT_CHECKPOINT`, `DATA_ROOT`, `DATASET_AUDIT`, `EXPECTED_STEPS`, `MODEL`,
     `SAM2_CHECKPOINT`, `TRAIN_JSONL`, and `VAL_JSONL`.
   - `scripts/eval/omni_eval_settings.py` provides a CLI (`--data-dir`,
     `--splits`, `--output`) to generate a local `audit.json` from relocated
     splits, and standalone entrypoints auto-insert repo/SAM2 import paths.
   - `scripts/eval/test_qwen2_5_omni_refavs.py` and native evaluators keep
     strict split, hash, FPS, projector, trained-sidecar, and mask-saving
     verification intact while supporting relocated paths via CLI/env overrides.

### Rights/provenance items to resolve

The [official REF-AVS project page](https://gewu-lab.github.io/Ref-AVS/)
describes its released **dataset** under CC BY-NC 4.0; its
[code repository](https://github.com/GeWu-Lab/Ref-AVS) lists MIT. Treat
those as separate statements; verify the specific copies and third-party
media you intend to transfer. Official Qwen2.5-Omni
[3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) and
[7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) model cards list different
licenses (Qwen Research and Apache 2.0 respectively); a collaborator should
obtain and accept the applicable model weights/terms. The licenses for the
vendored SAM/SAM2 and MOPD/verl source are present in their own directories.
The terms under which externally generated CoT/captions may be transferred
must be verified from their actual generation services and agreements.

## 1. Share in three separate channels

| Channel | What a collaborator needs | What not to put in the code repository |
|---|---|---|
| **Code (Git)** | `swift_plugin/`, actual `scripts/` and `configs/`, `model/segment_anything/`, `third_party/sam2/sam2/`, `third_party/MOPD-Router-Omni/` including complete vendored `verl/models`, tests, source licenses, `AGENTS.md`, setup and method docs. Use commit IDs and code review. | `.venv*`, locally cloned `third_party/ms-swift/` (bootstrap it from its pinned tag plus patch), log files, `outputs/`, API credentials. |
| **Dataset (restricted/versioned storage)** | Authorized REF-AVS media and GT masks; `metadata.csv`, the specific `train`/`val`/`test_s`/`test_u` JSONL; structured CoT SFT and prompt-only GRPO/online-OPD manifests; generation/review provenance and SHA256. Split test access from training write access. | Raw media and masks in ordinary Git, unlicensed exports, raw keys or proprietary model/API responses without permission. |
| **Model/checkpoints (artifact store)** | Pretrained Omni base and SAM2 initialization weights from their approved sources; the SFT initializer and exact trained SFT adapter **including** SEG embedding, projector config and trained SAM2 sidecar; optional later GRPO/OPD checkpoints and teacher exports with provenance. | Optimizer shards or all historical checkpoints when only inference is needed; weights with unclear redistribution rights. |

Share a *small rights-cleared fixture* (one known ten-frame clip, WAV, ten GT
masks, expected JSONL) for CI/smoke. Even a single raw dataset sample should
not be published until its sharing terms have been checked. Tests using
synthetic tensors and dummy media can be public without a dataset fixture.

## 2. Dataset inventory on the current machine

The current REF-AVS layout is:

```text
REFAVS/
├── metadata.csv                    # vid, uid, split, fid, exp
├── media/<clip_id>/audio.wav
├── media/<clip_id>/frames/{0,...,9}.jpg
└── gt_mask/<clip_id>/fid_<instance>/*.png
```

Current source is a separately mounted, access-controlled REF-AVS directory
(`$REFAVS_ROOT`), not part of this repository.
`media/` is approximately **50 GiB**, `gt_mask/` **466 MiB** and
`metadata.csv` **2.2 MiB**. This is a storage estimate, **not** a distribution
license. Timing audit:
`$EXPERIMENT_ROOT/outputs/artifacts/refavs_fps_audit_20261025/audit.json`.
There are 3566 distinct audited clips across the manifests, with ten stored
frames per roughly ten-second clip and approximately ten-second audio: nominal
**1 FPS**, exact original frame timestamps unknown. Never infer that a model
trained with a 2-FPS timeline used a 2-FPS source dataset.

| Logical artifact (host path) | Rows | Source SHA256 | Use |
|---|---:|---|---|
| SFT `outputs/data/cot/refavs_merged_sft_20261103_schemafix_v2/structured_cot.jsonl` | 12820 | `87d636211abd37fb24370253ffee7168839b3df3e2c33a2dd665554e8d9a41c5` | 7B reviewed CoT, user + assistant. Shares exact hash with the recorded 3-epoch SFT source. The actual distributed run padded 4 rows to 12824; keep its padding manifest too. |
| GRPO `outputs/data/merged_grpo_20261102_merged_v4_full/train.jsonl` | 12820 | `4b71f21d1658aeda5fa4792f09715973e623afc28d2888cede49df3324241e00` | Prompt-only, assistant labels removed. Include `manifest.json` and validation JSONL (1277 rows). |
| Online OPD `outputs/data/merged_student_only_online_20261103_v1/train.jsonl` | 12820 sampled / 6094 unique IDs | `4f6401e52743c8dc6574817fc61eef4f689de27884c87b3a3e008083e989e196` | Student-only with difficulty buckets and audio/visual privilege text; **not** an offline teacher trajectory cache. Include `manifest.json`, `audit.json` and probe provenance. |
| Official structured `outputs/data/refavs_official_template_matched_20261102/refavs_test_s.jsonl` | 2288 | `a75ff4b056284d3728e4d8392929a8f8e9d1ab338339fb41159b168197da46eb` | Evaluation only. |
| Official structured `outputs/data/refavs_official_template_matched_20261102/refavs_test_u.jsonl` | 1656 | `ee342befc29b83f8a448a175687f42788b1e9f685bab09609351b028d6dccf25` | Evaluation only. |

`outputs/data` above is relative to
`$EXPERIMENT_ROOT` (the original experiment workspace), **not** this code checkout.
Every JSONL row contains `videos`, `audios`, `sam_frame_paths`, `mask_paths`,
`messages`, `sample_id`; OPD rows may add privilege text and difficulty
metadata. The referenced JPEG/WAV/PNG files are *not* inside the JSONL
folders. The train/test CoT prompt header must match the selected
checkpoint; see `scripts/docs/LLM_TEMPLATE_AUDIT_20261117.md`.
The official evaluation JSONL also contains an assistant field for
annotation/protocol compatibility: inference must pass only the **user**
message (`messages[:1]`), never that assistant answer or GT masks as input
to the LLM.

For a different mount point, first obtain authorized media, then relocate
each JSONL separately:

```bash
python scripts/data/relocate_media_paths.py \
  --source /restricted/manifests/refavs_test_s.jsonl \
  --output /restricted/relocated/refavs_test_s.jsonl \
  --old-root /original/REFAVS \
  --new-root "$REFAVS_ROOT"

# After relocating splits, generate a local dataset timing audit JSON:
python scripts/eval/omni_eval_settings.py \
  --data-dir /restricted/relocated \
  --splits test_s test_u \
  --output outputs/artifacts/audit.json
```

The tool checks every referenced file by default, retains row labels and
messages, rejects paths outside the declared media root, refuses overwrites
and writes a `.media_manifest.json` with **both** pre-/post-relocation hashes.
It only handles its documented media-path fields; audit novel privilege
columns separately. Never merge `test_s`/`test_u` into a training manifest.

## 3. Model artifact inventory

| Purpose | Required material | Notes |
|---|---|---|
| Base Omni | `Qwen2___5-Omni-7B` (current local copy ~21 GiB) plus its processor/tokenizer/config; 3B (~12 GiB) only for an explicitly chosen 3B experiment | Obtain from the authorized model source; do not silently switch model size or tokenizer. |
| SAM2 base | `sam2_hiera_large.pt` (~857 MiB), matching `configs/sam2/sam2_hiera_l.yaml` in vendored SAM2 | Initializes image encoder/base; **not** the trained mask decoder. |
| 3-epoch 7B SFT initializer | `outputs/train/sam2_sft_initializer_from_answer_cot_7b_20261030_v6/adapter_model.safetensors`, base model and SFT manifest | Needed to **retrain** the current SFT recipe. This initializer is not the 3-epoch SFT result. |
| 3-epoch 7B SFT result | `outputs/train/sam2_merged_sft_3epoch_8gpu_20261102_merged_v4_full/checkpoint-4809/` | For inference/continued LoRA training, transfer `adapter_model.safetensors` (~2.1 GiB), `adapter_config.json`, `args.json`, `aurora_projector_config.json` and **trained** `sam2_video_decoder.safetensors` (~23 MiB), along with the parent `training_manifest.json`. Retain optimizer/RNG shards **only if exact resume is required**. |
| Native reasoning-only MOPD | A complete BF16 merged Thinker export (~17 GiB model shards) plus processor/tokenizer/config/export manifest; optionally the independently specialized frozen teacher exports and a genuine shared pre-specialization base | A teacher/model name alone is insufficient. The existing `native_mopd_step500_v1` was trained under historical capped-video preparation; its manifest and original input protocol must accompany it. Do **not** use arbitrary checkpoints as three “experts.” |

SFT trained FP16, some post-training runs BF16; model timeline 1 FPS on
the selected 7B run, visual limits `FPS_MAX_FRAMES=768`,
`VIDEO_MAX_TOKEN_NUM=768`, `IMAGE_MAX_TOKEN_NUM=16384`,
`max_pixels=200704`, `max_length=5120`, SAM frames=10, SDPA.
Always read each checkpoint's actual `args.json`, projector config, launcher
and manifest rather than applying these numbers to every old experiment.
Native MOPD's old video-grid mismatch was fixed **for future training only**;
historical step500 must use its historical evaluation media path.

## 4. Reproducibility packet (code + provenance, not just a checkpoint)

For each experiment, attach:

- **Version/recipe:** Git commit and diff, upstream ms-swift v4.5.2 commit
  `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` plus
  `scripts/setup/ms-swift-v4.5.2-teacher-adapters.patch`, SAM2 revision,
  Python/CUDA/torch/transformers versions, launcher and env vars (redacted).
  `scripts/setup/bootstrap_ms_swift.sh` (`--install-deps` or `--no-deps`)
  reconstructs the patched checkout.
  SFT/GRPO/OPD and native MOPD overlays used **different installed NumPy
  versions**: consult their separate observed-version files under
  `scripts/setup/`; produce a clean-machine full lockfile before claiming
  identical environments.
- **Data provenance:** authorized dataset source and terms; split/clip IDs,
  annotation/model-generated-CoT revision, generation prompt and reviewer
  policy, filter/keyword rules, category/privilege/difficulty proportions,
  media root mapping and SHA256 of every manifest. The reviewed SFT CoT
  folder by itself cannot regenerate historical missing synthesis scripts.
- **Architecture/objective:** Omni/SAM2 base and trained mask sidecar,
  tokenizer `[SEG]` ID, LoRA targets/modules_to_save, max sequence/response,
  FPS/media token limits, sampler/batch/global steps/seeds, teacher origin,
  frozen vs current-policy update schedule, router, loss and reward formulas,
  failed-reward policy and dependency on a SAM IoU server if applicable.
- **Evaluation:** exact checkpoint, source split/hash, decode settings,
  `test_s` and `test_u` separately, class exact, `[SEG]` rate, per-frame
  IoU/Dice and **saved frame masks** by default; refusal of known preprocessing
  mismatches and a list of intentional differences. For an OPD gain, include
  the matched no-OPD control with the same initializer, mixture, optimizer,
  budget and evaluation protocol. A four-case smoke or validation metric
  is not benefit evidence.
- **Training-health traces:** GRPO reward columns and their nonzero/variance
  rates (a completed step can have *zero* verifiable reward), `[SEG]` rate,
  per-teacher routing/eligibility and active OPD token counts, KL/gradient
  norms, class-vs-mask breakdown, failure counts and NaN/OOM logs.
  Record CPU/GPU counts and job allocation for full recipes.

The authoritative current rules are in `AGENTS.md`. Demonstrated
single-card connectivity and 7B/native template audits are in
`scripts/docs/INDEPENDENT_VERIFY.md` and
`scripts/docs/LLM_TEMPLATE_AUDIT_20261117.md`; the 42-GiB temporary
`outputs/verify/` tree is **not** a collaboration artifact.
The new `.github/workflows/source-contract.yml` uses only Python's standard
library to check required source, Python/Bash syntax, local links, tracked
large-media/credential files and a synthetic media-path relocation fixture:

```bash
python3 -S scripts/setup/repo_doctor.py --strict
python3 -S -m unittest discover -s tests -p test_repository_contract.py -v
```

It intentionally does **not** download any model, run GPU training, audit
dataset licenses, or declare the repository ready for public release.
`CONTRIBUTING.md` and `.github/pull_request_template.md` describe the
additional human and GPU checks.
The merged SFT, SAM2 GRPO and current-policy OPD reference launchers require
**eight GPUs** for their formal recipes. The 3B single-card 1-step examples
only establish engineering connectivity; they are not a resource estimate
for 7B/three-teacher training. Agree on GPU quota, dataset/artifact storage,
the experiment queue and ownership of official test runs before collaboration.

## 5. Review checklist before sharing

- [ ] Choose private collaborators/roles and approve dataset/model/CoT
      distribution, attribution and third-party source notices; add the
      project's own license if publication is intended.
- [ ] Review and merge the code changes, README corrections and lightweight
      source CI; bring back the historical synthesis code only if required
      for from-scratch regeneration.
- [ ] Create a controlled data release with manifest, checksums, a path
      relocation smoke and authorized media; separate official test write access.
- [ ] Supply the **minimal** model/decoder/SEG components needed for the
      agreed task; avoid copying the entire 17-GiB optimizer checkpoint when
      only inference is needed.
- [ ] Perform one clean-machine SFT, GRPO and OPD *optimizer update* with
      nonzero, audited rewards/KD where applicable; run matched official
      `test_s` then `test_u` before any claimed benefit.
- [ ] Exclude `.env`, token files, API credentials, raw log secrets, private
      host paths from public examples, caches and generated test media.

This document is an engineering inventory, **not legal advice**. In
particular, a code license, a model license and a dataset/media license are
three distinct reviews.
