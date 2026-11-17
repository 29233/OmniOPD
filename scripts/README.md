# AURORA Scripts Directory Index

> **Scope:** This is an index of historical recipes. The trimmed OmniOPD
> checkout does not contain every synthesis/preparation script named below
> (for example `prepare_merged_sft_dataset.py`,
> `generate_refavs_structured_cot.py`, and
> `prepare_sam2_sft_initializer.py`). The already-generated reviewed
> manifests or the missing scripts and their dependencies must be supplied
> separately before claiming from-scratch data regeneration. Use
> `COLLABORATION_HANDOFF.md` for the verified minimal input set.

**Historical-only source not in this checkout:** `call_doubao_video.py`,
`check_refavs_structured_cot.py`, `generate_non_audio_answer_cot.py`,
`generate_refavs_clip_audio_privilege.py`,
`generate_refavs_first_mask_visual_privilege.py`,
`generate_refavs_structured_cot.py`, `match_refavs_templates.py`,
`prepare_merged_opd_source.py`, `prepare_merged_sft_dataset.py`,
`prepare_privileged_top10_opd.py`, `prepare_refavs_visual_opsd.py`,
`prepare_sam2_sft_initializer.py`,
`prepare_verified_opd_probe_conditions.py` and
`refavs_audio_keyword_policy.py`. Their names below document historical
pipeline provenance; none is a working command in this trimmed checkout.

This directory organizes all data preparation, training, evaluation, and utility scripts by functional stage:

```text
scripts/
├── setup/   # Dependency-free repo doctor, ms-swift v4.5.2 bootstrap & patch, observed requirements
├── data/    # Dataset conversion, media-path relocation, audio degradation, and privilege preparation
├── sft/     # Stage-1 / Stage-2 / SAM2 supervised fine-tuning (SFT) launchers
├── grpo/    # GRPO reinforcement learning launchers and SAM IoU reward server
├── opd/     # On-Policy Self-Distillation (OPSD) and Verified Routed / Online OPD launchers
├── eval/    # Official Ref-AVS evaluation, timing audit CLI, and two-stage batch evaluation entrypoints
├── utils/   # Model weight export tools and upstream model helpers
└── docs/    # Algorithm design notes and usage guides for each training/eval recipe
```

## 0. `scripts/setup/` — Environment Bootstrap & Checkout Audit
- `repo_doctor.py`: Dependency-free (`python3 -S`) checkout, Python `compile()`, shell `bash -n`, launcher override, and doc-link audit.
- `bootstrap_ms_swift.sh`: Fetches `ms-swift` `v4.5.2`, applies `ms-swift-v4.5.2-teacher-adapters.patch`, optionally installs observed/framework/SAM2 deps (`--install-deps`), and editable-installs `third_party/ms-swift`.
- `requirements_swift_observed_20261117.txt`, `requirements_sam2_runtime.txt`, `requirements_mopd_overlay_observed_20261116.txt`: Observed dependency pins (keep Swift `numpy==2.1.3` separate from MOPD `numpy==1.26.4`).

## 1. `scripts/data/` — Data & Privilege Preparation
- **Included Base / Relocation / GRPO / OPD Data Helpers**:
  - `relocate_media_paths.py`: Relocate media paths in reviewed JSONLs to a local REFAVS root with fail-closed checks and `.media_manifest.json` provenance.
  - `prepare_refavs_swift_sft.py`: Convert raw REF-AVS splits (`train`, `val`, `test_s`, `test_u`, `test_n`) into `ms-swift` JSONL manifests.
  - `prepare_sft_distributed_padding.py`: Pad SFT manifests for exact multi-GPU DDP rank partitioning.
  - `prepare_sam2_grpo_data.py`: Build prompt-only GRPO training and validation manifests.
  - `degrade_refavs_audio.py` & `prepare_opsd_audio_condition.py`: Generate deterministic noise-degraded or block-muted audio for clean-audio privilege distillation.
  - `prepare_refavs_opsd.py` & `prepare_dual_opsd_audio_class.py`: Build single-privilege and dual audio/class OPSD manifests.
  - `prepare_sam2_privileged_top10_opd_full.py`: Merge audio/visual text-privileged sparse top-10% OPD manifests.
  - `prepare_verified_routed_opd_data.py`, `prepare_student_only_opd_data.py`, `audit_verified_opd_probe_refresh.py`: Build/audit empirical difficulty mixtures and verified routed / student-only online OPD datasets.
- **Historical Synthesis Scripts (not in this trimmed checkout)**:
  - `[historical-only]` `prepare_merged_sft_dataset.py`, `prepare_sam2_sft_initializer.py`, `prepare_refavs_visual_opsd.py`, `prepare_privileged_top10_opd.py`, `prepare_verified_opd_probe_conditions.py`, `prepare_merged_opd_source.py`.
  - `[historical-only]` `generate_refavs_structured_cot.py`, `check_refavs_structured_cot.py`, `generate_non_audio_answer_cot.py`, `generate_refavs_clip_audio_privilege.py`, `generate_refavs_first_mask_visual_privilege.py`, `call_doubao_video.py`, `match_refavs_templates.py`, `refavs_audio_keyword_policy.py`.

## 2. `scripts/sft/` — Supervised Fine-Tuning (SFT)
- `train_qwen2_5_omni_swift_sft.sh`: Qwen2.5-Omni (3B/7B) + SAM (ViT-H) joint Stage-1 SFT.
- `train_qwen2_5_omni_7b_sam2_seg_sft.sh`: Qwen2.5-Omni-7B + SAM2 (Hiera-L Video-Memory) 3-epoch Answer-CoT SFT + automatic evaluation.
- `train_merged_sam2_sft_3epoch_8gpu.sh`: Qwen2.5-Omni-7B + SAM2 3-epoch merged structured CoT SFT.
- `train_qwen2_5_omni_7b_seg_only_1epoch_4gpu.sh`: Stage-2 segmentation-only fine-tuning (`aurora_seg_only.py`).
- `train_qwen3_omni_30b_sft_8gpu.sh` & `train_qwen3_omni_30b_joint_sft_4gpu.sh`: Qwen3-Omni-30B-A3B joint SFT.
- `train_videollama2_swift_sft.sh`: VideoLLaMA2 + SAM joint SFT.

## 3. `scripts/grpo/` — Reinforcement Learning (GRPO)
- `train_grpo_swift.sh` & `sam_iou_server.py`: Original AURORA 3-reward GRPO (`format + class + iou`) with HTTP SAM IoU service.
- `train_qwen2_5_omni_7b_sam2_grpo_8gpu.sh`: SAM2 baseline GRPO with `aurora_class_exact` and `aurora_sam2_mask_iou` rewards.
- `run_sam2_verified_reward_grpo_8gpu.sh`: Verifiable class-gated mask reward GRPO (`0.5 * C + 0.5 * C * IoU`).

## 4. `scripts/opd/` — On-Policy Distillation (OPSD / Routed OPD)
- **Single-Privilege & Feature Distillation OPSD**:
  - `train_opsd.py`, `train_opsd_swift.sh`, `train_opsd_first_mask_swift.sh`, `train_opsd_seg_feature_distill.sh`
  - `run_opsd_audio_mute50_4gpu.sh`, `run_opsd_audio_noise5db_4gpu.sh`, `run_opsd_audio_mute50_segdistill_4gpu.sh`, `run_opsd_audio_noise5db_segdistill_4gpu.sh`
- **Dual Audio/Class Gated OPSD**:
  - `run_opsd_audio_class_gate_4gpu.sh`, `run_opsd_audio_class_gate_8gpu.sh`, `run_opsd_audio_class_gate_reverse_8gpu.sh`
- **Sparse Low-Likelihood & Stage-Anchored OPD**:
  - `run_sam2_privileged_top10_opd_full_8gpu.sh`, `run_sam2_audio_stage_anchor_opd_8gpu.sh`
- **Verified Routed OPD & Online Current-Policy Joint**:
  - `run_sam2_verified_routed_opd_grpo_8gpu.sh`: Cached-probe verified hard-routed OPD + GRPO.
  - `run_current_policy_opd_8gpu.sh`: Unified launcher for Online Current-Policy (`ARM=grpo|opd|joint`).
  - `run_three_teacher_joint_8gpu.sh`: Online Audio/Visual privileged teachers + frozen GRPO anchor teacher (`K=3`).
  - `run_mask_primary_train_8gpu.sh`: Mask-primary reward (`0.2*C + 0.8*IoU`) and Decision-span OPD (`ARM=grpo|decision`).
  - `run_full_response_joint_8gpu.sh`: Full-response token scope Joint OPD ablation.

## 5. `scripts/eval/` — Evaluation Entrypoints
- `test_qwen2_5_omni_refavs.py`: Official Qwen2.5-Omni + SAM/SAM2 evaluator with per-frame IoU/Dice and mask saving.
- `test_qwen3_omni_refavs.py`: Official Qwen3-Omni-30B evaluator.
- `test_videollama2_refavs.py` & `test_videollama2_refavs_sam.py`: VideoLLaMA2 evaluators.
- `eval_omni_two_stage.py` & `eval_omni_two_stage.sh`: Two-stage batch LLM `[SEG]` embedding capture + standalone SAM segmentation.
- `omni_eval_settings.py`: Training/evaluation media and preprocessing parity verifier, plus CLI (`--data-dir`, `--splits`, `--output`) to generate a local `audit.json`.
- `audit_llm_template.py` & `audit_native_generation.py`: Template tokenization and native generation parity diagnostics.

## 6. `scripts/utils/` & `scripts/docs/`
- `utils/export_sam_weights.py`: Export standalone SAM decoder and projector weights.
- `utils/export_qwen3_thinker_merged.py` & `utils/vendor/`: Export merged Qwen3-Omni MoE Thinker weights.
- `docs/`: Detailed method specifications and launch guides (`SAM2_BACKEND_README.md`, `VERIFIED_ROUTED_OPD_README.md`, `CURRENT_POLICY_ONLINE_OPD_README.md`, etc.).
