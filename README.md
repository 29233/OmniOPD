# OmniOPD: On-Policy Distillation, Reinforcement Learning, and Video-Memory Segmentation for Reference Audio-Visual Segmentation

**OmniOPD** is a multimodal training and evaluation framework for **Reference Audio-Visual Segmentation (Ref-AVS)**. Built on top of **Omni Multimodal Large Language Models** (`Qwen2.5-Omni-3B/7B`, `Qwen3-Omni-30B-A3B`) and **SAM / SAM2 (Hiera-L Video-Memory)** segmentation heads, this repository provides a unified implementation of:

1. **Structured Chain-of-Thought (CoT) Joint Supervised Fine-Tuning (SFT)** with pixel-level mask supervision (`CE + 2.0 * BCE + 0.5 * Dice`) and sequential SAM2 video-memory propagation across frames;
2. **Reinforcement Learning via GRPO** with verifiable category-exact and 10-frame SAM2 mask IoU rewards;
3. **Privileged On-Policy Self-Distillation (OPSD) & Verified Routed OPD** on `ms-swift`, leveraging training-time privileged views (ground-truth category, clean audio vs. degraded audio, first-frame visual/mask hints, and clip-level acoustic evidence) with verifier-licensed token-level routing;
4. **MOPD-Router-Omni** (`third_party/MOPD-Router-Omni`), a native `verl` FSDP migration of multi-teacher token-level routing (`ExpertAlign`, `Calibrated Entropy`, and `Accessible Novelty`) for Omni language reasoning.

---

## Repository Structure

```text
OmniOPD/
├── swift_plugin/                  # ms-swift 4.x model registrations, trainers, rewards, and losses
│   ├── aurora_qwen2_5_omni.py     # Qwen2.5-Omni (3B/7B) + SAM/SAM2 model & template registration
│   ├── aurora_qwen3_omni.py       # Qwen3-Omni-30B-A3B MoE + SAM model & template registration
│   ├── aurora_videollama2.py      # VideoLLaMA2-7B-AV + SAM registration
│   ├── segmentation.py            # SAM (ViT-H) frame-independent segmentation head & losses
│   ├── sam2_segmentation.py       # SAM2 (Hiera-L) sequential video-memory segmentation backend
│   ├── sam_checkpoint.py          # Callback for saving standalone SAM/SAM2 decoder sidecars
│   ├── aurora_seg_only.py         # Stage-2 segmentation-only SFT trainer (freezes language backbone)
│   ├── seg_distillation.py        # 256-d [SEG] prompt embedding distillation (MSE / Cosine)
│   ├── aurora_opsd.py             # Single-privilege GKD OPSD trainer + SAM mask supervision
│   ├── aurora_dual_opsd.py        # Dual Audio/Class positive-delta gated OPSD trainer
│   ├── dual_opsd_loss.py          # Base + positive-delta gated full-vocabulary KL loss
│   ├── aurora_privileged_top10_opd.py # Sparse low-likelihood (bottom-10% ± 1) multi-privilege OPD
│   ├── privileged_top10_loss.py   # Low-likelihood token seed selection and chunked KL
│   ├── aurora_audio_stage_anchor_opd.py # Stage-aware audio OPD with unprivileged Decision/[SEG] anchor
│   ├── aurora_grpo.py             # Original 3-reward GRPO (format + class + SAM IoU) + mask loss
│   ├── aurora_sam2_grpo.py        # SAM2 baseline GRPO with frozen SAM2 video-memory reward decoder
│   ├── sam2_grpo_rewards.py       # Canonical target category parser and 10-frame mIoU reward helpers
│   ├── aurora_verified_reward_grpo.py # Class-gated mask reward GRPO (0.5 * C + 0.5 * C * IoU)
│   ├── verified_opd_routing.py    # Category-first, IoU-second teacher licensing & hard token routing
│   ├── aurora_verified_routed_opd.py  # Verified routed OPD + GRPO joint trainer (cached probes)
│   ├── aurora_online_verified_opd.py  # Online current-policy privileged Teacher self-distillation
│   ├── aurora_online_three_teacher_joint.py # K=3 routing: online Audio/Visual + frozen GRPO anchor
│   ├── aurora_mask_primary.py     # Mask-primary reward (0.2*C + 0.8*IoU) & Decision-span OPD
│   └── aurora_full_response_opd.py # Full-response token scope Joint OPD ablation
├── scripts/
│   ├── data/                      # Dataset converters, CoT synthesis, audio degradation & difficulty mixtures
│   ├── sft/                       # Stage-1, Stage-2, and SAM2 joint SFT launch scripts
│   ├── grpo/                      # GRPO reinforcement learning launchers and SAM IoU server
│   ├── opd/                       # OPSD, sparse OPD, and Verified Routed / Online OPD launchers
│   ├── eval/                      # Official Ref-AVS evaluators and two-stage batch evaluation pipeline
│   ├── utils/                     # SAM and merged Thinker checkpoint export utilities
│   └── docs/                      # Detailed method design notes and recipe guides
├── configs/
│   ├── deepspeed_zero2_*.json     # DeepSpeed ZeRO-2 configurations (FP16 / BF16)
│   └── prompts/                   # Settled Structured CoT, Answer-Conditioned CoT & privilege prompts
└── third_party/
    └── MOPD-Router-Omni/          # Native verl FSDP multi-teacher router migration for Qwen2.5-Omni Thinker
```

---

## Core Algorithms

### 1. Multimodal Joint SFT with SAM / SAM2 Video-Memory Backend
Given a 10-frame video clip $V$ (nominal 1 FPS over 10s), audio waveform $A$, and referring expression $r$, the model generates a structured 6-stage Chain-of-Thought (`Reference analysis` $\to$ `Audio evidence` $\to$ `Visual evidence` $\to$ `Cross-modal reasoning` $\to$ `Decision` $\to$ `Segmentation: It is [SEG].`).
- The last-layer hidden state of the first `[SEG]` token is projected via `text_hidden_fcs` to a 256-d prompt embedding.
- **SAM2 Video-Memory Backend (`AURORA_SEG_BACKEND=sam2`)**: Decodes all 10 frames sequentially—Frame 0 initializes the mask, object pointer, and memory features, while Frames $1\dots 9$ attend to conditioning and historical mask memories via `memory_attention` and `memory_encoder`.
- **Joint Objective**:
  $$\mathcal{L}_{\text{SFT}} = \mathcal{L}_{\text{CE}} + 2.0\,\mathcal{L}_{\text{BCE}}(M_{\text{pred}}, M^*) + 0.5\,\mathcal{L}_{\text{Dice}}(M_{\text{pred}}, M^*)$$

### 2. Verifiable-Reward GRPO
Using a frozen trained projector and SAM2 video-memory head as a deterministic reward decoder, the policy samples $G=8$ trajectories per prompt and optimizes group-relative advantages under:
- **Class-Gated Mask Reward** (`aurora_verified_reward_grpo.py`):
  $$R = 0.5 \cdot C_{\text{exact}} + 0.5 \cdot \big(C_{\text{exact}} \cdot \text{mIoU}_{10}\big)$$
  Gating mask IoU by exact target category match ($C_{\text{exact}} \in \{0, 1\}$) prevents reward hacking where a trajectory names the wrong object class while its `[SEG]` mask overlaps the foreground.
- **Mask-Primary Reward** (`aurora_mask_primary.py`):
  $$R = \mathbb{I}(\text{valid\_SEG}) \cdot \big(0.2 \cdot C_{\text{exact}} + 0.8 \cdot \text{mIoU}_{10}\big)$$

### 3. On-Policy Self-Distillation (OPSD) & Verified Routed OPD
Instead of distilling on static off-policy teacher trajectories, the Student samples on-policy rollouts $y \sim p_{\theta}(\cdot \mid x_S)$, and privileged Teachers score the exact same token sequence $y$ under privileged views $x_k^+$:
- **Single & Dual Audio/Class Gated OPSD** (`aurora_opsd.py`, `aurora_dual_opsd.py`): Supports category privilege, clean-audio privilege (Student receives `mute50` or `snr5db` degraded audio; Teacher receives clean audio), first-frame mask privilege, and `[SEG]` 256-d embedding alignment ($\mathcal{L}_{\text{SEG-KD}}$).
- **Low-Likelihood Token Localization** (`aurora_privileged_top10_opd.py`, `aurora_audio_stage_anchor_opd.py`): Restricts distillation to the bottom-$10\%$ lowest Teacher likelihood tokens (dilated by $\pm 1$) within modality-relevant CoT stages.
- **Outcome-Verified Teacher Licensing & Hard Routing** (`verified_opd_routing.py`, `aurora_online_verified_opd.py`):
  1. **Online Current-Policy Teachers**: At each step, the current Student policy (with LoRA enabled, under `torch.no_grad()`) generates fresh Audio-privileged and Visual-privileged trajectories and decodes their 10-frame SAM2 masks.
  2. **Licensing**: Teacher $k$ is licensed only if it rescues a Student category error ($C_k=1, C_S=0$) or improves 10-frame mIoU by more than $\mu_{\text{iou}}=0.01$ when both predict the correct category ($C_k=C_S=1$).
  3. **Signed Reverse-KL Policy Advantage**: Active tokens are hard-routed to the highest-scoring licensed Teacher $k^*$, adding the native signed log-ratio directly to the GRPO token advantage:
     $$A_{t}^{\text{total}} = A_{t}^{\text{GRPO}} + \lambda_{\text{KD}} \cdot w_{k^*, t}^{\text{verified}} \cdot \big(\log p_{T_{k^*}}(y_t) - \log p_{S_{\text{old}}}(y_t)\big)$$

### 4. Native Multi-Teacher Routing (`third_party/MOPD-Router-Omni`)
A standalone migration of `MOPD-Router` on `verl` for full-decoder Qwen2.5-Omni `Thinker` reasoning (`only_reverse_kl_advantages=True`), supporting three token-level routing rules across independent frozen specialist teachers:
- **`delta` (ExpertAlign)**: Routes by specialist log-prob shift relative to a shared pre-specialization base model;
- **`entropy` (Calibrated Entropy)**: Routes by batch-calibrated teacher predictive entropy;
- **`accessible_novelty` (Accessible Novelty)**: Scores each teacher on the shared Student/Teacher Top-$K$ support using reachability $\sqrt{M_S^{\text{overlap}} M_{T_k}^{\text{overlap}}}$ multiplied by saturated Jensen-Shannon novelty $1 - \exp(-\text{JS}/\tau_{\text{JS}})$.

---

## Installation & Environment

1. **For `ms-swift` (SFT / GRPO / OPSD / Verified Routed OPD)**:
   ```bash
   pip install -r requirements_ms_swift.txt
   ```
   For the SAM2 video-memory backend (`AURORA_SEG_BACKEND=sam2`), place the upstream `sam2` checkout under `third_party/sam2` and download `sam2_hiera_large.pt` (or `sam_vit_h_4b8939.pth` for SAM v1) into `models/`.

2. **For `third_party/MOPD-Router-Omni`**:
   See [third_party/MOPD-Router-Omni/OMNI_MIGRATION.md](third_party/MOPD-Router-Omni/OMNI_MIGRATION.md) and run:
   ```bash
   cd third_party/MOPD-Router-Omni
   bash bootstrap_omni_env.sh
   ```

---

## Quick Start

All scripts are organized by functional stage under `scripts/` (see [scripts/README.md](scripts/README.md) for a full index).

### 1. Data & Privilege Preparation (`scripts/data/`)
```bash
# Convert raw REF-AVS splits to ms-swift JSONL format
python scripts/data/prepare_refavs_swift_sft.py --split train --output outputs/data/refavs/refavs_train.jsonl
python scripts/data/prepare_refavs_swift_sft.py --split test_s --output outputs/data/refavs/refavs_test_s.jsonl
python scripts/data/prepare_refavs_swift_sft.py --split test_u --output outputs/data/refavs/refavs_test_u.jsonl

# Initialize SAM2 video-memory weights with a trained Stage-1 adapter
python scripts/data/prepare_sam2_sft_initializer.py \
  --source /path/to/stage1_checkpoint \
  --sam2-checkpoint models/sam2_hiera_large.pt \
  --output outputs/train/sam2_sft_initializer
```

### 2. Supervised Fine-Tuning (`scripts/sft/`)
```bash
# Qwen2.5-Omni-7B + SAM2 3-epoch joint SFT (8 GPUs)
MODEL=models/Qwen2___5-Omni-7B \
SAM2_CHECKPOINT=models/sam2_hiera_large.pt \
bash scripts/sft/train_merged_sam2_sft_3epoch_8gpu.sh

# Qwen2.5-Omni + SAM (ViT-H) Stage-1 joint SFT
MODEL_PATH=models/Qwen2___5-Omni-3B \
AURORA_SAM_CHECKPOINT=models/sam_vit_h_4b8939.pth \
bash scripts/sft/train_qwen2_5_omni_swift_sft.sh
```

### 3. GRPO & Online Verified Routed OPD (`scripts/grpo/`, `scripts/opd/`)
```bash
# SAM2 Baseline GRPO (8 GPUs)
bash scripts/grpo/train_qwen2_5_omni_7b_sam2_grpo_8gpu.sh

# Online Current-Policy Verified OPD / GRPO / Joint (ARM = grpo | opd | joint)
ARM=joint \
SFT_CHECKPOINT=/path/to/sam2_sft_checkpoint \
DATA_ROOT=/path/to/student_only_opd_data \
OUTPUT_DIR=outputs/train/online_joint_8gpu \
bash scripts/opd/run_current_policy_opd_8gpu.sh

# Mask-Primary GRPO or Decision-Span OPD (ARM = grpo | decision)
ARM=decision \
SFT_CHECKPOINT=/path/to/sam2_sft_checkpoint \
DATA_ROOT=/path/to/student_only_opd_data \
OUTPUT_DIR=outputs/train/mask_primary_decision_8gpu \
bash scripts/opd/run_mask_primary_train_8gpu.sh

# Dual Audio/Class Gated OPSD (8 GPUs)
bash scripts/opd/run_opsd_audio_class_gate_8gpu.sh
```

### 4. Evaluation (`scripts/eval/`)
```bash
# Official Qwen2.5-Omni + SAM/SAM2 evaluation on test_s and test_u (saves all predicted frame masks & per-frame IoU/Dice)
torchrun --standalone --nproc_per_node=8 scripts/eval/test_qwen2_5_omni_refavs.py \
  --model models/Qwen2___5-Omni-7B \
  --checkpoint /path/to/checkpoint \
  --data-dir /path/to/eval_data \
  --output-dir outputs/eval/run_official \
  --splits test_s test_u \
  --run-sam --save-masks

# Two-stage batch evaluation (Stage 1: LLM generation & [SEG] embedding capture -> Stage 2: standalone SAM decoding)
FAMILY=qwen2_5 \
CHECKPOINT=/path/to/checkpoint \
OUTPUT_DIR=outputs/eval/two_stage_run \
GPU_GROUPS='0;1;2;3' BATCH_SIZE=2 \
bash scripts/eval/eval_omni_two_stage.sh
```

---

## Documentation Index

- [MS_SWIFT_SFT.md](MS_SWIFT_SFT.md): Stage-1 SFT training objective, prompt contract, and mixed-precision setup.
- [OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md): Standard layout for `outputs/data`, `outputs/train`, and `outputs/eval`.
- [scripts/README.md](scripts/README.md): Complete index of scripts under `scripts/{data,sft,grpo,opd,eval,utils,docs}`.
- [scripts/docs/SAM2_BACKEND_README.md](scripts/docs/SAM2_BACKEND_README.md): SAM2 Hiera-L sequential video-memory integration.
- [scripts/docs/SAM2_GRPO_README.md](scripts/docs/SAM2_GRPO_README.md): SAM2 frozen-decoder GRPO specification.
- [scripts/docs/VERIFIED_ROUTED_OPD_README.md](scripts/docs/VERIFIED_ROUTED_OPD_README.md): Verifier-licensed hard-routed OPD + GRPO joint training.
- [scripts/docs/CURRENT_POLICY_ONLINE_OPD_README.md](scripts/docs/CURRENT_POLICY_ONLINE_OPD_README.md): Student-only difficulty stratification & online current-policy privileged Teachers.
- [scripts/docs/MASK_PRIMARY_DECISION_OPD_README.md](scripts/docs/MASK_PRIMARY_DECISION_OPD_README.md): Mask-primary reward and Decision-category span OPD.
- [scripts/docs/OPSD_AUDIO_CLASS_GATE_README.md](scripts/docs/OPSD_AUDIO_CLASS_GATE_README.md): Dual Audio/Class positive-delta gated OPSD.
- [scripts/docs/EVAL_OMNI_TWO_STAGE_README.md](scripts/docs/EVAL_OMNI_TWO_STAGE_README.md): Two-stage Omni evaluation pipeline.
- [third_party/MOPD-Router-Omni/OMNI_MIGRATION.md](third_party/MOPD-Router-Omni/OMNI_MIGRATION.md): Native `verl` FSDP multi-teacher routing migration for Qwen2.5-Omni Thinker.

---

## Acknowledgements

This project builds upon [AURORA](https://arxiv.org/pdf/2508.02149), [ms-swift](https://github.com/modelscope/ms-swift), [SAM 2](https://github.com/facebookresearch/sam2), and [verl](https://github.com/volcengine/verl).
