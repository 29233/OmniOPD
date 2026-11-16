#!/usr/bin/env bash
set -euo pipefail

# Qwen3-Omni-30B-A3B AURORA joint SFT on four GPUs.
# The implementation lives in train_qwen3_omni_30b_sft_8gpu.sh; this launcher
# supplies the four-GPU defaults while keeping every setting overridable.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen3_omni_30b_joint_sft_4gpu}"

# The wrapper is intentionally environment-only so the tested joint-training
# implementation, SAM loss, checkpoint format, and smoke-run controls remain
# identical to the 8-GPU entry point.
exec bash "${ROOT_DIR}/scripts/sft/train_qwen3_omni_30b_sft_8gpu.sh" "$@"
