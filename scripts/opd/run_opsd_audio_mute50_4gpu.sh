#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHON_BIN
export AUDIO_SEED="${AUDIO_SEED:-20261015}"

export RUN_TAG="${RUN_TAG:-opsd_audio_mute50_frozen_r2}"
SOURCE_DIR="${AURORA_SOURCE_DIR:-.}"
export BASE_MODEL="${BASE_MODEL:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}"
export STAGE1_ADAPTERS="${STAGE1_ADAPTERS:-${ROOT_DIR}/outputs/train/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262}"
export TEACHER_MODEL="${TEACHER_MODEL:-${BASE_MODEL}}"
export TEACHER_ADAPTERS="${TEACHER_ADAPTERS:-${STAGE1_ADAPTERS}}"
export DEGRADED_NAME="audio_opsd_random_mute50.wav"
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full_opsd_audio_mute50.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_opsd_audio_mute50.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/${RUN_TAG}}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/prepare_opsd_audio_condition.py" \
  --condition mute50 --seed "${AUDIO_SEED}" --output-name "${DEGRADED_NAME}" \
  --metadata "${ROOT_DIR}/outputs/data/audio_degraded/opsd_mute50.json"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/prepare_refavs_opsd.py" \
  --privilege-kind audio-degrade --degraded-audio-name "${DEGRADED_NAME}" \
  --dst "${TRAIN_JSONL}" --val-dst "${VAL_JSONL}"

export AURORA_OPSD_TEACHER_MODE=frozen
export AURORA_OPSD_REQUIRE_FROZEN_TEACHER=1
# A separate SFT teacher adds model residency; use the existing CPU offload
# path to keep the joint student/teacher/SAM forward within 80 GiB cards.
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-true}"
export OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"
export LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"
export AURORA_OPSD_TEACHER_AUDIO=original
export AURORA_OPSD_TEACHER_MASK_IMAGE=0
export AURORA_OPSD_MASK_LOSS=1
export AURORA_OPSD_SEG_DISTILL_WEIGHT="${AURORA_OPSD_SEG_DISTILL_WEIGHT:-0}"
export AURORA_OPSD_MASK_WEIGHT="${AURORA_OPSD_MASK_WEIGHT:-1.0}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"

exec bash "${ROOT_DIR}/scripts/opd/train_opsd_swift.sh"
