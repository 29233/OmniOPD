#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
if [[ "${OMNIOPD_LOAD_ENV:-0}" == "1" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi
if [[ -z "${SWIFT_ENV:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    SWIFT_ENV="${VIRTUAL_ENV}"
  elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != "base" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    SWIFT_ENV="${CONDA_PREFIX}"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    SWIFT_ENV="${ROOT_DIR}/.venv"
  fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${SWIFT_ENV:-}" && -x "${SWIFT_ENV}/bin/python" ]]; then
    PYTHON_BIN="${SWIFT_ENV}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
elif [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi
if [[ -z "${SWIFT_BIN:-}" ]]; then
  if [[ -n "${SWIFT_ENV:-}" && -x "${SWIFT_ENV}/bin/swift" ]]; then
    SWIFT_BIN="${SWIFT_ENV}/bin/swift"
  else
    SWIFT_BIN="$(command -v swift || true)"
  fi
elif [[ "${SWIFT_BIN}" != */* ]]; then
  SWIFT_BIN="$(command -v "${SWIFT_BIN}" || true)"
fi
export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
export SWIFT_ENV PYTHON_BIN SWIFT_BIN
export AUDIO_SEED="${AUDIO_SEED:-20261015}"

export RUN_TAG="${RUN_TAG:-opsd_audio_mute50_frozen_r2}"
SOURCE_DIR="${AURORA_SOURCE_DIR:-.}"
export BASE_MODEL="${BASE_MODEL:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}"
export STAGE1_ADAPTERS="${STAGE1_ADAPTERS:-${ROOT_DIR}/outputs/train/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262}"
export TEACHER_MODEL="${TEACHER_MODEL:-${BASE_MODEL}}"
export TEACHER_ADAPTERS="${TEACHER_ADAPTERS:-${STAGE1_ADAPTERS}}"
export DEGRADED_NAME="${DEGRADED_NAME:-audio_opsd_random_mute50.wav}"
_DEFAULT_MUTE_SRC="${ROOT_DIR}/outputs/data/refavs/refavs_full.jsonl"
if [[ ! -f "${_DEFAULT_MUTE_SRC}" && -f "${ROOT_DIR}/outputs/data/refavs/refavs_train.jsonl" ]]; then
  _DEFAULT_MUTE_SRC="${ROOT_DIR}/outputs/data/refavs/refavs_train.jsonl"
fi
SOURCE_TRAIN_JSONL="${SOURCE_TRAIN_JSONL:-${_DEFAULT_MUTE_SRC}}"
SOURCE_VAL_JSONL="${SOURCE_VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val.jsonl}"
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full_opsd_audio_mute50.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_opsd_audio_mute50.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/${RUN_TAG}}"

if [[ "${SKIP_DATA_PREP:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/prepare_opsd_audio_condition.py" \
    --condition mute50 --seed "${AUDIO_SEED}" --output-name "${DEGRADED_NAME}" \
    --metadata "${ROOT_DIR}/outputs/data/audio_degraded/opsd_mute50.json" \
    --manifest "${SOURCE_TRAIN_JSONL}" --manifest "${SOURCE_VAL_JSONL}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/prepare_refavs_opsd.py" \
    --privilege-kind audio-degrade --degraded-audio-name "${DEGRADED_NAME}" \
    --src "${SOURCE_TRAIN_JSONL}" --val-src "${SOURCE_VAL_JSONL}" \
    --dst "${TRAIN_JSONL}" --val-dst "${VAL_JSONL}"
fi

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
