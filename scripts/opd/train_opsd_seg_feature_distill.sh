#!/usr/bin/env bash
# Start the current OPSD + [SEG] feature-distillation experiment.
#
# This entry point uses the implementation in swift_plugin/aurora_opsd.py:
#   - lambda-1: forward-KL/JSD over the student rollout (beta=0)
#   - lambda-2: MSE between the student and privileged-teacher 256-d
#               text_hidden_fcs outputs at the [SEG] position
#   - AURORA mask supervision: 2*BCE + 0.5*Dice
#
# It is intentionally named "feature_distill": the repository does not yet
# implement a separate [SEG]-prediction-logit-only loss.
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
SOURCE_DIR="${AURORA_SOURCE_DIR:-.}"

# Experiment paths. Every value can be overridden before invoking this file.
export RUN_TAG="${RUN_TAG:-opsd_seg_feature_distill_r1}"
export BASE_MODEL="${BASE_MODEL:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}"
export STAGE1_ADAPTERS="${STAGE1_ADAPTERS:-${ROOT_DIR}/outputs/train/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262}"
export TEACHER_MODEL="${TEACHER_MODEL:-${BASE_MODEL}}"
export TEACHER_ADAPTERS="${TEACHER_ADAPTERS:-${STAGE1_ADAPTERS}}"
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full_opsd_first_mask.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_opsd_first_mask.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/${RUN_TAG}}"

# Distillation and AURORA settings.
export AURORA_OPSD_TEACHER_MASK_IMAGE="${AURORA_OPSD_TEACHER_MASK_IMAGE:-1}"
export AURORA_OPSD_SEG_DISTILL_WEIGHT="${AURORA_OPSD_SEG_DISTILL_WEIGHT:-1.0}"
export AURORA_OPSD_MASK_WEIGHT="${AURORA_OPSD_MASK_WEIGHT:-1.0}"
export LMBDA="${LMBDA:-1}"
export BETA="${BETA:-0}"
export TEMPERATURE="${TEMPERATURE:-1}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"

# Hardware and schedule.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"

# Fail early on configuration/path mistakes.
for required_path in \
  "${BASE_MODEL}" \
  "${STAGE1_ADAPTERS}" \
  "${TEACHER_ADAPTERS}" \
  "${TRAIN_JSONL}" \
  "${VAL_JSONL}" \
  "${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[error] required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"
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
export SWIFT_ENV PYTHON_BIN SWIFT_BIN
export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
if [[ -z "${SWIFT_BIN}" || ! -x "${SWIFT_BIN}" ]]; then
  echo "[error] swift executable is not executable: ${SWIFT_BIN:-<empty>}" >&2
  exit 2
fi

echo "[opsd] output: ${OUTPUT_DIR}"
echo "[opsd] train:  ${TRAIN_JSONL}"
echo "[opsd] val:    ${VAL_JSONL}"
echo "[opsd] teacher: ${TEACHER_MODEL} + ${TEACHER_ADAPTERS}"
echo "[opsd] seg feature weight: ${AURORA_OPSD_SEG_DISTILL_WEIGHT}"

exec bash "${ROOT_DIR}/scripts/opd/train_opsd_swift.sh" "$@"
