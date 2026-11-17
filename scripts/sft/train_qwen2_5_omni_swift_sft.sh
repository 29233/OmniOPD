#!/usr/bin/env bash
# Qwen2.5-Omni + SAM (ViT-H) Stage-1 joint SFT entry point on ms-swift.
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
export SWIFT_ENV PYTHON_BIN SWIFT_BIN
export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"

export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# REF-AVS timing audit: ten stored frames across ten-second clips, nominal
# 1 FPS; original per-frame sampling timestamps are not available.
export FPS="${FPS:-1.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export MODEL_PATH="${MODEL_PATH:-models/Qwen2___5-Omni-3B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_3b_sft}"
export DATA_DIR="${DATA_DIR:-${ROOT_DIR}/outputs/data/refavs}"
_DEFAULT_TRAIN_JSONL="${DATA_DIR}/refavs_train.jsonl"
if [[ ! -f "${_DEFAULT_TRAIN_JSONL}" && -f "${DATA_DIR}/refavs_full.jsonl" ]]; then
  _DEFAULT_TRAIN_JSONL="${DATA_DIR}/refavs_full.jsonl"
fi
export AURORA_DATA_JSONL="${AURORA_DATA_JSONL:-${TRAIN_JSONL:-${_DEFAULT_TRAIN_JSONL}}}"
export AURORA_VAL_JSONL="${AURORA_VAL_JSONL:-${VAL_JSONL:-${DATA_DIR}/refavs_val.jsonl}}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=0

for required in "${PYTHON_BIN}" "${SWIFT_BIN}" "${MODEL_PATH}" "${AURORA_DATA_JSONL}" "${AURORA_VAL_JSONL}" "${AURORA_SAM_CHECKPOINT}"; do
  [[ -n "${required}" && -e "${required}" ]] || { echo "Missing Stage-1 SFT requirement: ${required:-<empty>}" >&2; exit 1; }
done

TRAIN_DATASET_SPEC="${AURORA_DATA_JSONL}"
VAL_DATASET_SPEC="${AURORA_VAL_JSONL}"
if (( MAX_SAMPLES > 0 )); then
  TRAIN_DATASET_SPEC="${AURORA_DATA_JSONL}#${MAX_SAMPLES}"
fi
if (( VAL_MAX_SAMPLES > 0 )); then
  VAL_DATASET_SPEC="${AURORA_VAL_JSONL}#${VAL_MAX_SAMPLES}"
fi

mkdir -p "${OUTPUT_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: Stage-1 SFT preflight passed (model=${MODEL_PATH}, train=${TRAIN_DATASET_SPEC}, val=${VAL_DATASET_SPEC})"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" sft \
  --model "${MODEL_PATH}" \
  --model_type aurora_qwen2_5_omni \
  --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --dataset "${TRAIN_DATASET_SPEC}" \
  --val_dataset "${VAL_DATASET_SPEC}" \
  --new_special_tokens '[SEG]' \
  --tuner_type lora \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype float16 \
  --fp16 true \
  --bf16 false \
  --attn_impl sdpa \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --max_length 4096 \
  --max_pixels "${MAX_PIXELS}" \
  --learning_rate 2e-5 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --predict_with_generate true \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --deepspeed "${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json" \
  --callbacks aurora_sam_checkpoint \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
