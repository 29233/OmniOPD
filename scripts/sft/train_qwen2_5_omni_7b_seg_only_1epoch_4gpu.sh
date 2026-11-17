#!/usr/bin/env bash
# Stage-2 Qwen2.5-Omni-7B training: freeze the SFT language/LoRA weights and
# train only text_hidden_fcs + SAM mask_decoder for one epoch, then evaluate
# the resulting checkpoint on REF-AVS test_s and test_u with four GPUs.
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
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
  if [[ -n "${SWIFT_ENV:-}" && -x "${SWIFT_ENV}/bin/torchrun" ]]; then
    TORCHRUN_BIN="${SWIFT_ENV}/bin/torchrun"
  else
    TORCHRUN_BIN="$(command -v torchrun || true)"
  fi
elif [[ "${TORCHRUN_BIN}" != */* ]]; then
  TORCHRUN_BIN="$(command -v "${TORCHRUN_BIN}" || true)"
fi
export SWIFT_ENV SWIFT_BIN PYTHON_BIN TORCHRUN_BIN

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE=4
export MODEL_PATH="${MODEL_PATH:-${MODEL:-models/Qwen2___5-Omni-7B}}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sft_4gpu/v0-20260828-021854/checkpoint-5265}"
export DATA_DIR="${DATA_DIR:-${ROOT_DIR}/outputs/data/refavs}"
_DEFAULT_TRAIN_JSONL="${DATA_DIR}/refavs_train.jsonl"
if [[ ! -f "${_DEFAULT_TRAIN_JSONL}" && -f "${DATA_DIR}/refavs_full.jsonl" ]]; then
  _DEFAULT_TRAIN_JSONL="${DATA_DIR}/refavs_full.jsonl"
fi
export TRAIN_JSONL="${TRAIN_JSONL:-${_DEFAULT_TRAIN_JSONL}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_seg_only_1epoch_4gpu}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT_DIR}/outputs/eval/qwen2_5_omni_7b/seg_only_1epoch_4gpu}"

# checkpoint-5265 predates the corrected 1-fps timeline experiment. Preserve
# its training-time timeline so this run changes only the optimization target.
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-2.0}"
export AURORA_SEGMENTATION_ONLY=1
export AURORA_CE_WEIGHT=0
export AURORA_BCE_WEIGHT="${AURORA_BCE_WEIGHT:-2.0}"
export AURORA_DICE_WEIGHT="${AURORA_DICE_WEIGHT:-0.5}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"
export ENABLE_AUDIO_OUTPUT=0
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"

export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}"
export SAVE_MASKS="${SAVE_MASKS:-0}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

for required in \
  "${SWIFT_BIN}" "${PYTHON_BIN}" "${TORCHRUN_BIN}" \
  "${MODEL_PATH}" "${INIT_CHECKPOINT}/adapter_model.safetensors" \
  "${TRAIN_JSONL}" "${DATA_DIR}/refavs_test_s.jsonl" \
  "${DATA_DIR}/refavs_test_u.jsonl" "${AURORA_SAM_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path does not exist: ${required}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  gpu_count="$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -c \
    'import torch; print(torch.cuda.device_count())')"
  if [[ "${gpu_count}" -ne 4 ]]; then
    echo "Need exactly 4 visible GPUs, found ${gpu_count}: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
fi

echo "Environment:       ${SWIFT_ENV}"
echo "GPUs:              ${CUDA_VISIBLE_DEVICES} (4-rank DDP)"
echo "Base model:        ${MODEL_PATH}"
echo "Initial adapter:   ${INIT_CHECKPOINT}"
echo "Trainable modules: text_hidden_fcs, mask_decoder only"
echo "Objective:         ${AURORA_BCE_WEIGHT}*BCE + ${AURORA_DICE_WEIGHT}*Dice (CE disabled)"
echo "Schedule:          1 epoch, lr=${LEARNING_RATE}, cosine"
echo "Train output:      ${OUTPUT_DIR}"
echo "Eval splits:       test_s, test_u"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: paths and configuration validated; training not started."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_ROOT}"
TRAIN_LOG="${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log"

TRAIN_DATASET="${TRAIN_JSONL}"
if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  TRAIN_DATASET="${OUTPUT_DIR}/train_first_${MAX_SAMPLES}.jsonl"
  "${PYTHON_BIN}" - "${TRAIN_JSONL}" "${TRAIN_DATASET}" "${MAX_SAMPLES}" <<'PY'
import sys

source, destination, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source, encoding="utf-8") as src, open(destination, "w", encoding="utf-8") as dst:
    for index, line in enumerate(src):
        if index >= limit:
            break
        dst.write(line)
PY
fi

echo "Training log: ${TRAIN_LOG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE=4 \
"${SWIFT_BIN}" sft \
  --model "${MODEL_PATH}" \
  --adapters "${INIT_CHECKPOINT}" \
  --model_type aurora_qwen2_5_omni \
  --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/aurora_seg_only.py" \
  --dataset "${TRAIN_DATASET}" \
  --new_special_tokens '[SEG]' \
  --tuner_type lora \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype float16 \
  --attn_impl sdpa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num_train_epochs 1 \
  --max_steps -1 \
  --max_length 4096 \
  --max_pixels "${MAX_PIXELS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --gradient_checkpointing false \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --eval_strategy no \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --deepspeed "${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json" \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${TRAIN_LOG}"

TRAINED_CHECKPOINT="$(${PYTHON_BIN} - "${OUTPUT_DIR}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [
    p.parent for p in root.glob("**/checkpoint-*/adapter_model.safetensors")
]
if not candidates:
    raise SystemExit(f"No completed checkpoint found under {root}")
print(max(candidates, key=lambda p: os.path.getmtime(p / "adapter_model.safetensors")))
PY
)"

RUN_NAME="$(basename "$(dirname "${TRAINED_CHECKPOINT}")")_$(basename "${TRAINED_CHECKPOINT}")"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_OUTPUT_ROOT}/${RUN_NAME}}"
mkdir -p "${EVAL_OUTPUT_DIR}"

EVAL_ARGS=(
  "${ROOT_DIR}/scripts/eval/test_qwen2_5_omni_refavs.py"
  --model "${MODEL_PATH}"
  --checkpoint "${TRAINED_CHECKPOINT}"
  --data-dir "${DATA_DIR}"
  --output-dir "${EVAL_OUTPUT_DIR}"
  --splits test_s test_u
  --max-samples "${EVAL_MAX_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
if [[ "${SAVE_MASKS}" == "0" ]]; then
  EVAL_ARGS+=(--no-save-masks)
fi

echo "Training complete: ${TRAINED_CHECKPOINT}"
echo "Starting test_s/test_u evaluation: ${EVAL_OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${TORCHRUN_BIN}" --standalone --nproc_per_node=4 "${EVAL_ARGS[@]}"

echo "Evaluation complete: ${EVAL_OUTPUT_DIR}/summary.json"
