#!/usr/bin/env bash
set -euo pipefail

# Qwen3-Omni-30B-A3B AURORA joint SFT on REF-AVS.
# The thinker LoRA, [SEG] embedding/output, text projector, and SAM mask
# decoder are trained together.  The SAM image/prompt encoders stay frozen.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
if [[ "${OMNIOPD_LOAD_ENV:-0}" == "1" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

# Environment and hardware.
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-base}"
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
if [[ -n "${MS_SWIFT_ROOT:-}" ]]; then
  export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Qwen3-Omni creates only the thinker during SFT. This is required by ZeRO-3
# and avoids loading the unused speech-generation talker/code2wav modules.
export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=0
export FPS="${FPS:-1.0}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-10}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-1.0}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"

# Experiment paths.
export MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/Qwen3-Omni-30B-A3B-Instruct}"
_DEFAULT_Q3_TRAIN="${ROOT_DIR}/outputs/data/refavs/refavs_train.jsonl"
if [[ ! -f "${_DEFAULT_Q3_TRAIN}" && -f "${ROOT_DIR}/outputs/data/refavs/refavs_full.jsonl" ]]; then
  _DEFAULT_Q3_TRAIN="${ROOT_DIR}/outputs/data/refavs/refavs_full.jsonl"
fi
export TRAIN_JSONL="${TRAIN_JSONL:-${_DEFAULT_Q3_TRAIN}}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen3_omni_30b_joint_sft_8gpu}"

# Training schedule. With the defaults, the effective global batch is 8.
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export MAX_STEPS="${MAX_STEPS:--1}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "ms-swift launcher is not executable: ${SWIFT_BIN}" >&2
  exit 1
fi
for required_path in "${MODEL_PATH}" "${TRAIN_JSONL}" "${VAL_JSONL}" "${AURORA_SAM_CHECKPOINT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done
if (( NPROC_PER_NODE < 1 )); then
  echo "NPROC_PER_NODE must be at least 1 (got ${NPROC_PER_NODE})." >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  VISIBLE_GPU_COUNT=$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
  if (( VISIBLE_GPU_COUNT < NPROC_PER_NODE )); then
    echo "Need ${NPROC_PER_NODE} visible GPUs, found ${VISIBLE_GPU_COUNT} (${CUDA_VISIBLE_DEVICES})." >&2
    exit 1
  fi
fi

echo "[env] conda=${CONDA_ENV_NAME} python=${PYTHON_BIN} swift=${SWIFT_BIN}" >&2
"${PYTHON_BIN}" - <<'PY'
import sys
import torch
import transformers
import swift
import deepspeed
import qwen_omni_utils

print(f"[env] executable={sys.executable}", file=sys.stderr)
print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}", file=sys.stderr)
print(f"[env] transformers={transformers.__version__}", file=sys.stderr)
print(f"[env] ms-swift={swift.__version__} ({swift.__file__})", file=sys.stderr)
print(f"[env] deepspeed={deepspeed.__version__}", file=sys.stderr)
print(f"[env] qwen_omni_utils={qwen_omni_utils.__file__}", file=sys.stderr)
PY

# Optional bounded datasets for smoke runs. These copies live in OUTPUT_DIR so
# the source manifests are never changed.
mkdir -p "${OUTPUT_DIR}"
if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  TRAIN_SUBSET="${OUTPUT_DIR}/train_first_${MAX_SAMPLES}.jsonl"
  head -n "${MAX_SAMPLES}" "${TRAIN_JSONL}" > "${TRAIN_SUBSET}"
  TRAIN_JSONL="${TRAIN_SUBSET}"
fi
if [[ "${VAL_MAX_SAMPLES}" != "-1" ]]; then
  VAL_SUBSET="${OUTPUT_DIR}/val_first_${VAL_MAX_SAMPLES}.jsonl"
  head -n "${VAL_MAX_SAMPLES}" "${VAL_JSONL}" > "${VAL_SUBSET}"
  VAL_JSONL="${VAL_SUBSET}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "Resume checkpoint does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

SFT_ARGS=(
  sft
  --model "${MODEL_PATH}"
  --model_type aurora_qwen3_omni
  --template aurora_qwen3_omni
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen3_omni.py"
  --dataset "${TRAIN_JSONL}"
  --val_dataset "${VAL_JSONL}"
  --new_special_tokens '[SEG]'
  --tuner_type lora
  --target_modules all-linear
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder
  --lora_rank 8
  --lora_alpha 32
  --lora_dropout 0.05
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl flash_attn
  --deepspeed zero3
  --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant": true}'
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --max_length "${MAX_LENGTH}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.1
  --logging_steps 1
  --eval_strategy steps
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --dataloader_num_workers 2
  --dataset_num_proc 1
  --split_dataset_ratio 0
  --remove_unused_columns false
  --report_to "${REPORT_TO}"
  --output_dir "${OUTPUT_DIR}"
  "${RESUME_ARGS[@]}"
)

echo "[train] model=${MODEL_PATH}" >&2
echo "[train] train=${TRAIN_JSONL}" >&2
echo "[train] val=${VAL_JSONL}" >&2
echo "[train] output=${OUTPUT_DIR}" >&2
echo "[train] GPUs=${CUDA_VISIBLE_DEVICES} world_size=${NPROC_PER_NODE}" >&2

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run] CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q ' "${CUDA_VISIBLE_DEVICES}" "${NPROC_PER_NODE}"
  printf '%q ' "${SWIFT_BIN}" "${SFT_ARGS[@]}"
  printf '\n'
  exit 0
fi

LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log}"
echo "[train] log=${LOG_FILE}" >&2

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" "${SFT_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
