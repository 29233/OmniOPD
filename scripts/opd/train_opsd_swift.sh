#!/usr/bin/env bash
# AURORA OPSD training entry point (ms-swift 4.x GKD with teacher_prompt).
#
# Runs in the `swift46` conda env (ms-swift 4.5.2). Requires:
#   - dataset with a teacher_prompt column (scripts/data/prepare_refavs_opsd.py)
#   - swift_plugin/aurora_opsd.py (GKD trainer + AURORA mask supervision)
#
# Key configuration (see load.md analysis for rationale):
#   --lmbda 1        pure on-policy: every row uses the student's own rollout
#   --beta 0         forward KL (OPSD paper's ablation: reverse KL/JSD fail)
#   teacher          AURORA_OPSD_TEACHER_MODE=frozen (default): independent
#                    model + adapter or a Top-K HTTP service. current: reuse
#                    the current student + active adapter with no gradients.
#   mask loss        AURORA_OPSD_MASK_WEIGHT * (2*BCE + 0.5*Dice) added to the
#                    JSD loss by AuroraGKDTrainer (see swift_plugin/aurora_opsd.py)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# TRL reads its UTF-8 model-card template through locale-dependent
# Path.read_text() whenever a checkpoint is saved. Keep direct invocations
# safe even when the platform launches workers with an ASCII locale.
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
SOURCE_DIR="${AURORA_SOURCE_DIR:-.}"

# Base model: the SFT Stage-1 checkpoint directory (LoRA adapter + SAM sidecar).
# The trainer loads it via --adapters so LoRA/embed/lm_head/text_hidden_fcs/
# mask_decoder all resume from Stage-1.
export BASE_MODEL="${BASE_MODEL:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}"
export STAGE1_ADAPTERS="${STAGE1_ADAPTERS:-${ROOT_DIR}/outputs/train/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262}"
export TEACHER_MODEL="${TEACHER_MODEL:-${BASE_MODEL}}"
export TEACHER_ADAPTERS="${TEACHER_ADAPTERS:-${STAGE1_ADAPTERS}}"
export TEACHER_MODEL_SERVER="${TEACHER_MODEL_SERVER:-}"
export GKD_LOGITS_TOPK="${GKD_LOGITS_TOPK:-64}"
export AURORA_OPSD_TEACHER_MODE="${AURORA_OPSD_TEACHER_MODE:-frozen}"
case "${AURORA_OPSD_TEACHER_MODE}" in
  current)
    if [[ -n "${TEACHER_MODEL_SERVER}" ]]; then
      echo 'current OPSD requires the current local student, not a teacher server' >&2
      exit 1
    fi
    ;;
  frozen) ;;
  *) echo 'AURORA_OPSD_TEACHER_MODE must be frozen or current' >&2; exit 1 ;;
esac

if [[ "${AURORA_OPSD_REQUIRE_FROZEN_TEACHER:-0}" == "1" ]]; then
  if [[ "${AURORA_OPSD_TEACHER_MODE}" != "frozen" || -n "${TEACHER_MODEL_SERVER}" ]]; then
    echo 'This experiment requires a local frozen SFT teacher with teacher_adapters.' >&2
    exit 1
  fi
  for required in "${BASE_MODEL}" "${TEACHER_MODEL}" \
    "${STAGE1_ADAPTERS}/adapter_model.safetensors" \
    "${TEACHER_ADAPTERS}/adapter_model.safetensors" \
    "${TEACHER_ADAPTERS}/adapter_config.json"; do
    if [[ ! -e "${required}" ]]; then
      echo "Required model/adapter path missing: ${required}" >&2
      exit 1
    fi
  done
fi

# ms-swift treats an identical --model/--teacher_model string as implicit
# self-distillation and then calls disable_adapter() for the teacher. That
# silently ignores TEACHER_ADAPTERS, which is wrong for class-privileged OPSD:
# the teacher must use the frozen Stage-1/SFT adapter. Keep the same physical
# base weights but expose them through a distinct path so swift loads a real
# local teacher and applies --teacher_adapters.
if [[ "${AURORA_OPSD_TEACHER_MODE}" == "frozen" && -z "${TEACHER_MODEL_SERVER}" && "${TEACHER_MODEL}" == "${BASE_MODEL}" ]]; then
  TEACHER_ALIAS_ROOT="${ROOT_DIR}/outputs/.opsd_teacher_model_alias"
  TEACHER_ALIAS="${TEACHER_ALIAS_ROOT}/$(basename "${BASE_MODEL}")"
  mkdir -p "${TEACHER_ALIAS_ROOT}"
  if [[ -e "${TEACHER_ALIAS}" && ! -L "${TEACHER_ALIAS}" ]]; then
    echo "Refusing to replace non-symlink teacher alias: ${TEACHER_ALIAS}" >&2
    exit 1
  fi
  ln -sfn "${BASE_MODEL}" "${TEACHER_ALIAS}"
  export TEACHER_MODEL="${TEACHER_ALIAS}"
fi
if [[ "${AURORA_OPSD_TEACHER_MODE}" == "frozen" && -z "${TEACHER_MODEL_SERVER}" && "${TEACHER_MODEL}" == "${BASE_MODEL}" ]]; then
  echo "Teacher and student model paths are still identical; refusing self-distillation." >&2
  exit 1
fi
echo "[opsd] student base: ${BASE_MODEL}" >&2
if [[ "${AURORA_OPSD_TEACHER_MODE}" == "current" ]]; then
  echo '[opsd] teacher: current student parameters + active adapter; no_grad privileged view' >&2
elif [[ -n "${TEACHER_MODEL_SERVER}" ]]; then
  echo "[opsd] teacher server: ${TEACHER_MODEL_SERVER}" >&2
  echo "[opsd] teacher top-k: ${GKD_LOGITS_TOPK}" >&2
else
  echo "[opsd] teacher base: ${TEACHER_MODEL}" >&2
  echo "[opsd] teacher adapter: ${TEACHER_ADAPTERS}" >&2
fi
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/opsd_class_r1}"

# OPSD dataset (teacher_prompt column injected by prepare_refavs_opsd.py)
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full_opsd.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_opsd.jsonl}"

# OPSD / GKD hyperparameters
export LMBDA="${LMBDA:-1}"
export BETA="${BETA:-0}"
export TEMPERATURE="${TEMPERATURE:-1}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
export AURORA_OPSD_MASK_WEIGHT="${AURORA_OPSD_MASK_WEIGHT:-1.0}"
# Optional teacher privilege: append mask_paths[0] as a teacher-only image.
# Student inputs and the 10-frame SAM supervision remain unchanged.
export AURORA_OPSD_TEACHER_MASK_IMAGE="${AURORA_OPSD_TEACHER_MASK_IMAGE:-0}"
# Audio-degrade privilege: "original" makes the teacher hear the original
# audio (dataset's teacher_audio_paths column) while the student hears the
# degraded copy written into the row's audios. Empty (default) disables.
export AURORA_OPSD_TEACHER_AUDIO="${AURORA_OPSD_TEACHER_AUDIO:-}"
# λ2: [SEG] prompt (text_hidden_fcs output) distillation from the privileged
# teacher. 0 disables; requires the mask branch (default on) and a local
# teacher. Watch aurora/seg_distill_cos during the run — it should climb
# toward 1 as the student absorbs the privileged representation.
export AURORA_OPSD_SEG_DISTILL_WEIGHT="${AURORA_OPSD_SEG_DISTILL_WEIGHT:-0}"
export AURORA_OPSD_SEG_DISTILL_LOSS="${AURORA_OPSD_SEG_DISTILL_LOSS:-mse}"
echo "[opsd] mask prompt distillation: ${AURORA_OPSD_SEG_DISTILL_LOSS}, weight=${AURORA_OPSD_SEG_DISTILL_WEIGHT}" >&2
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-false}"
export OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"

# Training schedule. The OPSD paper converges in ~100 steps on math; we budget
# 10x plus headroom (full dataset, 1 epoch) and rely on early checkpoints.
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export SEED="${SEED:-42}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"

# Env / infrastructure
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# A100 supports bf16 — preferred: avoids the fp32/fp16 mixed-dtype crash in
# transformers 5.x Qwen2.5-Omni generate (audio tower runs fp32; on fp16
# models q_proj then receives fp32 activations). Keep float16 for V100.
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export ENABLE_AUDIO_OUTPUT=0
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export LOG_COMPLETIONS="${LOG_COMPLETIONS:-false}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export SWIFT_BIN="${SWIFT_BIN:-swift}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# SAM2 uses the pinned upstream implementation and vendored pure-Python deps.
if [[ "${AURORA_SEG_BACKEND:-sam}" == "sam2" ]]; then
  export PYTHONPATH="${ROOT_DIR}/third_party/sam2_runtime_deps:${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
  MODULES_TO_SAVE=(embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample)
else
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
  MODULES_TO_SAVE=(embed_tokens lm_head text_hidden_fcs mask_decoder)
fi

# --- optional smoke-mode truncation --------------------------------------
if [[ "${MAX_SAMPLES}" != "-1" ]]; then
  "${PYTHON_BIN}" - "${TRAIN_JSONL}" "${MAX_SAMPLES}" << 'EOF'
import json, sys
src, cap = sys.argv[1], int(sys.argv[2])
rows = []
with open(src) as f:
    for line in f:
        if len(rows) >= cap: break
        rows.append(line)
with open(src + '.smoke.jsonl', 'w') as f:
    f.writelines(rows)
print(f"smoke train set: {len(rows)} rows -> {src}.smoke.jsonl")
EOF
  export TRAIN_JSONL="${TRAIN_JSONL}.smoke.jsonl"
fi

if [[ "${VAL_MAX_SAMPLES}" != "-1" ]]; then
  "${PYTHON_BIN}" - "${VAL_JSONL}" "${VAL_MAX_SAMPLES}" << 'EOF'
import json, sys
src, cap = sys.argv[1], int(sys.argv[2])
rows = []
with open(src) as f:
    for line in f:
        if len(rows) >= cap: break
        rows.append(line)
with open(src + '.smoke.jsonl', 'w') as f:
    f.writelines(rows)
print(f"smoke val set: {len(rows)} rows -> {src}.smoke.jsonl")
EOF
  export VAL_JSONL="${VAL_JSONL}.smoke.jsonl"
fi

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${AURORA_OPSD_TEACHER_MODE}" == "current" ]]; then
  # Omit both teacher_model and teacher_adapters. Swift then reuses the
  # active student including its adapter; same-model explicit teacher args
  # may instead load a frozen model or disable the active adapter.
  TEACHER_ARGS=()
elif [[ -n "${TEACHER_MODEL_SERVER}" ]]; then
  TEACHER_ARGS=(
    --teacher_model_server "${TEACHER_MODEL_SERVER}"
    --gkd_logits_topk "${GKD_LOGITS_TOPK}"
  )
else
  TEACHER_ARGS=(
    --teacher_model "${TEACHER_MODEL}"
    --teacher_adapters "${TEACHER_ADAPTERS}"
    --teacher_model_type aurora_qwen2_5_omni
    --offload_teacher_model "${OFFLOAD_TEACHER_MODEL}"
  )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" rlhf \
  --rlhf_type gkd \
  --model "${BASE_MODEL}" \
  "${TEACHER_ARGS[@]}" \
  --offload_optimizer "${OFFLOAD_OPTIMIZER}" \
  --adapters "${STAGE1_ADAPTERS}" \
  --model_type aurora_qwen2_5_omni \
  --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${AURORA_OPSD_PLUGIN:-${ROOT_DIR}/swift_plugin/aurora_opsd.py}" \
    "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}" \
  --new_special_tokens '[SEG]' \
  --lmbda "${LMBDA}" \
  --beta "${BETA}" \
  --temperature "${TEMPERATURE}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_pixels "${MAX_PIXELS}" \
  --tuner_type lora \
  --ddp_static_graph true \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save "${MODULES_TO_SAVE[@]}" \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_impl sdpa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps "${MAX_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_length "${MAX_LENGTH}" \
  --learning_rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --logging_steps 1 \
  --log_completions "${LOG_COMPLETIONS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --callbacks aurora_sam_checkpoint \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}" \
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
  2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
