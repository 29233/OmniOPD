#!/usr/bin/env bash
# AURORA GRPO training entry point (ms-swift 4.x GRPO with the original
# paper's reward design, see swift_plugin/aurora_grpo.py).
#
# Reward set (== model/LISA_grpo.py): aurora_format + aurora_iou + aurora_class
#   format  0/1   completion ends with "It is [SEG]."
#   class   0..2  second-to-last sentence "the target is X" (+1 if X == GT)
#   iou     0..1  [SEG] hidden -> SAM -> IoU vs GT mask
# Plus the supervised mask loss (2*BCE + 0.5*Dice) attached by
# AuroraGRPOTrainer.compute_loss, mirroring loss = loss_r1 + mask_loss.
#
# Original hyperparameters: num_generations=3, T=1.0, top_p=0.9,
# max_new_tokens=512, beta=0.04 (KL).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${AURORA_SOURCE_DIR:-.}"

export BASE_MODEL="${BASE_MODEL:-${SOURCE_DIR}/models/Qwen2___5-Omni-3B}"
# GRPO starts from the Stage-1 SFT checkpoint (LoRA + SAM sidecar).
export STAGE1_ADAPTERS="${STAGE1_ADAPTERS:-${ROOT_DIR}/outputs/train/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/grpo_r1}"

# GRPO dataset: plain Stage-1 rows (no teacher_prompt needed - rewards are
# external signals). Reuse the OPSD dataset as-is; the extra column is inert.
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full_opsd.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_opsd.jsonl}"

# Env / infrastructure
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export AURORA_SAM_CHECKPOINT="${AURORA_SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
# Audited REF-AVS clips have ten stored frames across ten seconds, nominal
# 1 FPS (per-frame sampling timestamps unavailable).
export FPS="${FPS:-1.0}" AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
# --- remote SAM IoU service (scripts/grpo/sam_iou_server.py) --------------------
# Comma-separated URLs, one per service replica; the IoU ORM tries them in
# order. REQUIRED: with the remote-service refactor the in-process SAM IoU
# path was removed — without this var every iou reward fails (scores 0.0).
# Mask supervision is off by default: SAM lives on the service node, no
# gradients can flow back (set AURORA_GRPO_MASK_LOSS=0 explicitly to keep
# the intent visible).
export AURORA_SAM_IOU_URLS="${AURORA_SAM_IOU_URLS:-http://localhost:8877}"
export AURORA_GRPO_MASK_LOSS="${AURORA_GRPO_MASK_LOSS:-0}"
export AURORA_GRPO_REWARD_FAIL_OPEN="${AURORA_GRPO_REWARD_FAIL_OPEN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" - <<'PY'
import json, os, urllib.request
urls = [s.strip().rstrip("/") for s in os.environ["AURORA_SAM_IOU_URLS"].split(",") if s.strip()]
errors = []
for url in urls:
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as response:
            if response.status == 200 and json.load(response).get("ok") is True:
                print(f"[grpo] verified SAM IoU service: {url}", flush=True)
                break
    except Exception as exc:
        errors.append(f"{url}: {exc!r}")
else:
    raise SystemExit("No healthy SAM IoU reward service; refusing GRPO: " + "; ".join(errors))
PY
export ENABLE_AUDIO_OUTPUT=0
export MAX_PIXELS="${MAX_PIXELS:-200704}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
# repo root on sys.path so external_plugins can `import swift_plugin.*`
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Schedule
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
# generation_batch_size must be divisible by BOTH the global train batch
# (per_device_train_batch_size * NPROC_PER_NODE) and num_generations.
#
# Throughput/memory trade-off on 8 V100-32GB (rollout generation dominates
# wall-clock). per_device=2 + ZeRO-2 still OOMs on V100-32GB - the rollout
# KV cache scales with batch and the reward forward peaks on top of it, so
# the safe shape is 1 concurrent rollout per GPU:
#   per_device=1, num_generations=8, gbs=8  : stable, ~71s/step (DEFAULT)
#   per_device=2, num_generations=8, gbs=16 : OOM even with ZeRO-2
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$((NUM_GENERATIONS * PER_DEVICE_TRAIN_BATCH_SIZE))}"
# DeepSpeed ZeRO-2 shards optimizer states + gradients across the 8 GPUs
# (~8 GB/card saved vs plain DDP) - REQUIRED for per_device=2 on V100-32GB.
# The repo's V100 config (SFT-proven) lives in configs/. Set DEEPSPEED="" to
# fall back to plain DDP with per_device=1.
export DEEPSPEED="${DEEPSPEED:-${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
export ADD_VERSION="${ADD_VERSION:-true}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# --- optional smoke-mode truncation --------------------------------------
if [[ "${MAX_SAMPLES:-}" != "-1" && -n "${MAX_SAMPLES:-}" ]]; then
  python - "${TRAIN_JSONL}" "${MAX_SAMPLES}" << 'EOF'
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

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ ! -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json" ]]; then
    echo "resume checkpoint is missing trainer_state.json: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "Resuming GRPO from: ${RESUME_FROM_CHECKPOINT}"
fi

# Absolute path: a bare `swift` resolves to a hacked source checkout in the
# base env (cate-pred-embedding/swift, missing dw_config) - always use the
# swift46 env's CLI.
SWIFT_BIN="${SWIFT_BIN:-swift}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model "${BASE_MODEL}" \
  --adapters "${STAGE1_ADAPTERS}" \
  --model_type aurora_qwen2_5_omni \
  --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/aurora_grpo.py" \
    "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --reward_funcs aurora_format aurora_class aurora_iou \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${VAL_JSONL}" \
  --new_special_tokens '[SEG]' \
  --num_generations "${NUM_GENERATIONS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --temperature 1.0 \
  --top_p 0.9 \
  --max_completion_length 512 \
  --beta 0.04 \
  --tuner_type lora \
  --deepspeed "${DEEPSPEED}" \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs mask_decoder \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_impl sdpa \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --max_steps "${MAX_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_length 4096 \
  --max_pixels "${MAX_PIXELS}" \
  --learning_rate "${LEARNING_RATE}" \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --num_generations_eval 1 \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --split_dataset_ratio 0 \
  --remove_unused_columns false \
  --callbacks aurora_sam_checkpoint \
  --report_to "${REPORT_TO}" \
  --add_version "${ADD_VERSION}" \
  --output_dir "${OUTPUT_DIR}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
