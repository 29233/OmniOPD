#!/usr/bin/env bash
# Full 8-GPU sparse top-10 dual-teacher OPD. Requires completed Lite audio synthesis.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-20261031_full_v1}"
SFT_ROOT="${SFT_ROOT:-${ROOT_DIR}/outputs/data/cot/refavs_answer_conditioned_full_20261029_v1/sft_prompted_v1}"
VISUAL_ROOT="${VISUAL_ROOT:-${SFT_ROOT}/opsd_sam2_visual_class_free_reverse500}"
AUDIO_ROOT="${AUDIO_ROOT:-${ROOT_DIR}/outputs/data/audio_privilege/clip_task_aware_train_full_20261031_lite_v1}"
VAL_AUDIO="${VAL_AUDIO:-${ROOT_DIR}/outputs/data/audio_privilege/clip_task_aware_val_20261027_v2_sanitized/captions.jsonl}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/outputs/data/privilege_top10_full_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/sam2_privileged_top10_opd_${RUN_ID}}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_joint_sft_20261030_v2_3epoch/v0-20261029-184656/checkpoint-3825}"
[[ -f "${AUDIO_ROOT}/captions.jsonl" ]] || { echo 'Full Lite audio captions are not complete yet.' >&2; exit 1; }
[[ -f "${AUDIO_ROOT}/validation.json" ]] || { echo 'Audio synthesis validation.json is missing.' >&2; exit 1; }
[[ -f "${CHECKPOINT}/args.json" && -f "${CHECKPOINT}/aurora_projector_config.json" ]] || { echo 'SAM2 SFT checkpoint metadata missing.' >&2; exit 1; }
mkdir -p "${DATA_ROOT}"
"${PYTHON_BIN}" scripts/data/prepare_sam2_privileged_top10_opd_full.py \
  --train-baseline "${SFT_ROOT}/structured_cot.jsonl" \
  --train-visual "${VISUAL_ROOT}/train_visual.jsonl" \
  --train-captions "${AUDIO_ROOT}/captions.jsonl" \
  --train-output "${DATA_ROOT}/train.jsonl" \
  --val-baseline "${SFT_ROOT}/eval_data/refavs_val.jsonl" \
  --val-visual "${VISUAL_ROOT}/val_visual.jsonl" \
  --val-captions "${VAL_AUDIO}" \
  --val-output "${DATA_ROOT}/val.jsonl" \
  --manifest "${DATA_ROOT}/manifest.json"
export BASE_MODEL="${BASE_MODEL:-models/Qwen2___5-Omni-7B}"
export STAGE1_ADAPTERS="${CHECKPOINT}" TEACHER_MODEL="${BASE_MODEL}" TEACHER_ADAPTERS="${CHECKPOINT}"
export TRAIN_JSONL="${DATA_ROOT}/train.jsonl" VAL_JSONL="${DATA_ROOT}/val.jsonl" OUTPUT_DIR
export NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export AURORA_SEG_BACKEND=sam2 AURORA_SAM2_CHECKPOINT="${AURORA_SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
export AURORA_SAM2_CONFIG="${AURORA_SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
export DATASET_VIDEO_FPS=1.0 FPS=1.0 AURORA_VIDEO_FPS=1.0 AURORA_NUM_SAM_FRAMES=10
export AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384 ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0
export AURORA_OPSD_PLUGIN="${ROOT_DIR}/swift_plugin/aurora_privileged_top10_opd.py"
export AURORA_OPSD_TEACHER_MODE=frozen AURORA_OPSD_REQUIRE_FROZEN_TEACHER=1 AURORA_OPSD_MASK_WEIGHT=0 AURORA_OPSD_SEG_DISTILL_WEIGHT=0
export LMBDA=1 BETA=0 TEMPERATURE=1 AURORA_PRIVILEGED_TOP10_FRACTION=.10 AURORA_PRIVILEGED_TOP10_RADIUS=1 AURORA_PRIVILEGED_TOP10_DIRECTION=forward
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}" MAX_LENGTH=5120 MAX_PIXELS=200704 TORCH_DTYPE=bfloat16
export OFFLOAD_TEACHER_MODEL=true OFFLOAD_OPTIMIZER=true LOG_COMPLETIONS=false REPORT_TO=tensorboard
export MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 MAX_STEPS=-1 NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}" SAVE_STEPS="${SAVE_STEPS:-500}" EVAL_STEPS="${EVAL_STEPS:-500}" SAVE_TOTAL_LIMIT=3
bash scripts/opd/train_opsd_swift.sh
