#!/usr/bin/env bash
# Historical ms-swift Joint protocol with:
#   1) online Audio-privileged current-policy Teacher
#   2) online Visual-privileged current-policy Teacher
#   3) frozen GRPO checkpoint Teacher (ordinary Student prompt)
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
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
MODEL="${MODEL:-models/Qwen2___5-Omni-7B}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${ROOT_DIR}/outputs/train/sam2_merged_sft_3epoch_8gpu_20261102_merged_v4_full/checkpoint-4809}"
GRPO_CHECKPOINT="${GRPO_CHECKPOINT:-${ROOT_DIR}/outputs/train/merged_grpo_online_20261103_v1/checkpoint-1000}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/outputs/data/merged_student_only_online_20261103_v1}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"

export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#GPU_DEVICES[@]}" == 8 ]] || { echo "Three-teacher Joint requires 8 GPUs" >&2; exit 2; }

for required in \
  "$SWIFT_BIN" "$PYTHON_BIN" "$MODEL/config.json" \
  "$SFT_CHECKPOINT/adapter_model.safetensors" \
  "$SFT_CHECKPOINT/aurora_projector_config.json" \
  "$SFT_CHECKPOINT/sam2_video_decoder.safetensors" \
  "$GRPO_CHECKPOINT/adapter_model.safetensors" \
  "$GRPO_CHECKPOINT/aurora_projector_config.json" \
  "$GRPO_CHECKPOINT/sam2_video_decoder.safetensors" \
  "$TRAIN_JSONL" "$DATA_ROOT/manifest.json" "$SAM2_CHECKPOINT"; do
  [[ -n "$required" && -e "$required" ]] || { echo "Missing three-teacher requirement: ${required:-<empty>}" >&2; exit 1; }
done

export AURORA_SEG_BACKEND=sam2
export AURORA_SAM2_CHECKPOINT="$SAM2_CHECKPOINT"
export AURORA_SAM2_CONFIG="$SAM2_CONFIG"
export DATASET_VIDEO_FPS="${DATASET_VIDEO_FPS:-1.0}"
export FPS="${FPS:-1.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384
export AURORA_NUM_SAM_FRAMES=10 AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false
export AURORA_GRPO_REWARD_FAIL_OPEN=0

export AURORA_VERIFIED_TEACHERS=both
export AURORA_VERIFIED_KD_COEF="${AURORA_VERIFIED_KD_COEF:-0.25}"
export AURORA_VERIFIED_TOKEN_FRACTION="${AURORA_VERIFIED_TOKEN_FRACTION:-0.10}"
export AURORA_VERIFIED_TOKEN_RADIUS="${AURORA_VERIFIED_TOKEN_RADIUS:-1}"
export AURORA_VERIFIED_TASK_REWARD_COEF="${AURORA_VERIFIED_TASK_REWARD_COEF:-1.0}"
export AURORA_VERIFIED_IOU_MARGIN="${AURORA_VERIFIED_IOU_MARGIN:-0.01}"
export AURORA_VERIFIED_CATEGORY_BONUS="${AURORA_VERIFIED_CATEGORY_BONUS:-2.0}"
export AURORA_VERIFIED_IOU_GAIN_WEIGHT="${AURORA_VERIFIED_IOU_GAIN_WEIGHT:-0.75}"
export AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT="${AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT:-0.25}"
export AURORA_VERIFIED_ROUTE_TEMPERATURE="${AURORA_VERIFIED_ROUTE_TEMPERATURE:-0.25}"
export AURORA_VERIFIED_K3_CLIP="${AURORA_VERIFIED_K3_CLIP:-10}"
export AURORA_ONLINE_TEACHER_TEMPERATURE="${AURORA_ONLINE_TEACHER_TEMPERATURE:-1.0}"
export AURORA_ONLINE_TEACHER_TOP_P="${AURORA_ONLINE_TEACHER_TOP_P:-0.9}"

export MAX_STEPS="${MAX_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-7}"
export REFERENCE_KL_BETA="${REFERENCE_KL_BETA:-0.01}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
export MASTER_PORT="${MASTER_PORT:-29745}"
export MODEL SFT_CHECKPOINT GRPO_CHECKPOINT TRAIN_JSONL OUTPUT_DIR
export DATA_ROOT

# ms-swift treats an identical --model/--teacher_model path as implicit
# self-distillation and ignores the explicit teacher adapter.  Use a symlink
# alias so the frozen GRPO-1000 adapter is loaded as a genuine local teacher.
TEACHER_ALIAS_ROOT="${ROOT_DIR}/outputs/.three_teacher_model_alias"
TEACHER_MODEL="${TEACHER_ALIAS_ROOT}/$(basename "$MODEL")"
mkdir -p "$TEACHER_ALIAS_ROOT"
if [[ -e "$TEACHER_MODEL" && ! -L "$TEACHER_MODEL" ]]; then
  echo "Refusing non-symlink teacher alias: $TEACHER_MODEL" >&2
  exit 1
fi
ln -sfn "$MODEL" "$TEACHER_MODEL"
export TEACHER_MODEL

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" - "$OUTPUT_DIR/training_manifest.json" <<'PY'
import hashlib, json, os
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

root=Path(os.environ["OUTPUT_DIR"])
data=Path(os.environ["DATA_ROOT"])/"manifest.json"
manifest=json.loads(data.read_text())
if manifest.get("offline_teacher_required") is not False:
    raise SystemExit("Three-teacher Joint requires Student-only online data")
rows=sum(1 for line in Path(os.environ["TRAIN_JSONL"]).open() if line.strip())
if rows != 12820:
    raise SystemExit(f"Unexpected Student-only row count: {rows}")
for key in ("train_sha256","offline_teacher_required"):
    if key not in manifest: raise SystemExit(f"Student-only manifest misses {key}")
root.joinpath("training_manifest.json").write_text(json.dumps({
    "schema": 1,
    "purpose": "historical ms-swift Joint with online Audio/Visual plus frozen GRPO teacher",
    "initializer": str(Path(os.environ["SFT_CHECKPOINT"]).resolve()),
    "teacher_models": [
        {"name":"audio","source":"current-policy privileged Audio prompt","online":True},
        {"name":"visual","source":"current-policy privileged Visual prompt","online":True},
        {"name":"grpo_anchor","source":str(Path(os.environ["GRPO_CHECKPOINT"]).resolve()),
         "online":False,"prompt":"ordinary Student prompt"},
    ],
    "routing": {
        "implementation":"swift_plugin/aurora_online_three_teacher_joint.py",
        "license":"existing category/IoU verified routing generalized to K=3",
        "teacher_order":["audio","visual","grpo_anchor"],
        "anchor_token_scope":"all CoT stages before existing low-likelihood selection",
    },
    "task_reward":"0.5*class_exact + 0.5*class_exact*mean_10_frame_iou",
    "distillation":"native signed sampled-token logp_teacher-logp_student",
    "kd_coef":float(os.environ["AURORA_VERIFIED_KD_COEF"]),
    "token_fraction":float(os.environ["AURORA_VERIFIED_TOKEN_FRACTION"]),
    "token_radius":int(os.environ["AURORA_VERIFIED_TOKEN_RADIUS"]),
    "optimization":{"max_steps":int(os.environ["MAX_STEPS"]),
                    "learning_rate":float(os.environ["LEARNING_RATE"]),
                    "reference_kl_beta":float(os.environ["REFERENCE_KL_BETA"]),
                    "num_generations":int(os.environ["NUM_GENERATIONS"]),
                    "max_completion_length":int(os.environ["MAX_COMPLETION_LENGTH"]),
                    "dtype":"bfloat16"},
    "media":{"dataset_nominal_fps":1.0,"model_timeline_fps":1.0,
             "frames":10,"max_pixels":200704,"max_length":5120,
             "fps_max_frames":768,"video_max_token_num":768,
             "image_max_token_num":16384},
    "data_manifest":str(data.resolve()),
    "data_sha256":manifest["train_sha256"],
    "data_rows":rows,
}, ensure_ascii=False, indent=2)+"\n")
print("three-teacher preflight passed")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"$SWIFT_BIN" rlhf \
  --rlhf_type grpo \
  --model "$MODEL" --adapters "$SFT_CHECKPOINT" \
  --teacher_model "$TEACHER_MODEL" --teacher_adapters "$GRPO_CHECKPOINT" \
  --teacher_model_type aurora_qwen2_5_omni --teacher_kl_coef 0 \
  --offload_teacher_model true --offload_optimizer true \
  --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni \
  --custom_register_path "$ROOT_DIR/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins \
    "$ROOT_DIR/swift_plugin/aurora_online_three_teacher_joint.py" \
    "$ROOT_DIR/swift_plugin/sam_checkpoint.py" \
  --reward_funcs aurora_class_exact aurora_sam2_mask_iou \
  --reward_weights 0.5 0.5 \
  --dataset "$TRAIN_JSONL" --new_special_tokens '[SEG]' \
  --num_generations "$NUM_GENERATIONS" --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --temperature 1.0 --top_p 0.9 \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --beta "$REFERENCE_KL_BETA" --num_iterations 1 --dynamic_sample false \
  --tuner_type lora \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps "$MAX_STEPS" --num_train_epochs 1 \
  --max_length 5120 --truncation_strategy delete --max_pixels 200704 \
  --learning_rate "$LEARNING_RATE" --weight_decay 0 \
  --adam_beta1 .9 --adam_beta2 .95 --adam_epsilon 1e-8 --max_grad_norm 1 \
  --lr_scheduler_type cosine --warmup_ratio 0 \
  --gradient_checkpointing true --logging_steps 1 --log_completions true \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 5 \
  --eval_strategy no --dataloader_num_workers 0 --dataset_num_proc 1 \
  --split_dataset_ratio 0 --remove_unused_columns false \
  --callbacks aurora_sam_checkpoint --report_to tensorboard \
  --add_version false --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/train_$(date -u +%Y%m%d_%H%M%S).log"
