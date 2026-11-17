#!/usr/bin/env bash
# GRPO from the completed 7B SAM2 SFT baseline with exactly two rewards:
# binary class correctness and ten-frame predicted-mask mIoU.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
if [[ "${OMNIOPD_LOAD_ENV:-0}" == "1" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi
export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"

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
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_joint_sft_20261030_v2_3epoch/v0-20261029-184656/checkpoint-3825}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"

SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}/outputs/data/cot/refavs_answer_conditioned_full_20261029_v1/sft_prompted_v1}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${SOURCE_ROOT}/structured_cot.jsonl}"
VAL_SOURCE="${VAL_SOURCE:-${SOURCE_ROOT}/eval_data/refavs_val.jsonl}"
DATA_DIR="${DATA_DIR:-${SOURCE_ROOT}/grpo_sam2_class_mask_v1}"
DATASET_AUDIT="${DATASET_AUDIT:-${ROOT_DIR}/outputs/artifacts/refavs_fps_audit_20261025/audit.json}"
EXPECTED_STEPS="${EXPECTED_STEPS:-}"
EXPECTED_TRAIN_ROWS="${EXPECTED_TRAIN_ROWS:-}"
EXPECTED_VAL_ROWS="${EXPECTED_VAL_ROWS:-}"
EXPECTED_AUDIT_CLIPS="${EXPECTED_AUDIT_CLIPS:-}"
RUN_ID="${RUN_ID:-20261031_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_grpo_class_mask_${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_DEVICES[@]}}"
[[ "${NPROC_PER_NODE}" -eq 8 ]] || { echo "SAM2 GRPO requires exactly 8 GPUs" >&2; exit 2; }

# Checkpoint-matched media/model settings. The dataset has nominal 1 FPS
# coverage (10 stored frames over 10 seconds); exact frame times are unknown.
export AURORA_SEG_BACKEND=sam2
export AURORA_SAM2_CHECKPOINT="${SAM2_CHECKPOINT}"
export AURORA_SAM2_CONFIG="${SAM2_CONFIG}"
export FPS="${FPS:-1.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}" VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}" IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export AURORA_NUM_SAM_FRAMES=10 AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false
export AURORA_GRPO_REWARD_FAIL_OPEN="${AURORA_GRPO_REWARD_FAIL_OPEN:-0}"

# These defaults are intentionally configurable pending experiment approval.
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
CLASS_REWARD_WEIGHT="${CLASS_REWARD_WEIGHT:-1.0}"
MASK_REWARD_WEIGHT="${MASK_REWARD_WEIGHT:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.04}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"

(( GENERATION_BATCH_SIZE % NUM_GENERATIONS == 0 )) || {
  echo "GENERATION_BATCH_SIZE must be divisible by NUM_GENERATIONS" >&2; exit 2;
}
(( GENERATION_BATCH_SIZE % NPROC_PER_NODE == 0 )) || {
  echo "GENERATION_BATCH_SIZE must be divisible by the 8-rank global batch" >&2; exit 2;
}

SFT_TRAINING_MANIFEST=""
for candidate in \
  "${SFT_CHECKPOINT}/training_manifest.json" \
  "$(dirname "${SFT_CHECKPOINT}")/training_manifest.json" \
  "$(dirname "$(dirname "${SFT_CHECKPOINT}")")/training_manifest.json"; do
  if [[ -f "${candidate}" ]]; then
    SFT_TRAINING_MANIFEST="${candidate}"
    break
  fi
done
for required in "${SWIFT_BIN}" "${PYTHON_BIN}" "${MODEL}/config.json" \
  "${SFT_CHECKPOINT}/adapter_model.safetensors" "${SFT_CHECKPOINT}/sam2_video_decoder.safetensors" \
  "${SFT_CHECKPOINT}/args.json" "${SFT_TRAINING_MANIFEST}" \
  "${SFT_CHECKPOINT}/aurora_projector_config.json" "${SAM2_CHECKPOINT}" \
  "${TRAIN_SOURCE}" "${VAL_SOURCE}" "${DATASET_AUDIT}"; do
  [[ -n "${required}" && -e "${required}" ]] || { echo "Missing SAM2 GRPO requirement: ${required:-<missing training_manifest.json>}" >&2; exit 1; }
done

"${PYTHON_BIN}" scripts/data/prepare_sam2_grpo_data.py \
  --train-source "${TRAIN_SOURCE}" --val-source "${VAL_SOURCE}" --output-dir "${DATA_DIR}"

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" - "${OUTPUT_DIR}/training_manifest.json" "${SFT_CHECKPOINT}" \
  "${SFT_TRAINING_MANIFEST}" "${DATA_DIR}/manifest.json" "${DATASET_AUDIT}" "${SAM2_CHECKPOINT}" \
  "${NUM_GENERATIONS}" "${GENERATION_BATCH_SIZE}" "${CLASS_REWARD_WEIGHT}" \
  "${MASK_REWARD_WEIGHT}" "${LEARNING_RATE}" "${BETA}" "${TEMPERATURE}" \
  "${TOP_P}" "${MAX_COMPLETION_LENGTH}" "${NUM_TRAIN_EPOCHS}" "${MAX_STEPS}" \
  "${EXPECTED_STEPS}" "${EXPECTED_TRAIN_ROWS}" "${EXPECTED_VAL_ROWS}" "${EXPECTED_AUDIT_CLIPS}" <<'PY'
import json,sys
from pathlib import Path

(out,checkpoint,sft_manifest_path,data_manifest,audit,sam2_checkpoint)=map(Path,sys.argv[1:7])
num_generations=int(sys.argv[7]);generation_batch=int(sys.argv[8])
class_weight=float(sys.argv[9]);mask_weight=float(sys.argv[10]);lr=float(sys.argv[11])
beta=float(sys.argv[12]);temperature=float(sys.argv[13]);top_p=float(sys.argv[14])
max_completion=int(sys.argv[15]);epochs=float(sys.argv[16]);max_steps=int(sys.argv[17])
expected_steps_env=sys.argv[18];expected_train_rows=sys.argv[19];expected_val_rows=sys.argv[20];expected_audit_clips=sys.argv[21]
args=json.loads((checkpoint/'args.json').read_text())
training=json.loads(sft_manifest_path.read_text())
projector=json.loads((checkpoint/'aurora_projector_config.json').read_text())
data=json.loads(data_manifest.read_text());media=json.loads(audit.read_text())
for key,value in {'model_type':'aurora_qwen2_5_omni','template':'aurora_qwen2_5_omni',
                  'torch_dtype':'float16','attn_impl':'sdpa','max_length':5120,
                  'max_pixels':200704}.items():
    if args.get(key)!=value: raise SystemExit(f'SAM2 SFT mismatch: {key}={args.get(key)!r}')
for key,value in {'backend':'sam2_video_memory','input_hidden_size':3584,
                  'projector_hidden_size':1024,'output_hidden_size':256,
                  'model_timeline_fps':1.0,'sam_frames':10}.items():
    if projector.get(key)!=value: raise SystemExit(f'SAM2 projector mismatch: {key}={projector.get(key)!r}')
if training.get('epochs')!=3:
    raise SystemExit('The selected initializer is not a completed three-epoch SAM2 SFT baseline')
recorded_steps=training.get('expected_optimizer_steps')
if expected_steps_env:
    if expected_steps_env!='any':
        want_steps=int(expected_steps_env)
        padded=training.get('distributed_padding',{})
        derived=recorded_steps if recorded_steps is not None else ((int(padded['padded_rows'])//8)*3 if 'padded_rows' in padded else None)
        if derived!=want_steps:
            raise SystemExit(f'Expected {want_steps} SFT optimizer steps, got {derived!r}')
else:
    is_answer_cot=(recorded_steps==3825)
    is_merged_sft=(training.get('backend')=='sam2_video_memory' and training.get('rows')==12820 and training.get('expected_epoch_checkpoints')==3)
    if not (is_answer_cot or is_merged_sft or recorded_steps==4809):
        raise SystemExit(f'Unrecognized 3-epoch SAM2 SFT baseline metadata (expected_optimizer_steps={recorded_steps!r}); set EXPECTED_STEPS to override')
backend_info=training.get('backend')
if isinstance(backend_info,dict):
    if backend_info.get('name') not in (None,'sam2_video_memory'):
        raise SystemExit(f'SAM2 backend differs from SFT: {backend_info!r}')
    recorded_ckpt=backend_info.get('checkpoint')
    if recorded_ckpt and recorded_ckpt!=str(sam2_checkpoint.resolve()) and Path(recorded_ckpt).name!=sam2_checkpoint.name:
        raise SystemExit(f'SAM2 base checkpoint differs from SFT: {recorded_ckpt} vs {sam2_checkpoint}')
elif backend_info!='sam2_video_memory':
    raise SystemExit(f'SAM2 backend differs from SFT: {backend_info!r}')
train_rows=data['splits']['train']['rows']; val_rows=data['splits']['val']['rows']
if train_rows<=0 or val_rows<=0:
    raise SystemExit(f'Empty GRPO train/val rows: {train_rows}/{val_rows}')
if expected_train_rows and train_rows!=int(expected_train_rows):
    raise SystemExit(f'Unexpected GRPO train row count: {train_rows} != {expected_train_rows}')
if expected_val_rows and val_rows!=int(expected_val_rows):
    raise SystemExit(f'Unexpected GRPO val row count: {val_rows} != {expected_val_rows}')
if not data.get('assistant_labels_removed'):
    raise SystemExit('GRPO data still contains teacher assistant responses')
if media.get('nominal_fps')!=1 or media.get('unique_clips_checked',0)<1:
    raise SystemExit('Dataset timing audit mismatch')
if expected_audit_clips and media.get('unique_clips_checked')!=int(expected_audit_clips):
    raise SystemExit(f'Dataset timing audit clips mismatch: {media.get("unique_clips_checked")} != {expected_audit_clips}')
manifest={
  'schema':1,
  'purpose':'GRPO optimization of the completed SAM2 SFT baseline',
  'initializer':str(checkpoint.resolve()),
  'data_manifest':str(data_manifest.resolve()),
  'world_size':8,
  'rewards':[
    {'name':'aurora_class_exact','range':[0,1],'weight':class_weight,
     'definition':'1 iff the last The target is <category> prediction exactly matches the normalized GT category'},
    {'name':'aurora_sam2_mask_iou','range':[0,1],'weight':mask_weight,
     'definition':'mean binary IoU over all 10 SAM2 masks decoded from the first generated [SEG] hidden state'},
  ],
  'no_format_reward':True,
  'no_supervised_mask_loss':True,
  'reward_decoder':{'projector':'frozen trained checkpoint component',
                    'sam2':'frozen trained checkpoint decoder/memory and Hiera image encoder',
                    'missing_seg_reward':0.0,'threshold':'mask logit > 0'},
  'policy_trainable':['q/v LoRA','embed_tokens','lm_head'],
  'optimization':{'learning_rate':lr,'beta':beta,'num_generations':num_generations,
                  'generation_batch_size':generation_batch,'temperature':temperature,'top_p':top_p,
                  'max_completion_length':max_completion,'epochs':epochs,'max_steps':max_steps,
                  'per_device_batch_size':1,'gradient_accumulation_steps':1,
                  'scheduler':'cosine','weight_decay':0.1,'dtype':'float16','attention':'sdpa'},
  'media':{'dataset_nominal_fps':1.0,'model_timeline_fps':1.0,'frames':10,
           'exact_sampling_instants_verified':False,'max_pixels':200704,'max_length':5120},
  'intended_differences_from_sft':['stochastic rollout decoding (temperature/top-p)',
                                   'GRPO sequence objective replaces CE+BCE+Dice',
                                   'projector and SAM2 reward decoder are frozen'],
}
out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
PY

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "DRY_RUN=1: SAM2 GRPO data and checkpoint preflight passed"
  exit 0
fi

TRAIN_LOG="${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" --adapters "${SFT_CHECKPOINT}" \
  --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/aurora_sam2_grpo.py" "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --reward_funcs aurora_class_exact aurora_sam2_mask_iou \
  --reward_weights "${CLASS_REWARD_WEIGHT}" "${MASK_REWARD_WEIGHT}" \
  --dataset "${DATA_DIR}/train.jsonl" --new_special_tokens '[SEG]' \
  --num_generations "${NUM_GENERATIONS}" --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --beta "${BETA}" --num_iterations 1 \
  --tuner_type lora --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample \
  --freeze_vit true --freeze_aligner true --torch_dtype float16 --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" --max_steps "${MAX_STEPS}" \
  --max_length 5120 --truncation_strategy delete --max_pixels 200704 \
  --learning_rate "${LEARNING_RATE}" --weight_decay 0.1 \
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 --max_grad_norm 1.0 \
  --lr_scheduler_type cosine --warmup_ratio 0 --gradient_checkpointing true \
  --logging_steps 1 --log_completions true \
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy no --dataloader_num_workers 0 --dataset_num_proc 1 \
  --split_dataset_ratio 0 --remove_unused_columns false \
  --deepspeed "${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json" \
  --callbacks aurora_sam_checkpoint --report_to tensorboard \
  --add_version false --output_dir "${OUTPUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"
