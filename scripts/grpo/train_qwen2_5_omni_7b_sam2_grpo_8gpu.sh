#!/usr/bin/env bash
# GRPO from the completed 7B SAM2 SFT baseline with exactly two rewards:
# binary class correctness and ten-frame predicted-mask mIoU.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export PYTHONPATH="${ROOT_DIR}/third_party/sam2_runtime_deps:${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"

SWIFT_ENV="${SWIFT_ENV:-${CONDA_PREFIX:-/usr/local}}"
SWIFT_BIN="${SWIFT_BIN:-${SWIFT_ENV}/bin/swift}"
PYTHON_BIN="${PYTHON_BIN:-${SWIFT_ENV}/bin/python}"
MODEL="${MODEL:-models/Qwen2___5-Omni-7B}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_joint_sft_20261030_v2_3epoch/v0-20261029-184656/checkpoint-3825}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"

SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}/outputs/data/cot/refavs_answer_conditioned_full_20261029_v1/sft_prompted_v1}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${SOURCE_ROOT}/structured_cot.jsonl}"
VAL_SOURCE="${VAL_SOURCE:-${SOURCE_ROOT}/eval_data/refavs_val.jsonl}"
DATA_DIR="${DATA_DIR:-${SOURCE_ROOT}/grpo_sam2_class_mask_v1}"
DATASET_AUDIT="${DATASET_AUDIT:-${ROOT_DIR}/outputs/artifacts/refavs_fps_audit_20261025/audit.json}"
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
export FPS=1.0 AURORA_VIDEO_FPS=1.0
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384
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

for required in "${SWIFT_BIN}" "${PYTHON_BIN}" "${MODEL}/config.json" \
  "${SFT_CHECKPOINT}/adapter_model.safetensors" "${SFT_CHECKPOINT}/sam2_video_decoder.safetensors" \
  "${SFT_CHECKPOINT}/args.json" "${SFT_CHECKPOINT}/training_manifest.json" \
  "${SFT_CHECKPOINT}/aurora_projector_config.json" "${SAM2_CHECKPOINT}" \
  "${TRAIN_SOURCE}" "${VAL_SOURCE}" "${DATASET_AUDIT}"; do
  [[ -e "${required}" ]] || { echo "Missing SAM2 GRPO requirement: ${required}" >&2; exit 1; }
done

"${PYTHON_BIN}" scripts/data/prepare_sam2_grpo_data.py \
  --train-source "${TRAIN_SOURCE}" --val-source "${VAL_SOURCE}" --output-dir "${DATA_DIR}"

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" - "${OUTPUT_DIR}/training_manifest.json" "${SFT_CHECKPOINT}" \
  "${DATA_DIR}/manifest.json" "${DATASET_AUDIT}" "${SAM2_CHECKPOINT}" \
  "${NUM_GENERATIONS}" "${GENERATION_BATCH_SIZE}" "${CLASS_REWARD_WEIGHT}" \
  "${MASK_REWARD_WEIGHT}" "${LEARNING_RATE}" "${BETA}" "${TEMPERATURE}" \
  "${TOP_P}" "${MAX_COMPLETION_LENGTH}" "${NUM_TRAIN_EPOCHS}" "${MAX_STEPS}" <<'PY'
import json,sys
from pathlib import Path

(out,checkpoint,data_manifest,audit,sam2_checkpoint)=map(Path,sys.argv[1:6])
num_generations=int(sys.argv[6]);generation_batch=int(sys.argv[7])
class_weight=float(sys.argv[8]);mask_weight=float(sys.argv[9]);lr=float(sys.argv[10])
beta=float(sys.argv[11]);temperature=float(sys.argv[12]);top_p=float(sys.argv[13])
max_completion=int(sys.argv[14]);epochs=float(sys.argv[15]);max_steps=int(sys.argv[16])
args=json.loads((checkpoint/'args.json').read_text())
training=json.loads((checkpoint/'training_manifest.json').read_text())
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
if training.get('epochs')!=3 or training.get('expected_optimizer_steps')!=3825:
    raise SystemExit('The selected initializer is not the completed three-epoch SAM2 SFT baseline')
if training.get('backend',{}).get('checkpoint')!=str(sam2_checkpoint.resolve()):
    raise SystemExit('SAM2 base checkpoint differs from SFT')
if data['splits']['train']['rows']!=10201 or data['splits']['val']['rows']!=1277:
    raise SystemExit('Unexpected GRPO train/val row counts')
if data['splits']['train']['skipped_null_class_rows']!=0 or data['splits']['val']['skipped_null_class_rows']!=64:
    raise SystemExit('Unexpected count of invalid null-class rows')
if not data.get('assistant_labels_removed'):
    raise SystemExit('GRPO data still contains teacher assistant responses')
if media.get('nominal_fps')!=1 or media.get('unique_clips_checked')!=3566:
    raise SystemExit('Dataset timing audit mismatch')
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
