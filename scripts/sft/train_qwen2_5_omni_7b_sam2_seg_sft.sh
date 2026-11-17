#!/usr/bin/env bash
# Three-epoch Answer-CoT SFT with a SAM2 video-memory segmentation backend,
# followed by full test_s/test_u evaluation with every predicted mask saved.
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
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
  if [[ -n "${SWIFT_ENV:-}" && -x "${SWIFT_ENV}/bin/torchrun" ]]; then
    TORCHRUN_BIN="${SWIFT_ENV}/bin/torchrun"
  else
    TORCHRUN_BIN="$(command -v torchrun || true)"
  fi
elif [[ "${TORCHRUN_BIN}" != */* ]]; then
  TORCHRUN_BIN="$(command -v "${TORCHRUN_BIN}" || true)"
fi
export SWIFT_ENV PYTHON_BIN SWIFT_BIN TORCHRUN_BIN

RUN_ID="${RUN_ID:-20261030_v2_3epoch}"
MODEL="${MODEL:-models/Qwen2___5-Omni-7B}"
INIT_ADAPTER="${INIT_ADAPTER:-${ROOT_DIR}/outputs/train/sam2_sft_initializer_from_answer_cot_7b_20261030_v6}"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/cot/refavs_answer_conditioned_full_20261029_v1/sft_prompted_v1/structured_cot.jsonl}"
DATA_MANIFEST="${DATA_MANIFEST:-${ROOT_DIR}/outputs/data/cot/refavs_answer_conditioned_full_20261029_v1/sft_prompted_v1/manifest.json}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${ROOT_DIR}/outputs/data/cot/refavs_answer_conditioned_full_20261029_v1/sft_prompted_v1/eval_data}"
DATASET_AUDIT="${DATASET_AUDIT:-${ROOT_DIR}/outputs/artifacts/refavs_fps_audit_20261025/audit.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_joint_sft_${RUN_ID}}"
EVAL_DIR="${EVAL_DIR:-${ROOT_DIR}/outputs/eval/qwen2_5_omni_7b/sam2_joint_sft_${RUN_ID}}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EXPECTED_STEPS="${EXPECTED_STEPS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_DEVICES[@]}}"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == 1 ]]; then
  [[ "${NPROC_PER_NODE}" -eq 2 ]] || { echo "SAM2 smoke requires two GPUs" >&2; exit 2; }
  DATASET_ARG="${TRAIN_JSONL}#2";MAX_STEPS=1;SAVE_STEPS=1
else
  [[ "${NPROC_PER_NODE}" -eq 8 ]] || { echo "Full SAM2 SFT requires eight GPUs" >&2; exit 2; }
  DATASET_ARG="${TRAIN_JSONL}";MAX_STEPS=-1;SAVE_STEPS=500
fi

export AURORA_SEG_BACKEND=sam2 AURORA_SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" AURORA_SAM2_CONFIG="${SAM2_CONFIG}"
export FPS="${FPS:-1.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}" VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}" IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export AURORA_NUM_SAM_FRAMES=10 AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
unset AURORA_SEGMENTATION_ONLY
export AURORA_CE_WEIGHT=1.0 AURORA_BCE_WEIGHT=2.0 AURORA_DICE_WEIGHT=0.5
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false

for required in "${SWIFT_BIN}" "${PYTHON_BIN}" "${TORCHRUN_BIN}" "${MODEL}/config.json" \
  "${INIT_ADAPTER}/adapter_model.safetensors" "${INIT_ADAPTER}/adapter_config.json" \
  "${INIT_ADAPTER}/conversion_manifest.json" "${TRAIN_JSONL}" "${DATA_MANIFEST}" \
  "${EVAL_DATA_DIR}/manifest.json" "${EVAL_DATA_DIR}/refavs_test_s.jsonl" \
  "${EVAL_DATA_DIR}/refavs_test_u.jsonl" "${DATASET_AUDIT}" "${SAM2_CHECKPOINT}"; do
  [[ -e "${required}" ]] || { echo "Missing SAM2 SFT requirement: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/training_manifest.json" "${INIT_ADAPTER}" "${TRAIN_JSONL}" \
  "${DATA_MANIFEST}" "${DATASET_AUDIT}" "${SAM2_CHECKPOINT}" "${NPROC_PER_NODE}" "${SMOKE}" \
  "${EPOCHS}" "${LEARNING_RATE}" "${NPROC_PER_NODE}" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,adapter,data,data_manifest,audit,checkpoint=map(Path,sys.argv[1:7])
world=int(sys.argv[7]);smoke=int(sys.argv[8]);epochs=int(sys.argv[9]);lr=float(sys.argv[10]);global_batch=int(sys.argv[11])
conversion=json.loads((adapter/'conversion_manifest.json').read_text())
adapter_config=json.loads((adapter/'adapter_config.json').read_text())
source=Path(conversion['source']);args=json.loads((source/'args.json').read_text())
projector=json.loads((source/'aurora_projector_config.json').read_text())
source_training=json.loads((source/'training_manifest.json').read_text())
dataset=json.loads(data_manifest.read_text());media=json.loads(audit.read_text())
rows=sum(bool(x.strip()) for x in data.open())
if rows!=dataset.get('rows'): raise SystemExit(f'Training data/manifest rows mismatch: {rows}/{dataset.get("rows")}')
modalities=dataset.get('modality_counts',{})
if not all(modalities.get(name,0)>0 for name in ('Audio','Audio-Visual','Visual','Semantic')):
    raise SystemExit(f'Training data must contain all four referring modalities: {modalities}')
if conversion.get('sam2_checkpoint')!=str(checkpoint.resolve()): raise SystemExit('SAM2 initializer checkpoint mismatch')
for key,value in {'model_type':'aurora_qwen2_5_omni','template':'aurora_qwen2_5_omni','torch_dtype':'float16','attn_impl':'sdpa','max_length':5120,'max_pixels':200704}.items():
    if args.get(key)!=value: raise SystemExit(f'Answer-CoT initializer mismatch: {key}={args.get(key)!r}')
for key,value in {'input_hidden_size':3584,'projector_hidden_size':1024,'output_hidden_size':256,'layer_norm':False,'model_timeline_fps':1.0,'sam_frames':10}.items():
    if projector.get(key)!=value: raise SystemExit(f'Answer-CoT projector mismatch: {key}={projector.get(key)!r}')
required_modules={'embed_tokens','lm_head','text_hidden_fcs','sam_mask_decoder','memory_attention','memory_encoder','obj_ptr_proj','mask_downsample'}
if not required_modules.issubset(set(adapter_config.get('modules_to_save') or [])): raise SystemExit('SAM2 initializer lacks trainable modules')
if source_training.get('epochs')!=3 or source_training.get('max_length')!=5120: raise SystemExit('Source Answer-CoT SFT protocol mismatch')
if media.get('nominal_fps')!=1: raise SystemExit('Dataset manifest/FPS audit mismatch')
steps_per_epoch=max(1,rows//global_batch)
expected_steps=steps_per_epoch*epochs
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
manifest={'schema':2,'purpose':'SAM2 joint Answer-CoT SFT smoke' if smoke else f'{epochs}-epoch SAM2 joint Answer-CoT SFT',
 'initializer':str(adapter.resolve()),'training_data':str(data.resolve()),'rows':2 if smoke else rows,
 'modality_counts':modalities,
 'training_data_sha256':sha(data),'source_answer_cot_checkpoint':str(source.resolve()),
 'world_size':world,'effective_global_batch':world,'epochs':epochs,'max_steps':1 if smoke else -1,
 'steps_per_epoch_floor_estimate':steps_per_epoch,
 'expected_optimizer_steps':1 if smoke else expected_steps,
 'backend':{'name':'sam2_video_memory','checkpoint':str(checkpoint.resolve()),'config':'configs/sam2/sam2_hiera_l.yaml',
  'image_size':1024,'frames':10,'memory_frames':7,'object_presence_hard_gate':False},
 'trainable':['q/v LoRA','embed_tokens','lm_head','text_hidden_fcs','sam_mask_decoder','memory_attention','memory_encoder','obj_ptr_proj','mask_downsample'],
 'frozen':['Qwen base weights','vision tower','aligner','SAM2 Hiera image encoder','SAM2 prompt encoder'],
 'loss':{'ce':1.0,'bce':2.0,'dice':0.5},
 'optimizer':{'learning_rate':lr,'scheduler':'cosine','weight_decay':0.1,'warmup_ratio':0.0,'gradient_checkpointing':True},
 'dtype':'float16','attention':'sdpa','deepspeed':'zero2',
 'timeline':{'dataset_nominal_fps':1.0,'model_fps':1.0,'frames':10,'source':'2026-10-25 media audit'},
 'max_length':5120,'max_pixels':200704,
 'evaluation':{'automatic':not smoke,'splits':['test_s','test_u'],'audio':'original','decoding':'greedy','max_new_tokens':512,'save_masks':True,'per_frame_metrics':True}}
out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
PY

EXPECTED_STEPS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_optimizer_steps"])' \
  "${OUTPUT_DIR}/training_manifest.json")"

if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN=1: SAM2 SFT preflight passed";exit 0;fi
TRAIN_LOG="${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" sft \
  --model "${MODEL}" --adapters "${INIT_ADAPTER}" \
  --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --dataset "${DATASET_ARG}" --new_special_tokens '[SEG]' \
  --tuner_type lora --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample \
  --freeze_vit true --freeze_aligner true --torch_dtype float16 --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_train_epochs "${EPOCHS}" --max_steps "${MAX_STEPS}" \
  --max_length 5120 --truncation_strategy delete --max_pixels 200704 --learning_rate "${LEARNING_RATE}" --weight_decay 0.1 \
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 --max_grad_norm 1.0 \
  --lr_scheduler_type cosine --warmup_ratio 0 --gradient_checkpointing true --logging_steps 1 \
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit 4 --eval_strategy no \
  --dataloader_num_workers 0 --dataset_num_proc 1 --split_dataset_ratio 0 --remove_unused_columns false \
  --deepspeed "${ROOT_DIR}/configs/deepspeed_zero2_fp16_v100.json" --callbacks aurora_sam_checkpoint \
  --report_to tensorboard --output_dir "${OUTPUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"

FINAL_CHECKPOINT="$("${PYTHON_BIN}" - "${OUTPUT_DIR}/completion.json" "${OUTPUT_DIR}" "${SMOKE}" "${EPOCHS}" "${EXPECTED_STEPS}" <<'PY'
import json,sys
from pathlib import Path
out,root=map(Path,sys.argv[1:3]);smoke=int(sys.argv[3]);epochs=int(sys.argv[4]);expected=int(sys.argv[5]);states=list(root.glob('**/trainer_state.json'))
if not states: raise SystemExit('SAM2 SFT produced no trainer state')
state=max(states,key=lambda p:p.stat().st_mtime);checkpoint=state.parent;payload=json.loads(state.read_text())
for name in ('adapter_model.safetensors','sam2_video_decoder.safetensors','aurora_projector_config.json'):
    if not (checkpoint/name).is_file(): raise SystemExit(f'SAM2 checkpoint lacks {name}')
want_steps=1 if smoke else expected
if payload.get('global_step')!=want_steps or payload.get('max_steps')!=want_steps: raise SystemExit(f'Incomplete SAM2 SFT: {payload.get("global_step")}/{payload.get("max_steps")} expected {want_steps}')
if not smoke and payload.get('epoch')!=float(epochs): raise SystemExit(f'Incomplete SAM2 SFT epoch: {payload.get("epoch")}')
out.write_text(json.dumps({'status':'complete','smoke':bool(smoke),'global_step':payload.get('global_step'),'epochs':epochs,'max_length':5120,'checkpoint':str(checkpoint.resolve())},indent=2)+'\n')
print(checkpoint.resolve())
PY
)"

cp "${OUTPUT_DIR}/training_manifest.json" "${FINAL_CHECKPOINT}/training_manifest.json"
if [[ "${SMOKE}" == 1 ]]; then
  echo "[sam2-joint-sft] smoke complete=${FINAL_CHECKPOINT}"
  exit 0
fi

"${PYTHON_BIN}" - "${EVAL_DIR}/manifest.json" "${FINAL_CHECKPOINT}" "${EVAL_DATA_DIR}/manifest.json" <<'PY'
import json,sys
from pathlib import Path
out,checkpoint,data=map(Path,sys.argv[1:]);args=json.loads((checkpoint/'args.json').read_text());training=json.loads((checkpoint/'training_manifest.json').read_text());projector=json.loads((checkpoint/'aurora_projector_config.json').read_text())
for key,value in {'max_length':5120,'max_pixels':200704,'torch_dtype':'float16','attn_impl':'sdpa','num_train_epochs':float(training['epochs']),'weight_decay':0.1,'gradient_checkpointing':True}.items():
    if args.get(key)!=value: raise SystemExit(f'SAM2 training/eval mismatch: {key}={args.get(key)!r}')
if training.get('epochs') not in (1,2,3) or projector.get('backend')!='sam2_video_memory' or projector.get('model_timeline_fps')!=1.0 or projector.get('sam_frames')!=10:
    raise SystemExit('SAM2 checkpoint protocol mismatch')
manifest={'schema':1,'purpose':f"automatic full test_s/test_u evaluation after {training['epochs']}-epoch SAM2 joint SFT",
 'checkpoint':str(checkpoint.resolve()),'evaluation_data_manifest':str(data.resolve()),'splits':['test_s','test_u'],'world_size':8,
 'settings':{'model_fps':1.0,'dataset_nominal_fps':1.0,'frames':10,'backend':'sam2_video_memory',
  'max_pixels':200704,'max_length':5120,'max_new_tokens':512,'dtype':'float16','attention':'sdpa',
  'audio':'original','decoding':'greedy','save_masks':True,'per_frame_metrics':True},
 'inference_differences':['greedy generated CoT replaces teacher-forced training response','no dropout or optimizer state','original test audio']}
out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
PY

"${TORCHRUN_BIN}" --standalone --nproc_per_node=8 scripts/eval/test_qwen2_5_omni_refavs.py \
  --model "${MODEL}" --checkpoint "${FINAL_CHECKPOINT}" --data-dir "${EVAL_DATA_DIR}" \
  --output-dir "${EVAL_DIR}" --splits test_s test_u --max-samples -1 --max-new-tokens 512 \
  --torch-dtype float16 --attn-impl sdpa --max-pixels 200704 --max-length 5120 \
  --no-privileged-category --run-sam --save-masks --dist-backend gloo --log-every 20 \
  2>&1 | tee "${EVAL_DIR}/run.log"

echo "[sam2-joint-sft] train=${FINAL_CHECKPOINT} eval=${EVAL_DIR}/summary.json"
