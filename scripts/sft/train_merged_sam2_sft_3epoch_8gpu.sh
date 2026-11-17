#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT_DIR"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
if [[ "${OMNIOPD_LOAD_ENV:-0}" == "1" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi
export PYTHONPATH="$ROOT_DIR/third_party/sam2:$ROOT_DIR:${PYTHONPATH:-}"
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
INIT_ADAPTER="${INIT_ADAPTER:-$ROOT_DIR/outputs/train/sam2_sft_initializer_from_answer_cot_7b_20261030_v6}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/outputs/data/cot/refavs_merged_sft_20261103_v1}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/structured_cot.jsonl}"; DATA_MANIFEST="${DATA_MANIFEST:-$DATA_DIR/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/sam2_merged_sft_3epoch_8gpu_20261103_v1}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"; EPOCHS="${EPOCHS:-3}"
EXPECTED_ROWS="${EXPECTED_ROWS:-12820}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export FPS="${FPS:-1.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}" VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}" IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#DEVICES[@]}" -eq 8 ]] || { echo "Merged SFT requires eight GPUs" >&2; exit 2; }
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export AURORA_SEG_BACKEND=sam2 AURORA_SAM2_CHECKPOINT="$SAM2_CHECKPOINT" AURORA_SAM2_CONFIG="$SAM2_CONFIG" AURORA_NUM_SAM_FRAMES=10 AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false AURORA_CE_WEIGHT=1.0 AURORA_BCE_WEIGHT=2.0 AURORA_DICE_WEIGHT=0.5
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { echo "Missing executable PYTHON_BIN: ${PYTHON_BIN:-<unset>}" >&2; exit 1; }
[[ -n "$SWIFT_BIN" && -x "$SWIFT_BIN" ]] || { echo "Missing executable SWIFT_BIN: ${SWIFT_BIN:-<unset>}" >&2; exit 1; }
[[ -f "$MODEL/config.json" && -f "$SAM2_CHECKPOINT" && -f "$TRAIN_JSONL" && -f "$DATA_MANIFEST" && -f "$INIT_ADAPTER/adapter_model.safetensors" ]] || { echo "missing merged SFT input (MODEL/SAM2_CHECKPOINT/TRAIN_JSONL/DATA_MANIFEST/INIT_ADAPTER)" >&2; exit 1; }
if [[ -f "$OUTPUT_DIR/completion.json" ]]; then
  echo "merged_sft_already_complete=$OUTPUT_DIR/completion.json"
  exit 0
fi
if [[ "${DRY_RUN:-0}" != 1 ]] && find "$OUTPUT_DIR" -name 'trainer_state.json' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to overwrite an existing training run: $OUTPUT_DIR" >&2; exit 1
fi
mkdir -p "$OUTPUT_DIR"
SOURCE_TRAIN_JSONL="$TRAIN_JSONL"
"$PYTHON_BIN" scripts/data/prepare_sft_distributed_padding.py --source "$SOURCE_TRAIN_JSONL" \
  --output "$OUTPUT_DIR/train_distributed_padded.jsonl" --world-size 8
PADDED_TRAIN_JSONL="$OUTPUT_DIR/train_distributed_padded.jsonl"
"$PYTHON_BIN" - "$OUTPUT_DIR/training_manifest.json" "$DATA_MANIFEST" "$INIT_ADAPTER" "$PADDED_TRAIN_JSONL" "$SOURCE_TRAIN_JSONL" "$EXPECTED_ROWS" <<'PY'
import json,sys,hashlib
from pathlib import Path
out,data,adapter,train,source_train=map(Path,sys.argv[1:6]);expected_rows=int(sys.argv[6])
d=json.loads(data.read_text()); rows=sum(1 for l in source_train.open() if l.strip())
if rows!=d['rows'] or (expected_rows > 0 and rows!=expected_rows):raise SystemExit(f'expected {expected_rows} merged rows, got {rows}')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'schema':1,'purpose':'3-epoch merged reviewed Audio/non-audio SAM2 SFT','initializer':str(adapter.resolve()),'training_data':str(source_train.resolve()),'rows':rows,'modality_counts':d['modality_counts'],'epochs':3,'save_strategy':'epoch','expected_epoch_checkpoints':3,'world_size':8,'effective_global_batch':8,'dataloader_drop_last':False,'strict_data_loading':True,'max_length':5120,'max_pixels':200704,'dtype':'float16','attn_impl':'sdpa','dataset_nominal_fps':1.0,'model_timeline_fps':1.0,'sam_frames':10,'backend':'sam2_video_memory','loss':{'ce':1.0,'bce':2.0,'dice':0.5},'trainable':['q/v LoRA','embed_tokens','lm_head','text_hidden_fcs','trained SAM2 decoder/memory'],'evaluation':'external 4-GPU queue on checkpoint-epoch-3 then epoch-1 then epoch-2','data_sha256':sha(source_train),'launcher':str(Path('scripts/sft/train_merged_sam2_sft_3epoch_8gpu.sh').resolve()),'launcher_sha256':sha(Path('scripts/sft/train_merged_sam2_sft_3epoch_8gpu.sh'))}
manifest['distributed_padding']=json.loads(train.with_suffix('.manifest.json').read_text())
manifest['actual_training_jsonl']=str(train.resolve())
out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
PY
if [[ "${DRY_RUN:-0}" == 1 ]];then echo merged SFT preflight passed;exit 0;fi
"$SWIFT_BIN" sft --model "$MODEL" --adapters "$INIT_ADAPTER" --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni --custom_register_path "$ROOT_DIR/swift_plugin/aurora_qwen2_5_omni.py" --external_plugins "$ROOT_DIR/swift_plugin/sam_checkpoint.py" --dataset "$PADDED_TRAIN_JSONL" --strict true --new_special_tokens '[SEG]' --tuner_type lora --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample --freeze_vit true --freeze_aligner true --torch_dtype float16 --attn_impl sdpa --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_train_epochs "$EPOCHS" --max_steps -1 --max_length 5120 --truncation_strategy delete --max_pixels 200704 --learning_rate 1e-5 --weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 --max_grad_norm 1 --lr_scheduler_type cosine --warmup_ratio 0 --gradient_checkpointing true --logging_steps 10 --save_strategy epoch --save_total_limit 4 --eval_strategy no --dataloader_drop_last false --dataloader_num_workers 0 --dataset_num_proc 1 --split_dataset_ratio 0 --remove_unused_columns false --deepspeed "$ROOT_DIR/configs/deepspeed_zero2_fp16_v100.json" --callbacks aurora_sam_checkpoint --report_to tensorboard --add_version false --output_dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/train.log"
"$PYTHON_BIN" - "$OUTPUT_DIR" "$OUTPUT_DIR/completion.json" <<'PY'
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]); cps=sorted(root.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
if len(cps)<3:raise SystemExit(f'expected >=3 epoch checkpoints, found {[p.name for p in cps]}')
for cp in cps[-3:]:
 for name in ('adapter_model.safetensors','sam2_video_decoder.safetensors','aurora_projector_config.json','trainer_state.json'):
  if not (cp/name).is_file():raise SystemExit(f'{cp} missing {name}')
epochs=[float(json.loads((cp/"trainer_state.json").read_text())["epoch"]) for cp in cps[-3:]]
if any(abs(actual-expected)>0.01 for actual,expected in zip(epochs,(1,2,3))):
 raise SystemExit(f'Expected epoch 1/2/3 checkpoints, got {epochs}')
out.write_text(json.dumps({'status':'complete','checkpoints':[str(p.resolve()) for p in cps[-3:]],'epoch_checkpoints':len(cps[-3:])},indent=2)+'\n');print(out.read_text())
PY
