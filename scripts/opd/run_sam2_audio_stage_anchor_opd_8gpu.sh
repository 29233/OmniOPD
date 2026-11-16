#!/usr/bin/env bash
# Short task-aware follow-up after the dual audio/visual sparse OPD diagnosis.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_ID="${RUN_ID:-20261031_v1}"
BASE_MODEL="${BASE_MODEL:-models/Qwen2___5-Omni-7B}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_joint_sft_20261030_v2_3epoch/v0-20261029-184656/checkpoint-3825}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/outputs/data/privilege_top10_full_20261031_full_v1}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_ROOT}/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/sam2_audio_stage_anchor_opd_${RUN_ID}}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
MAX_STEPS="${MAX_STEPS:-300}"
LEARNING_RATE="${LEARNING_RATE:-2e-7}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#GPU_DEVICES[@]}" -eq 8 ]] || { echo "Audio-stage anchor OPD requires eight GPUs" >&2; exit 2; }

for required in "${PYTHON_BIN}" "${BASE_MODEL}/config.json" \
  "${CHECKPOINT}/adapter_model.safetensors" "${CHECKPOINT}/args.json" \
  "${CHECKPOINT}/aurora_projector_config.json" "${CHECKPOINT}/sam2_video_decoder.safetensors" \
  "${TRAIN_JSONL}" "${VAL_JSONL}" "${DATA_ROOT}/manifest.json" "${SAM2_CHECKPOINT}"; do
  [[ -e "${required}" ]] || { echo "Missing audio-stage anchor requirement: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"

export BASE_MODEL STAGE1_ADAPTERS="${CHECKPOINT}" TEACHER_MODEL="${BASE_MODEL}" TEACHER_ADAPTERS="${CHECKPOINT}"
export TRAIN_JSONL VAL_JSONL OUTPUT_DIR NPROC_PER_NODE=8
export AURORA_SEG_BACKEND=sam2 AURORA_SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" AURORA_SAM2_CONFIG="${SAM2_CONFIG}"
export DATASET_VIDEO_FPS=1.0 FPS=1.0 AURORA_VIDEO_FPS=1.0 AURORA_NUM_SAM_FRAMES=10
export AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false
export AURORA_OPSD_PLUGIN="${ROOT_DIR}/swift_plugin/aurora_audio_stage_anchor_opd.py"
export AURORA_OPSD_TEACHER_MODE=frozen AURORA_OPSD_REQUIRE_FROZEN_TEACHER=1
export AURORA_OPSD_MASK_LOSS=0 AURORA_OPSD_MASK_WEIGHT=0 AURORA_OPSD_SEG_DISTILL_WEIGHT=0
export AURORA_AUDIO_STAGE_FRACTION="${AURORA_AUDIO_STAGE_FRACTION:-0.10}"
export AURORA_AUDIO_STAGE_RADIUS="${AURORA_AUDIO_STAGE_RADIUS:-1}"
export AURORA_AUDIO_STAGE_ANCHOR_WEIGHT="${AURORA_AUDIO_STAGE_ANCHOR_WEIGHT:-0.5}"
export AURORA_AUDIO_STAGE_ANCHOR_MODE="${AURORA_AUDIO_STAGE_ANCHOR_MODE:-stages}"
export AURORA_AUDIO_STAGE_NAMES="${AURORA_AUDIO_STAGE_NAMES:-audio_evidence,cross_modal_reasoning}"
export AURORA_ANCHOR_STAGE_NAMES="${AURORA_ANCHOR_STAGE_NAMES:-decision,decision_target,segmentation,seg_token}"
export LMBDA=1 BETA=0 TEMPERATURE=1 MAX_COMPLETION_LENGTH=256
export MAX_STEPS NUM_TRAIN_EPOCHS=1 LEARNING_RATE
export SAVE_STEPS="${SAVE_STEPS:-100}" EVAL_STEPS="${EVAL_STEPS:-100}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
export MAX_SAMPLES=-1 VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
export TORCH_DTYPE=bfloat16 MAX_LENGTH=5120 MAX_PIXELS=200704
export OFFLOAD_TEACHER_MODEL=true OFFLOAD_OPTIMIZER=true LOG_COMPLETIONS=false REPORT_TO=tensorboard

"${PYTHON_BIN}" - "${OUTPUT_DIR}/training_manifest.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
out=Path(os.environ["OUTPUT_DIR"])/"training_manifest.json"
checkpoint=Path(os.environ["STAGE1_ADAPTERS"]).resolve()
plugin=Path(os.environ["AURORA_OPSD_PLUGIN"]).resolve()
args=json.loads((checkpoint/"args.json").read_text())
projector=json.loads((checkpoint/"aurora_projector_config.json").read_text())
if args.get("torch_dtype")!="float16" or args.get("max_length")!=5120 or args.get("max_pixels")!=200704:
    raise SystemExit("Initializer settings mismatch")
if projector.get("backend")!="sam2_video_memory" or projector.get("model_timeline_fps")!=1.0 or projector.get("sam_frames")!=10:
    raise SystemExit("Initializer SAM2 protocol mismatch")
manifest={
  "schema":1,
  "purpose":"task-aware audio-only stage-localized OPD with frozen decision/SEG anchor",
  "initializer":str(checkpoint),
  "teacher":{"checkpoint":str(checkpoint),"frozen":True,
             "audio_view":"clip-level task-aware audio evidence",
             "anchor_view":"ordinary unprivileged prompt"},
  "objective":{"audio_teacher":"bottom 10% likelihood within Audio evidence and Cross-modal reasoning, dilated +/-1",
               "anchor":(
                   "full-vocabulary Forward KL on every response token outside the audio-selected mask"
                   if os.environ["AURORA_AUDIO_STAGE_ANCHOR_MODE"]=="complement"
                   else "full-vocabulary Forward KL on Decision and Segmentation tokens"
               ),
               "anchor_weight":float(os.environ["AURORA_AUDIO_STAGE_ANCHOR_WEIGHT"]),
               "anchor_mode":os.environ["AURORA_AUDIO_STAGE_ANCHOR_MODE"],
               "audio_stage_names":os.environ["AURORA_AUDIO_STAGE_NAMES"].split(","),
               "anchor_stage_names":os.environ["AURORA_ANCHOR_STAGE_NAMES"].split(","),
               "mask_loss":0.0,"seg_feature_loss":0.0},
  "implementation":{"plugin":str(plugin),
                    "plugin_sha256":hashlib.sha256(plugin.read_bytes()).hexdigest(),
                    "strict_stage_key_validation":True,
                    "optional_stages_may_be_absent_per_response":True},
  "trainable":["q/v LoRA only"],
  "frozen":["embed_tokens","lm_head","SEG projector","SAM2 decoder and memory","vision/audio towers","teacher"],
  "optimization":{"max_steps":int(os.environ["MAX_STEPS"]),"learning_rate":float(os.environ["LEARNING_RATE"]),
                  "scheduler":"cosine","dtype":"bfloat16","global_batch":8,
                  "max_length":5120,"max_completion_length":256},
  "media":{"dataset_nominal_fps":1.0,"model_fps":1.0,"frames":10,"max_pixels":200704},
  "motivation":["drop visual teacher because its paired direct privilege effect was not positive",
                "localize audio supervision to the stages it is expected to improve",
                "anchor final target binding and SEG syntax to the frozen SFT policy",
                "freeze large embedding/lm-head modules to limit destructive global drift"],
}
out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: audio-stage anchor OPD preflight passed"
  exit 0
fi

bash "${ROOT_DIR}/scripts/opd/train_opsd_swift.sh"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/completion.json" <<'PY'
import json
import os
from pathlib import Path
root=Path(os.environ["OUTPUT_DIR"])
states=list(root.glob("v*/checkpoint-*/trainer_state.json"))
if not states: raise SystemExit("No audio-stage anchor checkpoint was saved")
state_path=max(states,key=lambda path:int(path.parent.name.rsplit("-",1)[-1]))
state=json.loads(state_path.read_text())
if int(state.get("global_step",-1))!=int(os.environ["MAX_STEPS"]):
    raise SystemExit(f"Training incomplete: {state.get('global_step')}/{os.environ['MAX_STEPS']}")
checkpoint=state_path.parent
for name in ("adapter_model.safetensors","aurora_projector_config.json","sam2_video_decoder.safetensors"):
    if not (checkpoint/name).is_file(): raise SystemExit(f"Final checkpoint lacks {name}")
(checkpoint/"training_manifest.json").write_text((root/"training_manifest.json").read_text())
Path(os.environ["OUTPUT_DIR"],"completion.json").write_text(json.dumps({
    "status":"complete","checkpoint":str(checkpoint.resolve()),"global_step":state["global_step"],
},indent=2)+"\n")
PY
echo "audio_stage_anchor_complete=${OUTPUT_DIR}/completion.json"
