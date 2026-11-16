#!/usr/bin/env bash
# Category-first verified routing + verifiable reward + native signed OPD-RL.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

SWIFT_ENV="${SWIFT_ENV:-${CONDA_PREFIX:-/usr/local}}"
SWIFT_BIN="${SWIFT_BIN:-${SWIFT_ENV}/bin/swift}"
PYTHON_BIN="${PYTHON_BIN:-${SWIFT_ENV}/bin/python}"
MODEL="${MODEL:-models/Qwen2___5-Omni-7B}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-${ROOT_DIR}/outputs/train/qwen2_5_omni_7b_sam2_joint_sft_20261030_v2_3epoch/v0-20261029-184656/checkpoint-3825}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to prepare_verified_routed_opd_data.py output}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
DATA_MANIFEST="${DATA_MANIFEST:-${DATA_ROOT}/manifest.json}"
RUN_ID="${RUN_ID:-20261101_verified_routed_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/sam2_verified_routed_opd_${RUN_ID}}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_DEVICES[@]}}"
[[ "${NPROC_PER_NODE}" -eq 8 ]] || { echo "Verified routed OPD requires eight GPUs" >&2; exit 2; }

for required in "${SWIFT_BIN}" "${PYTHON_BIN}" "${MODEL}/config.json" \
  "${SFT_CHECKPOINT}/adapter_model.safetensors" "${SFT_CHECKPOINT}/adapter_config.json" \
  "${SFT_CHECKPOINT}/args.json" "${SFT_CHECKPOINT}/aurora_projector_config.json" \
  "${SFT_CHECKPOINT}/sam2_video_decoder.safetensors" "${TRAIN_JSONL}" \
  "${DATA_MANIFEST}" "${SAM2_CHECKPOINT}"; do
  [[ -e "${required}" ]] || { echo "Missing verified routed OPD requirement: ${required}" >&2; exit 1; }
done

TEACHER_ALIAS_ROOT="${ROOT_DIR}/outputs/.verified_opd_teacher_model_alias"
TEACHER_MODEL="${TEACHER_ALIAS_ROOT}/$(basename "${MODEL}")"
mkdir -p "${TEACHER_ALIAS_ROOT}"
if [[ -e "${TEACHER_MODEL}" && ! -L "${TEACHER_MODEL}" ]]; then
  echo "Refusing non-symlink Teacher alias: ${TEACHER_MODEL}" >&2
  exit 1
fi
ln -sfn "${MODEL}" "${TEACHER_MODEL}"

export PYTHONPATH="${ROOT_DIR}/third_party/sam2_runtime_deps:${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
export AURORA_SEG_BACKEND=sam2 AURORA_SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" AURORA_SAM2_CONFIG="${SAM2_CONFIG}"
export DATASET_VIDEO_FPS=1.0 FPS=1.0 AURORA_VIDEO_FPS=1.0
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384
export AURORA_NUM_SAM_FRAMES=10 AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false
export AURORA_GRPO_REWARD_FAIL_OPEN=0

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
REFERENCE_KL_BETA="${REFERENCE_KL_BETA:-0.01}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
MASTER_PORT="${MASTER_PORT:-29621}"

export AURORA_VERIFIED_IOU_MARGIN="${AURORA_VERIFIED_IOU_MARGIN:-0.01}"
export AURORA_VERIFIED_CATEGORY_BONUS="${AURORA_VERIFIED_CATEGORY_BONUS:-2.0}"
export AURORA_VERIFIED_IOU_GAIN_WEIGHT="${AURORA_VERIFIED_IOU_GAIN_WEIGHT:-0.75}"
export AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT="${AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT:-0.25}"
export AURORA_VERIFIED_ROUTE_TEMPERATURE="${AURORA_VERIFIED_ROUTE_TEMPERATURE:-0.25}"
export AURORA_VERIFIED_K3_CLIP="${AURORA_VERIFIED_K3_CLIP:-10}"
export AURORA_VERIFIED_KD_COEF="${AURORA_VERIFIED_KD_COEF:-0.25}"
export AURORA_VERIFIED_TOKEN_FRACTION="${AURORA_VERIFIED_TOKEN_FRACTION:-0.10}"
export AURORA_VERIFIED_TOKEN_RADIUS="${AURORA_VERIFIED_TOKEN_RADIUS:-1}"
export AURORA_VERIFIED_TEACHERS="${AURORA_VERIFIED_TEACHERS:-both}"
export AURORA_VERIFIED_TASK_REWARD_COEF="${AURORA_VERIFIED_TASK_REWARD_COEF:-1.0}"
export MODEL SFT_CHECKPOINT TEACHER_MODEL TRAIN_JSONL OUTPUT_DIR
export MAX_STEPS LEARNING_RATE REFERENCE_KL_BETA NUM_GENERATIONS MAX_COMPLETION_LENGTH

mkdir -p "${OUTPUT_DIR}"
REQUIRE_ALL_MODALITIES="${REQUIRE_ALL_MODALITIES:-1}"
"${PYTHON_BIN}" - "${OUTPUT_DIR}/training_manifest.json" "${DATA_MANIFEST}" "${REQUIRE_ALL_MODALITIES}" <<'PY'
import json,os,sys
from pathlib import Path
out,data_path=map(Path,sys.argv[1:3]); require_all=sys.argv[3]=="1"
data=json.loads(data_path.read_text(encoding="utf-8"))
rows=[json.loads(line) for line in Path(os.environ["TRAIN_JSONL"]).open(encoding="utf-8") if line.strip()]
counts={}
for row in rows:
    modality=str((row.get("template_match") or {}).get("template_reference") or "")
    if modality not in {"Audio","Audio-Visual","Visual","Semantic"}:
        raise SystemExit(f"Missing/unknown template modality in {row.get('sample_id')}: {modality!r}")
    counts[modality]=counts.get(modality,0)+1
if require_all and any(counts.get(name,0)==0 for name in ("Audio","Audio-Visual","Visual","Semantic")):
    raise SystemExit(f"Verified OPD data must contain all four modalities; got {counts}")
required={
    "audio_teacher_completion","audio_teacher_class_correct","audio_teacher_iou",
    "visual_teacher_completion","visual_teacher_class_correct","visual_teacher_iou",
    "difficulty_bucket","audio_privilege_text","visual_privilege_text",
}
first=json.loads(next(line for line in Path(os.environ["TRAIN_JSONL"]).open(encoding="utf-8") if line.strip()))
missing=required-set(first)
if missing: raise SystemExit(f"Verified data row misses {sorted(missing)}")
manifest={
 "schema":1,
 "purpose":("joint category-gated SAM2 GRPO and verified two-teacher sampled reverse-KL OPD"
            if float(os.environ["AURORA_VERIFIED_TASK_REWARD_COEF"]) else
            "pure verified two-teacher sampled reverse-KL OPD; verifier only licenses Teachers"),
 "initializer":str(Path(os.environ["SFT_CHECKPOINT"]).resolve()),
 "teacher":{"model":str(Path(os.environ["TEACHER_MODEL"]).resolve()),
            "adapter":str(Path(os.environ["SFT_CHECKPOINT"]).resolve()),
            "probes":"cached one audio and one visual trajectory per prompt",
            "token_scoring":"same frozen Teacher scores Student sampled tokens under each privileged prompt"},
 "license":{"category":"Teacher GT-correct and Student GT-wrong",
            "mask":"both GT-correct and Teacher IoU exceeds Student by margin",
            "iou_margin":float(os.environ["AURORA_VERIFIED_IOU_MARGIN"])},
 "routing":{"category_rescue_bonus":float(os.environ["AURORA_VERIFIED_CATEGORY_BONUS"]),
            "iou_gain_weight":float(os.environ["AURORA_VERIFIED_IOU_GAIN_WEIGHT"]),
            "absolute_iou_weight":float(os.environ["AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT"]),
            "temperature":float(os.environ["AURORA_VERIFIED_ROUTE_TEMPERATURE"]),
            "overlap":"hard one-Teacher-per-token"},
 "task_reward":{"formula":"0.5*class_exact + 0.5*class_exact*mean_10_frame_iou",
                "policy_gradient_coefficient":float(os.environ["AURORA_VERIFIED_TASK_REWARD_COEF"]),
                "native_dynamic_sample":False,
                "difficulty_mixture":data.get("realized_ratios")},
 "distillation":{"direction":"sampled-token reverse KL KL(student||teacher)",
                 "optimization_signal":"ms-swift native signed k1: logp_teacher-logp_student",
                 "monitoring_estimator":"k3=exp(logp_teacher-logp_student)-delta-1",
                 "monitoring_k3_clip":float(os.environ["AURORA_VERIFIED_K3_CLIP"]),
                 "kd_coef":float(os.environ["AURORA_VERIFIED_KD_COEF"]),
                 "enabled_teachers":os.environ["AURORA_VERIFIED_TEACHERS"],
                 "token_fraction":float(os.environ["AURORA_VERIFIED_TOKEN_FRACTION"]),
                 "radius":int(os.environ["AURORA_VERIFIED_TOKEN_RADIUS"]),
                 "teacher_stages":{"audio":["Audio evidence","Cross-modal reasoning"],
                                   "visual":["Visual evidence","Cross-modal reasoning"]},
                 "modality_routing":{
                   "Audio":"audio teacher on Audio evidence; visual teacher on Visual evidence",
                   "Audio-Visual":"audio/visual teachers on their respective evidence and cross-modal stages",
                   "Visual":"audio teacher disabled; visual teacher on Visual evidence only",
                   "Semantic":"audio teacher disabled; visual teacher on Visual evidence only"}},
 "modality_counts":counts,
 "trainable":["q/v LoRA only"],
 "optimization":{"max_steps":int(os.environ["MAX_STEPS"]),
                 "learning_rate":float(os.environ["LEARNING_RATE"]),
                 "reference_kl_beta":float(os.environ["REFERENCE_KL_BETA"]),
                 "num_generations":int(os.environ["NUM_GENERATIONS"]),
                 "max_completion_length":int(os.environ["MAX_COMPLETION_LENGTH"]),
                 "dtype":"bfloat16"},
 "media":{"dataset_nominal_fps":1.0,"model_timeline_fps":1.0,
          "frames":10,"max_pixels":200704,"max_length":5120},
 "data_manifest":str(data_path.resolve()),
}
out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: verified routed OPD preflight passed"
  exit 0
fi

TRAIN_LOG="${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log"
MASTER_PORT="${MASTER_PORT}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" --adapters "${SFT_CHECKPOINT}" \
  --teacher_model "${TEACHER_MODEL}" --teacher_adapters "${SFT_CHECKPOINT}" \
  --teacher_model_type aurora_qwen2_5_omni --teacher_kl_coef 0 \
  --offload_teacher_model true --offload_optimizer true \
  --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${ROOT_DIR}/swift_plugin/aurora_verified_routed_opd.py" \
    "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --reward_funcs aurora_class_exact aurora_sam2_mask_iou \
  --reward_weights 0.5 0.5 \
  --dataset "${TRAIN_JSONL}" --new_special_tokens '[SEG]' \
  --num_generations "${NUM_GENERATIONS}" --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" --top_p "${TOP_P}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --beta "${REFERENCE_KL_BETA}" --num_iterations 1 \
  --dynamic_sample false \
  --tuner_type lora \
  --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps "${MAX_STEPS}" --num_train_epochs 1 \
  --max_length 5120 --truncation_strategy delete --max_pixels 200704 \
  --learning_rate "${LEARNING_RATE}" --weight_decay 0.0 \
  --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 --max_grad_norm 1.0 \
  --lr_scheduler_type cosine --warmup_ratio 0 \
  --gradient_checkpointing true --logging_steps 1 --log_completions true \
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy no --dataloader_num_workers 0 --dataset_num_proc 1 \
  --split_dataset_ratio 0 --remove_unused_columns false \
  --callbacks aurora_sam_checkpoint --report_to tensorboard \
  --add_version false --output_dir "${OUTPUT_DIR}" 2>&1 | tee "${TRAIN_LOG}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/completion.json" <<'PY'
import json,os
from pathlib import Path
root=Path(os.environ["OUTPUT_DIR"])
states=list(root.glob("checkpoint-*/trainer_state.json"))+list(root.glob("v*/checkpoint-*/trainer_state.json"))
if not states: raise SystemExit("No verified routed OPD checkpoint was saved")
state_path=max(states,key=lambda p:int(p.parent.name.rsplit("-",1)[-1]))
state=json.loads(state_path.read_text())
checkpoint=state_path.parent
Path(os.environ["OUTPUT_DIR"],"completion.json").write_text(json.dumps({
 "status":"complete","checkpoint":str(checkpoint.resolve()),
 "global_step":state["global_step"],
},indent=2)+"\n")
PY
echo "verified_routed_opd_complete=${OUTPUT_DIR}/completion.json"
