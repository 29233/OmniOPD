#!/usr/bin/env bash
# Reward-only matched ablation for verified routed OPD.
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
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to refreshed verified-OPD data}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
DATA_MANIFEST="${DATA_MANIFEST:-${DATA_ROOT}/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/sam2_verified_reward_grpo_20261101_v1}"
TRAINER_PLUGIN="${TRAINER_PLUGIN:-${ROOT_DIR}/swift_plugin/aurora_verified_reward_grpo.py}"
export TRAINER_PLUGIN TRAIN_JSONL
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-models/sam2_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#DEVICES[@]}" -eq 8 ]] || { echo "Reward-only GRPO requires eight GPUs" >&2; exit 2; }
export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
export AURORA_SEG_BACKEND=sam2 AURORA_SAM2_CHECKPOINT="${SAM2_CHECKPOINT}" AURORA_SAM2_CONFIG="${SAM2_CONFIG}"
export DATASET_VIDEO_FPS="${DATASET_VIDEO_FPS:-1.0}"
export FPS="${FPS:-1.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}" VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}" IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export AURORA_NUM_SAM_FRAMES=10 AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false AURORA_GRPO_REWARD_FAIL_OPEN="${AURORA_GRPO_REWARD_FAIL_OPEN:-0}"

MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
REFERENCE_KL_BETA="${REFERENCE_KL_BETA:-0.01}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATA_SEED="${DATA_SEED:-${TRAIN_SEED}}"
MASTER_PORT="${MASTER_PORT:-29634}"
export SFT_CHECKPOINT MAX_STEPS LEARNING_RATE REFERENCE_KL_BETA NUM_GENERATIONS MAX_COMPLETION_LENGTH TRAIN_SEED DATA_SEED

for path in "${PYTHON_BIN}" "${SWIFT_BIN}" "${MODEL}/config.json" "${SFT_CHECKPOINT}/adapter_model.safetensors" \
  "${SFT_CHECKPOINT}/sam2_video_decoder.safetensors" "${TRAIN_JSONL}" \
  "${DATA_MANIFEST}" "${SAM2_CHECKPOINT}"; do
  [[ -n "${path}" && -e "${path}" ]] || { echo "Missing reward-only requirement: ${path:-<empty>}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" - "${OUTPUT_DIR}/training_manifest.json" "${DATA_MANIFEST}" <<'PY'
import json,os,sys
from pathlib import Path
out,data=map(Path,sys.argv[1:])
manifest={
 "schema":1,
 "purpose":"matched category-gated SAM2 GRPO reward-only ablation",
 "initializer":str(Path(os.environ["SFT_CHECKPOINT"]).resolve()),
 "data_manifest":str(data.resolve()),
 "task_reward":{"formula":"0.5*class_exact + 0.5*class_exact*mean_10_frame_iou"},
 "teacher":None,"distillation":None,"trainable":["q/v LoRA only"],
 "optimization":{"max_steps":int(os.environ["MAX_STEPS"]),
                 "learning_rate":float(os.environ["LEARNING_RATE"]),
                 "reference_kl_beta":float(os.environ["REFERENCE_KL_BETA"]),
                 "num_generations":int(os.environ["NUM_GENERATIONS"]),
                 "max_completion_length":int(os.environ["MAX_COMPLETION_LENGTH"]),
                 "seed":int(os.environ["TRAIN_SEED"]),"data_seed":int(os.environ["DATA_SEED"]),
                 "dtype":"bfloat16"},
 "media":{"dataset_nominal_fps":1.0,"model_timeline_fps":1.0,
          "frames":10,"max_pixels":200704,"max_length":5120},
}
if Path(os.environ["TRAINER_PLUGIN"]).name=="aurora_online_verified_opd.py":
    coefficient=float(os.environ.get("AURORA_VERIFIED_TASK_REWARD_COEF","1"))
    manifest.update(
      purpose="online current-policy privileged joint" if coefficient else "online current-policy privileged pure OPD",
      teacher={"source":"current Student policy; adapters enabled; stop-gradient",
               "refresh":"each rollout/update","offline_teacher_cache":False,
               "sampling":{"temperature":float(os.environ.get("AURORA_ONLINE_TEACHER_TEMPERATURE","1")),
                           "top_p":float(os.environ.get("AURORA_ONLINE_TEACHER_TOP_P","0.9")),
                           "trajectories_per_view":1}},
      distillation={"direction":"sampled reverse KL(student||current-policy privileged Teacher)",
                    "signal":"ms-swift native compute_teacher_logratio",
                    "kd_coef":float(os.environ.get("AURORA_VERIFIED_KD_COEF",".25")),
                    "fraction":float(os.environ.get("AURORA_VERIFIED_TOKEN_FRACTION",".1")),
                    "radius":int(os.environ.get("AURORA_VERIFIED_TOKEN_RADIUS","1")),
                    "license":"online category rescue or both-correct IoU gain > 0.01",
                    "modality":"Audio Teacher disabled for Visual/Semantic"},
    )
    manifest["task_reward"]["policy_gradient_coefficient"]=coefficient
if Path(os.environ["TRAINER_PLUGIN"]).name=="aurora_mask_primary.py":
    arm=os.environ["AURORA_MASK_PRIMARY_ARM"]
    alpha=float(os.environ.get("AURORA_MASK_CLASS_ALPHA",".2"))
    manifest.update(
      purpose="mask-primary reward-only GRPO" if arm=="grpo" else "mask-primary GRPO plus verified Decision-category online OPD",
      experiment="mask_primary_"+arm,
      teacher=None if arm=="grpo" else {
        "source":"current Student policy; adapters enabled; stop-gradient",
        "refresh":"each rollout/update","offline_teacher_cache":False,
        "sampling":{"temperature":1.0,"top_p":.9,"trajectories_per_view":1}},
      distillation=None if arm=="grpo" else {
        "direction":"sampled reverse KL(student||current-policy privileged Teacher)",
        "signal":"ms-swift native compute_teacher_logratio","kd_coef":.25,
        "tokens":"BPE tokens overlapping Decision category span only; no evidence/header/SEG/EOS",
        "selection":"all category tokens; no lowest-likelihood truncation",
        "license":"class rescue only; valid teacher SEG and minimum IoU, no material mask regression",
        "min_teacher_iou":float(os.environ.get("AURORA_DECISION_MIN_IOU",".5")),
        "mask_regression_tolerance":float(os.environ.get("AURORA_DECISION_IOU_TOLERANCE",".01")),
        "modality":"Audio Teacher disabled for Visual/Semantic"},
    )
    manifest["task_reward"]={"formula":"valid_SEG*(alpha*class_exact+(1-alpha)*mean_10_frame_iou)",
                             "alpha":alpha,"native_reward_weights":[.5,.5],
                             "column_scaling":[2*alpha,2*(1-alpha)],
                             "class_gate_on_iou":False,"LM_judge":False}
manifest["training_jsonl"]=str(Path(os.environ["TRAIN_JSONL"]).resolve())
manifest["data_sha256"]=__import__("hashlib").sha256(Path(os.environ["TRAIN_JSONL"]).read_bytes()).hexdigest()
manifest["media"].update(fps_max_frames=768,video_max_token_num=768,image_max_token_num=16384,
                         exact_frame_sampling_instants_verified=False)
manifest["intended_training_differences_from_sft"]=["bf16 post-training vs fp16 SFT; stochastic online rollouts vs teacher forcing; q/v LoRA only"]
out.write_text(json.dumps(manifest,indent=2)+"\n")
PY
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN=1: verified reward GRPO preflight passed"; exit 0; fi

MASTER_PORT="${MASTER_PORT}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE=8 \
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo --model "${MODEL}" --adapters "${SFT_CHECKPOINT}" \
  --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni \
  --custom_register_path "${ROOT_DIR}/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "${TRAINER_PLUGIN}" \
    "${ROOT_DIR}/swift_plugin/sam_checkpoint.py" \
  --reward_funcs aurora_class_exact aurora_sam2_mask_iou --reward_weights 0.5 0.5 \
  --dataset "${TRAIN_JSONL}" --new_special_tokens '[SEG]' \
  --num_generations "${NUM_GENERATIONS}" --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --temperature 1.0 --top_p 0.9 --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --seed "${TRAIN_SEED}" --data_seed "${DATA_SEED}" \
  --teacher_kl_coef 0 \
  --beta "${REFERENCE_KL_BETA}" --num_iterations 1 --dynamic_sample false \
  --tuner_type lora --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample \
  --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps "${MAX_STEPS}" --num_train_epochs 1 --max_length 5120 \
  --truncation_strategy delete --max_pixels 200704 --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_epsilon 1e-8 \
  --max_grad_norm 1 --lr_scheduler_type cosine --warmup_ratio 0 \
  --gradient_checkpointing true --logging_steps 1 --log_completions true \
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --eval_strategy no --dataloader_num_workers 0 --dataset_num_proc 1 \
  --train_dataloader_shuffle "${TRAIN_DATALOADER_SHUFFLE:-true}" \
  --split_dataset_ratio 0 --remove_unused_columns false \
  --callbacks aurora_sam_checkpoint --report_to tensorboard \
  --add_version false --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train_$(date -u +%Y%m%d_%H%M%S).log"
