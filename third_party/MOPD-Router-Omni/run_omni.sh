#!/usr/bin/env bash
# Native upstream MOPD-Router; only Omni model/data/rollout I/O is adapted.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$REPO/../.." && pwd)"
if [[ "${OMNIOPD_LOAD_ENV:-0}" == "1" && -f "$PROJECT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT/.env"
  set +a
fi
OMNI_ENV="${OMNI_ENV:-$PROJECT/outputs/envs/mopd-router-omni-v1}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$OMNI_ENV/bin/python" ]]; then
    PYTHON_BIN="$OMNI_ENV/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
elif [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi
export PYTHON_BIN
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable PYTHON_BIN for MOPD-Router-Omni. Run bash third_party/MOPD-Router-Omni/bootstrap_omni_env.sh or set PYTHON_BIN." >&2
  exit 2
fi
export PYTHONPATH="$REPO:$REPO/verl"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0
export FPS=1.0 AURORA_VIDEO_FPS=1.0
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export STUDENT_MODEL="${STUDENT_MODEL:?Provide a merged standalone Thinker export}"
export TEACHER_BASE_MODEL="${TEACHER_BASE_MODEL:?Provide the shared pre-specialization Thinker export}"
export MATH_TEACHER_MODEL="${MATH_TEACHER_MODEL:?Provide independent frozen teacher export #1}"
export CODE_TEACHER_MODEL="${CODE_TEACHER_MODEL:?Provide independent frozen teacher export #2}"
export INSTRUCT_FOLLOWING_TEACHER_MODEL="${INSTRUCT_FOLLOWING_TEACHER_MODEL:?Provide independent frozen teacher export #3}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$PROJECT/outputs/data/merged_student_only_online_20261103_v1/train.jsonl}"
export OMNI_TIMING_AUDIT="${OMNI_TIMING_AUDIT:-$PROJECT/outputs/artifacts/refavs_fps_audit_20261025/audit.json}"
export CKPT_DIR="${CKPT_DIR:?Set a NEW output directory}"
export TEACHER_SIGNAL_AGGREGATION="${TEACHER_SIGNAL_AGGREGATION:-delta}"
case "$TEACHER_SIGNAL_AGGREGATION" in delta|entropy|accessible_novelty|mean) ;; *) exit 2 ;; esac
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MAX_PROMPT_LENGTH=4864 VAL_MAX_PROMPT_LENGTH=4864 MAX_RESPONSE_LENGTH=256
export ROLLOUT_MAX_MODEL_LEN=5120 ROLLOUT_TP_SIZE=1
export ROLLOUT_N=1 ROLLOUT_TEMPERATURE=1.0 ROLLOUT_TOP_P=1.0
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
# A debug dataloader is mandatory upstream even with validation disabled.
# Deliberately use training data, not test_s/test_u, for this disabled debug path.
export AIME24_VAL_PATH="$TRAIN_DATA_PATH" AIME25_VAL_PATH="$TRAIN_DATA_PATH"
export VALIDATION=false VAL_BEFORE_TRAIN=false TEST_FREQ=-1 WANDB_MODE=disabled
cd "$REPO"
"$PYTHON_BIN" -m omni_mopd.preflight --out "$CKPT_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
# Keep public run.sh and its native objective/router byte-identical upstream.
# Put required migration settings last: do not allow CLI overrides to silently
# re-enable text-only rollout, non-native losses, wrong media or reference paths.
bash "$REPO/run.sh" "$@" \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.ref.model.path="$MATH_TEACHER_MODEL" \
  actor_rollout_ref.ref.model.base_model_path="$CODE_TEACHER_MODEL" \
  actor_rollout_ref.ref.model.teacher_base_model_path="$TEACHER_BASE_MODEL" \
  actor_rollout_ref.ref.model.additional_teacher_model_paths.instruction_following="$INSTRUCT_FOLLOWING_TEACHER_MODEL" \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  data.train_files="$TRAIN_DATA_PATH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.model.use_fused_kernels=false \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_adapter_path=null \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=true \
  actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
  actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true \
  actor_rollout_ref.actor.policy_loss.entropy_aware_router=true \
  actor_rollout_ref.actor.policy_loss.teacher_signal_aggregation="$TEACHER_SIGNAL_AGGREGATION" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_liger=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  data.max_prompt_length=4864 \
  data.val_max_prompt_length=4864 \
  data.max_response_length=256 \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  algorithm.use_kl_in_reward=false \
  data.custom_cls.path="$REPO/omni_mopd/data.py" \
  data.custom_cls.name=OmniReasoningDataset \
  data.filter_overlong_prompts=false \
  data.dataloader_num_workers=0 \
  +data.omni_reasoning_only=true \
  +data.omni_timing_audit="$OMNI_TIMING_AUDIT" \
  +data.omni_model_path="$STUDENT_MODEL" \
  +data.omni_max_pixels=200704 \
  custom_reward_function.path="$REPO/omni_mopd/reward.py" \
  custom_reward_function.name=compute_score
