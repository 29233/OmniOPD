#!/usr/bin/env bash
# Scope-only ablation against merged_joint_online_20261103_v1.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/third_party/sam2_runtime_deps:$ROOT_DIR/third_party/sam2:$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
export SFT_CHECKPOINT="${SFT_CHECKPOINT:-$ROOT_DIR/outputs/train/sam2_merged_sft_3epoch_8gpu/checkpoint-4809}"
export DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/outputs/data/merged_student_only_online}"
export OUTPUT_DIR="${OUTPUT_DIR:?}"
export TRAIN_JSONL="${TRAIN_JSONL:-$DATA_ROOT/train.jsonl}"
export MAX_STEPS="${MAX_STEPS:-1000}" SAVE_STEPS="${SAVE_STEPS:-250}"
export TRAIN_SEED="${TRAIN_SEED:-42}" DATA_SEED="${DATA_SEED:-42}"
MODEL="${MODEL:?Set MODEL to the Qwen2.5-Omni-7B checkpoint path}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#DEVICES[@]}" == 8 ]] || { echo "Full-response Joint requires 8 GPUs"; exit 2; }
export AURORA_SEG_BACKEND=sam2
export AURORA_SAM2_CHECKPOINT="${AURORA_SAM2_CHECKPOINT:?Set AURORA_SAM2_CHECKPOINT to sam2_hiera_large.pt}"
export AURORA_SAM2_CONFIG="${AURORA_SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
export DATASET_VIDEO_FPS=1.0 FPS=1.0 AURORA_VIDEO_FPS=1.0 AURORA_NUM_SAM_FRAMES=10
export FPS_MAX_FRAMES=768 VIDEO_MAX_TOKEN_NUM=768 IMAGE_MAX_TOKEN_NUM=16384
export AURORA_PROJECTOR_HIDDEN_SIZE=1024 AURORA_PROJECTOR_LAYER_NORM=0
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0 TOKENIZERS_PARALLELISM=false AURORA_GRPO_REWARD_FAIL_OPEN=0
export AURORA_VERIFIED_TEACHERS=both AURORA_VERIFIED_KD_COEF=.25 AURORA_VERIFIED_TASK_REWARD_COEF=1
export AURORA_VERIFIED_TOKEN_FRACTION=1 AURORA_VERIFIED_TOKEN_RADIUS=0
export AURORA_VERIFIED_IOU_MARGIN=.01 AURORA_VERIFIED_CATEGORY_BONUS=2
export AURORA_VERIFIED_IOU_GAIN_WEIGHT=.75 AURORA_VERIFIED_ABSOLUTE_IOU_WEIGHT=.25
export AURORA_VERIFIED_ROUTE_TEMPERATURE=.25 AURORA_VERIFIED_K3_CLIP=10
export AURORA_ONLINE_TEACHER_TEMPERATURE=1 AURORA_ONLINE_TEACHER_TOP_P=.9
mkdir -p "$OUTPUT_DIR"
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "Full-response preflight passed; no model loaded"; exit 0; fi
MASTER_PORT="${MASTER_PORT:-29750}" NPROC_PER_NODE=8 "$SWIFT_BIN" rlhf \
  --rlhf_type grpo --model "$MODEL" --adapters "$SFT_CHECKPOINT" \
  --model_type aurora_qwen2_5_omni --template aurora_qwen2_5_omni \
  --custom_register_path "$ROOT_DIR/swift_plugin/aurora_qwen2_5_omni.py" \
  --external_plugins "$ROOT_DIR/swift_plugin/aurora_online_verified_opd.py" \
    "$ROOT_DIR/swift_plugin/aurora_full_response_opd.py" "$ROOT_DIR/swift_plugin/sam_checkpoint.py" \
  --reward_funcs aurora_class_exact aurora_sam2_mask_iou --reward_weights 0.5 0.5 \
  --dataset "$TRAIN_JSONL" --new_special_tokens '[SEG]' \
  --num_generations 8 --generation_batch_size 8 --temperature 1.0 --top_p 0.9 \
  --max_completion_length 256 --seed 42 --data_seed 42 --teacher_kl_coef 0 \
  --beta .01 --num_iterations 1 --dynamic_sample false \
  --tuner_type lora --target_regex '^thinker\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$' \
  --modules_to_save embed_tokens lm_head text_hidden_fcs sam_mask_decoder memory_attention memory_encoder obj_ptr_proj mask_downsample \
  --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --attn_impl sdpa \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps "$MAX_STEPS" --num_train_epochs 1 --max_length 5120 \
  --truncation_strategy delete --max_pixels 200704 --learning_rate 1e-7 \
  --weight_decay 0 --adam_beta1 .9 --adam_beta2 .95 --adam_epsilon 1e-8 \
  --max_grad_norm 1 --lr_scheduler_type cosine --warmup_ratio 0 \
  --gradient_checkpointing true --logging_steps 1 --log_completions true \
  --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit 5 \
  --eval_strategy no --dataloader_num_workers 0 --dataset_num_proc 1 \
  --train_dataloader_shuffle "${TRAIN_DATALOADER_SHUFFLE:-true}" \
  --split_dataset_ratio 0 --remove_unused_columns false \
  --callbacks aurora_sam_checkpoint --report_to tensorboard \
  --add_version false --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/train_$(date -u +%Y%m%d_%H%M%S).log"
