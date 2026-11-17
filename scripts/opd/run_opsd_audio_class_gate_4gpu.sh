#!/usr/bin/env bash
# One current Student, two privileged views; platform foreground entrypoint.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# TRL generates a model card at every checkpoint. Its bundled lm_model_card.md
# contains UTF-8 author text, while some platform workers start with the C/ASCII
# locale. Force Python's UTF-8 mode for all DDP ranks so checkpoint saves cannot
# fail in ModelCard.from_template().
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
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
export PYTHONPATH="$ROOT_DIR/third_party/sam2:$ROOT_DIR:${PYTHONPATH:-}"
unset MS_SWIFT_ROOT
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/opsd_audio_class_gate_mute50_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/launcher.log") 2>&1
export TOKENIZERS_PARALLELISM=false
export ENABLE_AUDIO_OUTPUT=0 USE_AUDIO_IN_VIDEO=0
# Legacy Qwen2.5-Omni-3B Stage-1 audio/class MOPD runs used model timeline 2.0 FPS;
# overridable via FPS / AURORA_VIDEO_FPS while preserving the legacy 2.0 default.
export FPS="${FPS:-2.0}"
export AURORA_VIDEO_FPS="${AURORA_VIDEO_FPS:-${FPS}}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-768}" VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-768}" IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export AURORA_NUM_SAM_FRAMES="${AURORA_NUM_SAM_FRAMES:-10}"
export AURORA_OPSD_TEACHER_MODE="${AURORA_OPSD_TEACHER_MODE:-frozen}"
export AURORA_OPSD_REQUIRE_FROZEN_TEACHER="${AURORA_OPSD_REQUIRE_FROZEN_TEACHER:-1}"
export AURORA_OPSD_TEACHER_AUDIO="${AURORA_OPSD_TEACHER_AUDIO:-}"
export AURORA_OPSD_TEACHER_MASK_IMAGE="${AURORA_OPSD_TEACHER_MASK_IMAGE:-0}"
export AURORA_OPSD_SEG_DISTILL_WEIGHT="${AURORA_OPSD_SEG_DISTILL_WEIGHT:-0}"
export AURORA_OPSD_MASK_LOSS="${AURORA_OPSD_MASK_LOSS:-1}" AURORA_OPSD_MASK_WEIGHT="${AURORA_OPSD_MASK_WEIGHT:-1}"
export AURORA_OPSD_PLUGIN="${AURORA_OPSD_PLUGIN:-$ROOT_DIR/swift_plugin/aurora_dual_opsd.py}"
export LMBDA="${LMBDA:-1}" BETA="${BETA:-0}" TEMPERATURE="${TEMPERATURE:-1}"
export MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export LOG_COMPLETIONS="${LOG_COMPLETIONS:-true}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
export OFFLOAD_TEACHER_MODEL="${OFFLOAD_TEACHER_MODEL:-true}"
export OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"
export AURORA_DUAL_CLASS_THRESHOLD="${AURORA_DUAL_CLASS_THRESHOLD:-0.02}"
export AURORA_DUAL_AUDIO_THRESHOLD="${AURORA_DUAL_AUDIO_THRESHOLD:-0.02}"
export AURORA_DUAL_CLASS_SCALE="${AURORA_DUAL_CLASS_SCALE:-0.1}"
export AURORA_DUAL_AUDIO_SCALE="${AURORA_DUAL_AUDIO_SCALE:-0.1}"
export AURORA_DUAL_WEIGHT_CAP="${AURORA_DUAL_WEIGHT_CAP:-2}"
export AURORA_DUAL_BASE_WEIGHT="${AURORA_DUAL_BASE_WEIGHT:-1}"
export AURORA_DUAL_GATE_WEIGHT="${AURORA_DUAL_GATE_WEIGHT:-1}"
export AURORA_DUAL_TEMPERATURE="${AURORA_DUAL_TEMPERATURE:-1}"
export AURORA_DUAL_CHUNK_SIZE="${AURORA_DUAL_CHUNK_SIZE:-32}"
# Audio degradation is configurable for controlled robustness experiments.
# The historical default remains 50% in 100 ms blocks. A new launcher can set
# MUTE_FRACTION/MUTE_BLOCK_MS and use a separate cache/output filename.
export MUTE_FRACTION="${MUTE_FRACTION:-0.5}"
export MUTE_BLOCK_MS="${MUTE_BLOCK_MS:-100}"
export MUTE_CONDITION="${MUTE_CONDITION:-mute50}"
export MUTE_OUTPUT_NAME="${MUTE_OUTPUT_NAME:-audio_opsd_random_mute50.wav}"
export MUTE_METADATA="${MUTE_METADATA:-$ROOT_DIR/outputs/data/audio_degraded/opsd_mute50.json}"
# This launcher owns smoke truncation; the generic launcher must not truncate again.
PREP_TRAIN_LIMIT="${DUAL_TRAIN_SAMPLES:--1}"
PREP_VAL_LIMIT="${DUAL_VAL_SAMPLES:-64}"
export MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1
_DEFAULT_RAW_TRAIN="$ROOT_DIR/outputs/data/refavs/refavs_full.jsonl"
if [[ ! -f "$_DEFAULT_RAW_TRAIN" && -f "$ROOT_DIR/outputs/data/refavs/refavs_train.jsonl" ]]; then
  _DEFAULT_RAW_TRAIN="$ROOT_DIR/outputs/data/refavs/refavs_train.jsonl"
fi
SOURCE_TRAIN_JSONL="${SOURCE_TRAIN_JSONL:-${TRAIN_JSONL:-$_DEFAULT_RAW_TRAIN}}"
SOURCE_VAL_JSONL="${SOURCE_VAL_JSONL:-${VAL_JSONL:-$ROOT_DIR/outputs/data/refavs/refavs_val.jsonl}}"
PREP_DATA_DIR="${PREP_DATA_DIR:-$OUTPUT_DIR/data}"
if [[ "${SKIP_DATA_PREP:-0}" == "1" ]]; then
  TRAIN_JSONL="${TRAIN_JSONL:-$SOURCE_TRAIN_JSONL}"
  VAL_JSONL="${VAL_JSONL:-$SOURCE_VAL_JSONL}"
  export TRAIN_JSONL VAL_JSONL
  for required in "$PYTHON_BIN" "$SWIFT_BIN" "$TRAIN_JSONL" "$VAL_JSONL"; do
    [[ -n "$required" && -e "$required" ]] || { echo "Missing dual OPSD requirement: ${required:-<empty>}" >&2; exit 1; }
  done
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: dual audio/class OPSD preflight passed (SKIP_DATA_PREP=1)"
    exit 0
  fi
else
  for required in "$PYTHON_BIN" "$SWIFT_BIN" "$SOURCE_TRAIN_JSONL" "$SOURCE_VAL_JSONL"; do
    [[ -n "$required" && -e "$required" ]] || { echo "Missing dual OPSD source requirement: ${required:-<empty>}" >&2; exit 1; }
  done
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: dual audio/class OPSD preflight passed (source_train=$SOURCE_TRAIN_JSONL, source_val=$SOURCE_VAL_JSONL, FPS=$FPS)"
    exit 0
  fi
  mkdir -p "$PREP_DATA_DIR"
  SELECT_ARGS=()
  if [[ "${DUAL_SMOKE:-0}" == 1 ]]; then SELECT_ARGS=(--include-null); fi
  for split in train val; do
    if [[ "$split" == train ]]; then
      source="$SOURCE_TRAIN_JSONL"
      limit="$PREP_TRAIN_LIMIT"
    else
      source="$SOURCE_VAL_JSONL"
      limit="$PREP_VAL_LIMIT"
    fi
    "$PYTHON_BIN" scripts/data/prepare_dual_opsd_audio_class.py --source "$source" \
      --output "$PREP_DATA_DIR/${split}_raw.jsonl" --max-samples "$limit" --selection-only "${SELECT_ARGS[@]}"
  done
  "$PYTHON_BIN" scripts/data/prepare_opsd_audio_condition.py --condition "$MUTE_CONDITION" \
    --seed "${MUTE_SEED:-20261015}" --fraction "$MUTE_FRACTION" --block-ms "$MUTE_BLOCK_MS" \
    --output-name "$MUTE_OUTPUT_NAME" \
    --metadata "$MUTE_METADATA" \
    --recover-orphan-cache \
    --preserve-cache-records \
    --manifest "$PREP_DATA_DIR/train_raw.jsonl" --manifest "$PREP_DATA_DIR/val_raw.jsonl"
  for split in train val; do
    "$PYTHON_BIN" scripts/data/prepare_dual_opsd_audio_class.py --source "$PREP_DATA_DIR/${split}_raw.jsonl" \
      --output "$PREP_DATA_DIR/$split.jsonl" --degraded-name "$MUTE_OUTPUT_NAME"
  done
  TRAIN_JSONL="$PREP_DATA_DIR/train.jsonl"
  VAL_JSONL="$PREP_DATA_DIR/val.jsonl"
  export TRAIN_JSONL VAL_JSONL
fi
"$PYTHON_BIN" - <<'PY'
import json, os, torch
from pathlib import Path
expected = int(os.environ['NPROC_PER_NODE'])
if torch.cuda.device_count() != expected:
    raise SystemExit(f'Expected {expected} visible GPUs, got {torch.cuda.device_count()}')
settings = {k:v for k,v in os.environ.items() if k.startswith('AURORA_DUAL_') or k in
            ['CUDA_VISIBLE_DEVICES', 'NPROC_PER_NODE', 'MAX_STEPS', 'NUM_TRAIN_EPOCHS', 'LEARNING_RATE',
             'MAX_COMPLETION_LENGTH', 'TRAIN_JSONL', 'VAL_JSONL', 'FPS', 'AURORA_VIDEO_FPS',
             'AURORA_NUM_SAM_FRAMES', 'AURORA_OPSD_TEACHER_MODE', 'BASE_MODEL',
             'STAGE1_ADAPTERS', 'TEACHER_MODEL', 'TEACHER_ADAPTERS', 'OFFLOAD_TEACHER_MODEL',
             'FPS_MAX_FRAMES', 'VIDEO_MAX_TOKEN_NUM', 'IMAGE_MAX_TOKEN_NUM',
             'ENABLE_AUDIO_OUTPUT', 'USE_AUDIO_IN_VIDEO', 'MUTE_FRACTION', 'MUTE_BLOCK_MS',
             'MUTE_OUTPUT_NAME', 'MUTE_METADATA', 'MUTE_SEED', 'MUTE_CONDITION']}
Path(os.environ['OUTPUT_DIR'], 'dual_launch.json').write_text(json.dumps(settings, indent=2)+'\n')
print(json.dumps(settings, indent=2))
PY
trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT_DIR/exit_code.txt"' EXIT
bash "$ROOT_DIR/scripts/opd/train_opsd_swift.sh"
