#!/usr/bin/env bash
# Shared protocol: GRPO / pure online OPD / online joint, selected only by ARM.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
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
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || { echo "Missing executable PYTHON_BIN: ${PYTHON_BIN:-<empty>}" >&2; exit 1; }
export PYTHONPATH="$ROOT_DIR/third_party/sam2:$ROOT_DIR:${PYTHONPATH:-}"
ARM="${ARM:?Set ARM=grpo,opd,joint}"
export AURORA_VERIFIED_TEACHERS="${AURORA_VERIFIED_TEACHERS:-both}" AURORA_VERIFIED_KD_COEF="${AURORA_VERIFIED_KD_COEF:-.25}"
export AURORA_VERIFIED_TOKEN_FRACTION="${AURORA_VERIFIED_TOKEN_FRACTION:-.10}" AURORA_VERIFIED_TOKEN_RADIUS="${AURORA_VERIFIED_TOKEN_RADIUS:-1}"
export AURORA_ONLINE_TEACHER_TEMPERATURE="${AURORA_ONLINE_TEACHER_TEMPERATURE:-1.0}" AURORA_ONLINE_TEACHER_TOP_P="${AURORA_ONLINE_TEACHER_TOP_P:-.9}"
case "$ARM" in
 grpo) export TRAINER_PLUGIN="$ROOT_DIR/swift_plugin/aurora_verified_reward_grpo.py" AURORA_VERIFIED_TASK_REWARD_COEF=1;;
 opd) export TRAINER_PLUGIN="$ROOT_DIR/swift_plugin/aurora_online_verified_opd.py" AURORA_VERIFIED_TASK_REWARD_COEF=0;;
 joint) export TRAINER_PLUGIN="$ROOT_DIR/swift_plugin/aurora_online_verified_opd.py" AURORA_VERIFIED_TASK_REWARD_COEF=1;;
 *) exit 2;;
esac
export MAX_STEPS="${MAX_STEPS:-1000}" SAVE_STEPS="${SAVE_STEPS:-250}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export LEARNING_RATE="${LEARNING_RATE:-1e-7}" REFERENCE_KL_BETA="${REFERENCE_KL_BETA:-.01}" NUM_GENERATIONS="${NUM_GENERATIONS:-8}" GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}" MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
export MASTER_PORT="${MASTER_PORT:-29705}"
export SFT_CHECKPOINT="${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the trained SAM2 SFT checkpoint}"
export DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to student-only online OPD data}"
export OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory}"
"$PYTHON_BIN" - <<'PY'
import json,os
from pathlib import Path
cp=Path(os.environ["SFT_CHECKPOINT"])
args=json.loads((cp/"args.json").read_text())
projector=json.loads((cp/"aurora_projector_config.json").read_text())
manifest_path=next(
    (p for p in (cp/"training_manifest.json", cp.parent/"training_manifest.json", cp.parent.parent/"training_manifest.json") if p.is_file()),
    None,
)
if manifest_path is None:
    raise SystemExit(f"Missing training_manifest.json for checkpoint: {cp}")
train=json.loads(manifest_path.read_text())
for k,v in {"max_length":5120,"max_pixels":200704,"torch_dtype":"float16",
            "template":"aurora_qwen2_5_omni","attn_impl":"sdpa"}.items():
 if args.get(k)!=v:raise SystemExit(f"Initializer mismatch: {k}")
for k,v in {"model_timeline_fps":1.0,"sam_frames":10,"backend":"sam2_video_memory"}.items():
 if projector.get(k)!=v:raise SystemExit(f"Media mismatch: {k}")
if train["model_timeline_fps"]!=1.0:raise SystemExit("Training manifest FPS mismatch")
data=json.loads((Path(os.environ["DATA_ROOT"])/"manifest.json").read_text())
if data.get("offline_teacher_required") is not False:raise SystemExit("Expected Student-only data")
for required in ("sam2_video_decoder.safetensors","adapter_model.safetensors","aurora_projector_config.json"):
 if not (cp/required).is_file():raise SystemExit(f"Missing trained component: {required}")
print("current-policy post-training preflight passed")
PY
exec bash "${ROOT_DIR}/scripts/grpo/run_sam2_verified_reward_grpo_8gpu.sh"
