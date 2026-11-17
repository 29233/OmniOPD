#!/usr/bin/env bash
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
export SWIFT_ENV PYTHON_BIN SWIFT_BIN
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || { echo "Missing executable PYTHON_BIN: ${PYTHON_BIN:-<empty>}" >&2; exit 1; }
export AURORA_MASK_PRIMARY_ARM="${ARM:?ARM=grpo or decision}"
[[ "$ARM" == grpo || "$ARM" == decision ]] || exit 2
export TRAINER_PLUGIN="${TRAINER_PLUGIN:-${ROOT_DIR}/swift_plugin/aurora_mask_primary.py}"
export SFT_CHECKPOINT="${SFT_CHECKPOINT:-${ROOT_DIR}/outputs/train/sam2_merged_sft_3epoch_8gpu_20261102_merged_v4_full/checkpoint-4809}"
export DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/outputs/data/merged_student_only_online_20261103_v1}"
export DATASET_AUDIT="${DATASET_AUDIT:-${ROOT_DIR}/outputs/artifacts/refavs_fps_audit_20261025/audit.json}"
export EXPECTED_SAMPLED_ROWS="${EXPECTED_SAMPLED_ROWS:-12820}"
export EXPECTED_UNIQUE_IDS="${EXPECTED_UNIQUE_IDS:-6094}"
export EXPECTED_AUDIT_CLIPS="${EXPECTED_AUDIT_CLIPS:-}"
export OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory}"
export AURORA_MASK_CLASS_ALPHA="${AURORA_MASK_CLASS_ALPHA:-.2}"
export AURORA_DECISION_MIN_IOU="${AURORA_DECISION_MIN_IOU:-.5}" AURORA_DECISION_IOU_TOLERANCE="${AURORA_DECISION_IOU_TOLERANCE:-.01}"
export AURORA_VERIFIED_TASK_REWARD_COEF="${AURORA_VERIFIED_TASK_REWARD_COEF:-1}" AURORA_VERIFIED_KD_COEF="${AURORA_VERIFIED_KD_COEF:-.25}"
export AURORA_VERIFIED_TEACHERS="${AURORA_VERIFIED_TEACHERS:-both}" AURORA_VERIFIED_TOKEN_FRACTION="${AURORA_VERIFIED_TOKEN_FRACTION:-1}" AURORA_VERIFIED_TOKEN_RADIUS="${AURORA_VERIFIED_TOKEN_RADIUS:-0}"
export AURORA_ONLINE_TEACHER_TEMPERATURE="${AURORA_ONLINE_TEACHER_TEMPERATURE:-1.0}" AURORA_ONLINE_TEACHER_TOP_P="${AURORA_ONLINE_TEACHER_TOP_P:-.9}"
export MAX_STEPS="${MAX_STEPS:-1000}" SAVE_STEPS="${SAVE_STEPS:-250}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
export LEARNING_RATE="${LEARNING_RATE:-1e-7}" REFERENCE_KL_BETA="${REFERENCE_KL_BETA:-.01}" NUM_GENERATIONS="${NUM_GENERATIONS:-8}" GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}" MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export DATA_SEED="${DATA_SEED:-${TRAIN_SEED}}"
export MASTER_PORT="${MASTER_PORT:-29730}"
"$PYTHON_BIN" - <<'PY'
import hashlib,json,os
from pathlib import Path
cp=Path(os.environ["SFT_CHECKPOINT"]);args=json.loads((cp/"args.json").read_text())
p=json.loads((cp/"aurora_projector_config.json").read_text())
manifest_path=next(
    (cand for cand in (cp/"training_manifest.json", cp.parent/"training_manifest.json", cp.parent.parent/"training_manifest.json") if cand.is_file()),
    None,
)
if manifest_path is None:
 raise SystemExit(f"Missing training_manifest.json for {cp}")
t=json.loads(manifest_path.read_text())
for k,v in {"torch_dtype":"float16","max_length":5120,"max_pixels":200704,
            "template":"aurora_qwen2_5_omni","attn_impl":"sdpa"}.items():
 if args.get(k)!=v:raise SystemExit(f"Initializer mismatch {k}")
for k,v in {"model_timeline_fps":1.0,"sam_frames":10,"backend":"sam2_video_memory"}.items():
 if p.get(k)!=v:raise SystemExit(f"Projector mismatch {k}")
if t["model_timeline_fps"]!=1 or t["dataset_nominal_fps"]!=1:raise SystemExit("SFT media mismatch")
d=Path(os.environ["DATA_ROOT"]);m=json.loads((d/"manifest.json").read_text())
if hashlib.sha256((d/"train.jsonl").read_bytes()).hexdigest()!=m["train_sha256"]:
 raise SystemExit("Training data changed")
expected_rows=int(os.environ["EXPECTED_SAMPLED_ROWS"])
expected_ids=int(os.environ["EXPECTED_UNIQUE_IDS"])
if expected_rows>0 and m["sampled_rows"]!=expected_rows:raise SystemExit(f"Mixture rows mismatch: {m['sampled_rows']} != {expected_rows}")
if expected_ids>0 and m["sampled_unique_ids"]!=expected_ids:raise SystemExit(f"Mixture unique IDs mismatch: {m['sampled_unique_ids']} != {expected_ids}")
audit=json.loads(Path(os.environ["DATASET_AUDIT"]).read_text())
if audit["nominal_fps"]!=1 or audit.get("unique_clips_checked",0)<1:raise SystemExit("Timing audit mismatch")
expected_clips=os.environ.get("EXPECTED_AUDIT_CLIPS","")
if expected_clips and audit["unique_clips_checked"]!=int(expected_clips):raise SystemExit("Timing audit clip count mismatch")
for line in (d/"train.jsonl").open():
 r=json.loads(line)
 if len(r["sam_frame_paths"])!=10 or len(r["mask_paths"])!=10 or len(r["videos"][0])!=10:
  raise SystemExit("Stored frame count mismatch")
print("mask-primary preflight: nominal/timeline 1 FPS; 10 stored frames; train mixture SHA matched",flush=True)
PY
# Snapshot before loading models; final manifest adds these immutable sources.
"$PYTHON_BIN" - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
root=Path(os.environ["OUTPUT_DIR"]);snapshot=root/"source_snapshot";snapshot.mkdir(parents=True,exist_ok=True)
paths=["scripts/opd/run_mask_primary_train_8gpu.sh","scripts/grpo/run_sam2_verified_reward_grpo_8gpu.sh",
       "swift_plugin/aurora_mask_primary.py","swift_plugin/mask_primary_utils.py",
       "swift_plugin/aurora_online_verified_opd.py","swift_plugin/aurora_verified_routed_opd.py",
       "swift_plugin/verified_opd_routing.py","swift_plugin/aurora_verified_reward_grpo.py"]
records=[]
for path in paths:
 p=Path(path);dest=snapshot/p.name
 if dest.exists() and dest.read_bytes()!=p.read_bytes():raise SystemExit(f"Refusing changed launch snapshot: {p}")
 shutil.copy2(p,dest);records.append({"original":str(p.resolve()),"snapshot":str(dest.resolve()),
                                    "sha256":hashlib.sha256(dest.read_bytes()).hexdigest()})
(root/"launch_manifest.json").write_text(json.dumps({"arm":os.environ["ARM"],
 "seed":int(os.environ["TRAIN_SEED"]),"sources":records},indent=2)+"\n")
PY
bash "${ROOT_DIR}/scripts/grpo/run_sam2_verified_reward_grpo_8gpu.sh"
"$PYTHON_BIN" - <<'PY'
import hashlib,json,os,shutil
from pathlib import Path
root=Path(os.environ["OUTPUT_DIR"]);m=json.loads((root/"training_manifest.json").read_text())
m["optimization"].update(seed=int(os.environ["TRAIN_SEED"]),data_seed=int(os.environ["DATA_SEED"]))
m["launcher"]=str(Path("scripts/opd/run_mask_primary_train_8gpu.sh").resolve())
m["source_snapshot"]=json.loads((root/"launch_manifest.json").read_text())["sources"]
(root/"training_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
if os.environ.get("DRY_RUN")!="1":
 cp=root/f"checkpoint-{os.environ['MAX_STEPS']}";s=json.loads((cp/"trainer_state.json").read_text())
 if s["global_step"]!=int(os.environ["MAX_STEPS"]):raise SystemExit("Incomplete optimizer updates")
 for name in ("adapter_model.safetensors","sam2_video_decoder.safetensors","aurora_projector_config.json"):
  if not (cp/name).is_file():raise SystemExit(f"Missing trained sidecar {name}")
 (root/"completion.json").write_text(json.dumps({"status":"complete","arm":os.environ["ARM"],
  "global_step":s["global_step"],"checkpoint":str(cp.resolve()),"seed":int(os.environ["TRAIN_SEED"])},indent=2)+"\n")
PY
