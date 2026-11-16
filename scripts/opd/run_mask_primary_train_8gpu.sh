#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/third_party/sam2_runtime_deps:${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export AURORA_MASK_PRIMARY_ARM="${ARM:?ARM=grpo or decision}"
[[ "$ARM" == grpo || "$ARM" == decision ]] || exit 2
export TRAINER_PLUGIN="${ROOT_DIR}/swift_plugin/aurora_mask_primary.py"
export SFT_CHECKPOINT="${ROOT_DIR}/outputs/train/sam2_merged_sft_3epoch_8gpu_20261102_merged_v4_full/checkpoint-4809"
export DATA_ROOT="${ROOT_DIR}/outputs/data/merged_student_only_online_20261103_v1"
export OUTPUT_DIR="${OUTPUT_DIR:?}"
export AURORA_MASK_CLASS_ALPHA=.2
export AURORA_DECISION_MIN_IOU=.5 AURORA_DECISION_IOU_TOLERANCE=.01
export AURORA_VERIFIED_TASK_REWARD_COEF=1 AURORA_VERIFIED_KD_COEF=.25
export AURORA_VERIFIED_TEACHERS=both AURORA_VERIFIED_TOKEN_FRACTION=1 AURORA_VERIFIED_TOKEN_RADIUS=0
export AURORA_ONLINE_TEACHER_TEMPERATURE=1.0 AURORA_ONLINE_TEACHER_TOP_P=.9
export MAX_STEPS="${MAX_STEPS:-1000}" SAVE_STEPS="${SAVE_STEPS:-250}" SAVE_TOTAL_LIMIT=5
export LEARNING_RATE=1e-7 REFERENCE_KL_BETA=.01 NUM_GENERATIONS=8 GENERATION_BATCH_SIZE=8 MAX_COMPLETION_LENGTH=256
export TRAIN_SEED="${TRAIN_SEED:-42}"
export DATA_SEED="${TRAIN_SEED}"
export MASTER_PORT="${MASTER_PORT:-29730}"
"$PYTHON_BIN" - <<'PY'
import hashlib,json,os
from pathlib import Path
cp=Path(os.environ["SFT_CHECKPOINT"]);args=json.loads((cp/"args.json").read_text())
p=json.loads((cp/"aurora_projector_config.json").read_text())
t=json.loads((cp.parent/"training_manifest.json").read_text())
for k,v in {"torch_dtype":"float16","max_length":5120,"max_pixels":200704,
            "template":"aurora_qwen2_5_omni","attn_impl":"sdpa"}.items():
 if args.get(k)!=v:raise SystemExit(f"Initializer mismatch {k}")
for k,v in {"model_timeline_fps":1.0,"sam_frames":10,"backend":"sam2_video_memory"}.items():
 if p.get(k)!=v:raise SystemExit(f"Projector mismatch {k}")
if t["model_timeline_fps"]!=1 or t["dataset_nominal_fps"]!=1:raise SystemExit("SFT media mismatch")
d=Path(os.environ["DATA_ROOT"]);m=json.loads((d/"manifest.json").read_text())
if hashlib.sha256((d/"train.jsonl").read_bytes()).hexdigest()!=m["train_sha256"]:
 raise SystemExit("Training data changed")
if m["sampled_rows"]!=12820 or m["sampled_unique_ids"]!=6094:raise SystemExit("Mixture mismatch")
audit=json.loads(Path("outputs/artifacts/refavs_fps_audit_20261025/audit.json").read_text())
if audit["nominal_fps"]!=1 or audit["unique_clips_checked"]!=3566:raise SystemExit("Timing audit mismatch")
for line in (d/"train.jsonl").open():
 r=json.loads(line)
 if len(r["sam_frame_paths"])!=10 or len(r["mask_paths"])!=10 or len(r["videos"][0])!=10:
  raise SystemExit("Stored frame count mismatch")
print("mask-primary preflight: nominal/timeline1FPS;10stored frames; train mixture SHA matched",flush=True)
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
