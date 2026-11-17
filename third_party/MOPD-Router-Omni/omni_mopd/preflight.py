"""Fail closed before Ray/model loading; record sources and native-core hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from .data import audit_timing
from .modeling import assert_shared_vocabulary

UPSTREAM = "a630dc3891c5b1c92cdc2815193fdcbd8fa4291b"
CORE = (
    "run.sh",
    "verl/verl/workers/actor/dp_actor.py",
    "verl/verl/utils/teacher_routing.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/trainer/ppo/core_algos.py",
)
# Fixed upstream content hashes: the OmniOPD source distribution does not ship
# the nested MOPD-Router Git history. Comparing against the parent repository
# instead could silently verify the wrong project.
UPSTREAM_CORE_SHA256 = {
    "run.sh": "33f6d17dc5923247d3aab978c1294acec2db8a9fbe523a7db7264207d16ee558",
    "verl/verl/workers/actor/dp_actor.py": "4db9af0a7337425b71ca0a15fb847e14d0e770c060cff2e97ee3ccdff726d905",
    "verl/verl/utils/teacher_routing.py": "996c9bf32adeff61fdd2b12f356bc323f64bc2ba6abc3b1d04a24bc096feae45",
    "verl/verl/trainer/ppo/ray_trainer.py": "ca8afb25438416e005852b48643d0458fdd5f40c565fcccf40046b2240db518c",
    "verl/verl/trainer/ppo/core_algos.py": "e7817157e4f13e9f85d65321a9e81f5eef1cc9d18d70fe73ca54a2026213528c",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_native_core(repo):
    import verl

    if not Path(verl.__file__).resolve().is_relative_to(repo / "verl"):
        raise ValueError(f"Wrong verl imported (old migration/environment?): {verl.__file__}")
    result = {}
    for name in CORE:
        actual = digest(repo / name)
        if actual != UPSTREAM_CORE_SHA256[name]:
            raise ValueError(f"Native routing/training core modified: {name}")
        result[name] = actual
    return result


def inspect_export(path):
    path = Path(path).resolve()
    info = json.loads((path / "omni_export_manifest.json").read_text())
    config = json.loads((path / "config.json").read_text())
    if (
        not info.get("complete") or info.get("segmentation_exported") is not False
        or info.get("format") != "merged-native-thinker-v1"
        or config.get("model_type") != "qwen2_5_omni_thinker"
    ):
        raise ValueError(f"Not a completed native reasoning-only Thinker export: {path}")
    if not list(path.glob("*.safetensors")):
        raise ValueError(f"No model weights in {path}")
    return info


def check(out, *, repo=None):
    from transformers import AutoTokenizer

    repo = repo or Path(__file__).resolve().parents[1]
    core = verify_native_core(repo)
    protocol = audit_timing(os.environ["OMNI_TIMING_AUDIT"])
    paths = {
        key: Path(os.environ[key]).resolve() for key in (
            "STUDENT_MODEL", "TEACHER_BASE_MODEL", "MATH_TEACHER_MODEL",
            "CODE_TEACHER_MODEL", "INSTRUCT_FOLLOWING_TEACHER_MODEL",
        )
    }
    teachers = [paths[k] for k in (
        "MATH_TEACHER_MODEL", "CODE_TEACHER_MODEL", "INSTRUCT_FOLLOWING_TEACHER_MODEL",
    )]
    if len(set(teachers)) != 3:
        raise ValueError("Provide three distinct frozen teacher exports; no implicit online/self teacher")
    exports = {k: inspect_export(path) for k, path in paths.items()}
    tokenizer = AutoTokenizer.from_pretrained(paths["STUDENT_MODEL"], local_files_only=True)
    for path in paths.values():
        assert_shared_vocabulary(tokenizer, AutoTokenizer.from_pretrained(path, local_files_only=True))
    # Reject teacher exports whose recorded base differs. The shared parent
    # is supplied explicitly; path identity alone does not establish lineage.
    bases = {exports[k]["base"] for k in exports}
    if len(bases) != 1:
        raise ValueError("Teachers/shared parent have different recorded Omni base lineages")
    data = Path(os.environ["TRAIN_DATA_PATH"]).resolve()
    rows = [json.loads(line) for line in data.open() if line.strip()]
    if not rows:
        raise ValueError("Empty training dataset")
    for row in rows:
        if len(row["videos"][0]) != 10 or len(row["audios"]) != 1:
            raise ValueError("Invalid audited audio/video preparation")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "upstream_commit": UPSTREAM, "upstream_core_sha256": core,
        "omni_adapter_source_sha256": {
            name: digest(repo / name) for name in (
                "run_omni.sh", "omni_mopd/data.py", "omni_mopd/modeling.py",
                "omni_mopd/rollout.py", "omni_mopd/preflight.py",
                "verl/verl/workers/fsdp_workers.py", "verl/verl/trainer/main_ppo.py",
                "verl/verl/utils/model.py",
            )
        },
        "models": {k: str(v) for k, v in paths.items()}, "exports": exports,
        "objective": "unchanged upstream pure MOPD; only_reverse_kl_advantages=true",
        "teacher_prompt": "same Student prompt/media; independent frozen model checkpoints",
        "router": os.environ.get("TEACHER_SIGNAL_AGGREGATION", "delta"),
        "online_privilege_teachers": False, "historical_gates": False,
        "joint_grpo_reward": False, "segmentation_loaded": False,
        "segmentation_mask_saving": False,
        "mask_saving_exception": "Explicit language-only migration; no SAM decoding",
        "trainable": "native Thinker model.layers and model.norm only",
        "optimizer": {"lr": float(os.environ.get("LEARNING_RATE", "1e-5")),
                      "scheduler": "constant", "reference_kl": False},
        "media": protocol | {"max_pixels": 200704, "max_length": 5120,
                             "video_preprocessing": "swift_training_parity",
                             "max_prompt_length": 4864,
                             "max_response_length": 256, "compute_dtype": "bfloat16",
                             "actor_parameter_dtype": "float32",
                             "reference_parameter_dtype": "bfloat16",
                             "attention": "sdpa", "visual_limits": [768, 768, 16384]},
        "data": {"path": str(data), "rows": len(rows), "sha256": digest(data)},
        "scope": "migration/preflight, not an official test or a model gain result",
        "runtime": {"rollout": "shared-actor HF", "trainer": "native RayPPOTrainer/FSDP1",
                    "gpus": int(os.environ.get("NGPUS_PER_NODE", "2")),
                    "python": sys.executable,
                    "versions": {name: version(name)
                                 for name in ("torch", "transformers", "ray", "numpy")}},
        "training_budget": {"steps": os.environ.get("TOTAL_TRAINING_STEPS", "3"),
                            "train_batch": int(os.environ.get("TRAIN_BATCH_SIZE", "2")),
                            "mini_batch": int(os.environ.get("PPO_MINI_BATCH_SIZE", "2")),
                            "rollouts_per_prompt": 1},
    }
    (out / "omni_training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    print(json.dumps(check(args.out), indent=2))


if __name__ == "__main__":
    main()
