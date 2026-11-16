"""Persistent 8-GPU real-model acceptance -> fresh native pure-OPD training."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def parse_metrics(line):
    line = ANSI.sub("", line).strip()
    if not re.search(r"\bstep:\d+ - actor/", line):
        return None
    result = {}
    for part in line.split(" - "):
        match = re.search(r"([\w./]+):([^\s]+)$", part)
        if match:
            try:
                result[match[1]] = float(match[2])
            except ValueError:
                pass
    return result


def terminate_owned_tree(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([parent, *children], timeout=20)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def run_stage(stage, output, steps, batch, save, env, state, state_path):
    stage_env = dict(env, CKPT_DIR=str(output), TOTAL_TRAINING_STEPS=str(steps),
                     TRAIN_BATCH_SIZE=str(batch), PPO_MINI_BATCH_SIZE=str(batch),
                     SAVE_FREQ=str(save))
    log = output.parent / (output.name + ".log")
    rollout_dir = output / "rollouts"
    cmd = [
        "bash", str(REPO / "run_omni.sh"),
        "ray_kwargs.ray_init.num_cpus=24",
        "+ray_kwargs.ray_init.object_store_memory=8589934592",
        f"+ray_kwargs.ray_init._temp_dir=/tmp/omopd_{stage}_{os.getpid()}",
        f"trainer.rollout_data_dir={rollout_dir}",
        "trainer.max_actor_ckpt_to_keep=3",
        "trainer.resume_mode=disable",
    ]
    state.update(stage=stage, status="running", command=cmd, output=str(output),
                 log=str(log), updated=datetime.now(timezone.utc).isoformat())
    write(state_path, state)
    log.parent.mkdir(parents=True, exist_ok=True)
    records, short_streak = [], 0
    with log.open("w", buffering=1) as stream:
        process = subprocess.Popen(
            cmd, env=stage_env, cwd=REPO, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        state["child_pid"] = process.pid
        write(state_path, state)
        try:
            for line in process.stdout:
                stream.write(line)
                metrics = parse_metrics(line)
                if metrics is None:
                    continue
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("Nonfinite native training metric; stop rather than extend a failed run")
                records.append(metrics)
                short_streak = short_streak + 1 if metrics.get("response_length/mean", 999) <= 4 else 0
                state.update(last_step=int(metrics.get("training/global_step", len(records))),
                             last_metrics=metrics, updated=datetime.now(timezone.utc).isoformat())
                write(state_path, state)
                with state_path.with_name(f"{stage}_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(f"[{stage}] step={state['last_step']} "
                      f"grad={metrics.get('actor/grad_norm')} "
                      f"OPD={metrics.get('actor/delta_router/selected_token_ratio')} "
                      f"response={metrics.get('response_length/mean')}", flush=True)
                # Job safety watchdog only; never modifies token weights/loss.
                if short_streak >= 3:
                    raise RuntimeError("Three consecutive near-empty response batches; stopped for diagnosis")
            code = process.wait()
            if code:
                raise RuntimeError(f"{stage} native trainer exited {code}; see {log}")
        except BaseException:
            terminate_owned_tree(process.pid)
            raise
    if len(records) != steps:
        raise RuntimeError(f"{stage}: expected {steps} completed updates, observed {len(records)}")
    if stage == "acceptance":
        if not any(row.get("actor/grad_norm", 0) > 0 for row in records):
            raise RuntimeError("Acceptance produced no nonzero gradient")
        if not any(row.get("actor/delta_router/selected_token_ratio", 0) > 0 for row in records):
            raise RuntimeError("Acceptance produced no effective native OPD routing")
        if any(row.get("response_length/mean", 0) <= 4 for row in records):
            raise RuntimeError("Acceptance produced near-empty responses")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    if args.steps < 1 or args.batch < 8 or args.batch % 8:
        raise ValueError("Use positive steps and global batch divisible by 8")
    root = PROJECT / "outputs/automation" / f"clean_omni_mopd_{args.run_id}"
    root.mkdir(parents=True, exist_ok=False)
    state_path = root / "state.json"
    state = {"run_id": args.run_id, "host": socket.gethostname(),
             "status": "preparing", "started": datetime.now(timezone.utc).isoformat(),
             "official_test_gain_claim": False}
    write(state_path, state)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        import torch
        if torch.cuda.device_count() != 8:
            raise RuntimeError(f"Expected actual 8 visible CUDA GPUs, got {torch.cuda.device_count()}")
        gpu_info = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader",
        ], text=True)
        write(root / "hardware.json", {
            "host": socket.gethostname(), "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "count": torch.cuda.device_count(), "python": sys.executable, "gpu_info": gpu_info,
        })
        models = PROJECT / "outputs/models/mopd_omni_reasoning"
        sources = {
            "MATH_TEACHER_MODEL": ("grpo1000_v1", "merged_grpo_online_20261103_v1"),
            "CODE_TEACHER_MODEL": ("joint1000_v1", "merged_joint_online_20261103_v1"),
            "INSTRUCT_FOLLOWING_TEACHER_MODEL": ("full_response_joint1000_v1",
                                                "full_response_joint_20261104_v1_seed42"),
        }
        initializer = PROJECT / "outputs/train/sam2_merged_sft_3epoch_8gpu_20261102_merged_v4_full/checkpoint-4809"
        from omni_mopd.preflight import inspect_export, verify_native_core
        core = verify_native_core(REPO)
        snapshot = root / "source_snapshot"
        shutil.copytree(REPO / "omni_mopd", snapshot / "omni_mopd", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("run_omni.sh", "run.sh", "verl/verl/workers/fsdp_workers.py",
                     "verl/verl/trainer/main_ppo.py", "verl/verl/utils/model.py"):
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, path)
        env = dict(os.environ, NGPUS_PER_NODE="8", TEACHER_SIGNAL_AGGREGATION="delta",
                   STUDENT_MODEL=str(models / "sft4809_v2"),
                   TEACHER_BASE_MODEL=str(models / "sft4809_v2"),
                   LEARNING_RATE="1e-5", PYTHON_BIN=sys.executable,
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        pool = {}
        for key, (folder, source) in sources.items():
            checkpoint = PROJECT / "outputs/train" / source / "checkpoint-1000"
            saved = json.loads((checkpoint / "args.json").read_text())
            if saved.get("adapters") != [str(initializer)]:
                raise ValueError(f"{source}: teacher does not descend from shared SFT parent")
            export = inspect_export(models / folder)
            if Path(export["adapter"]).resolve() != checkpoint.resolve():
                raise ValueError(f"{source}: export source mismatch")
            if export.get("verified_loaded_lora_tensors") != 112:
                raise ValueError(f"{source}: full language adapter loading has not been verified")
            env[key] = str(models / folder)
            pool[key] = {"export": env[key], "source_checkpoint": str(checkpoint),
                         "lineage_verified": True, "specialized_expert_quality_proven": False}
        write(root / "plan.json", {
            "student": str(initializer), "shared_pre_posttraining_base": str(initializer),
            "teachers": pool, "teacher_pool_selection": "available completed same-parent posttraining snapshots; "
            "not claimed modality-specialized experts",
            "native_core": core, "steps": args.steps, "global_batch": args.batch,
            "dataset_rows": 12820, "sampling": "native shuffle seed42 over full data; n=1, no first-1000 subset",
            "formal_sample_positions": args.steps * args.batch,
            "acceptance_updates": 2, "acceptance_weights_used_for_formal": False,
            "objective": "native pure OPD, no task reward, no SAM",
        })
        train_root = PROJECT / "outputs/train"
        acceptance = train_root / f"clean_omni_mopd_8gpu_{args.run_id}_acceptance"
        formal = train_root / f"clean_omni_mopd_8gpu_{args.run_id}"
        run_stage("acceptance", acceptance, 2, args.batch, -1, env, state, state_path)
        state.update(stage="acceptance_passed", acceptance=str(acceptance),
                     formal_initializer_unchanged=True)
        write(state_path, state)
        print("Acceptance passed. Starting fresh full-budget native run.", flush=True)
        time.sleep(5)
        run_stage("formal", formal, args.steps, args.batch, 100, env, state, state_path)
        state.update(status="completed", stage="formal_completed",
                     completed=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                     updated=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise


if __name__ == "__main__":
    main()
