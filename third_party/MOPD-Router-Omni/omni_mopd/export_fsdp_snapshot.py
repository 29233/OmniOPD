"""Merge the eight FSDP1 shards into an HF Thinker; audit all frozen tensors."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def entries(state, world, rank):
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    for name, value in state.items():
        if isinstance(value, torch.distributed.tensor.DTensor):
            if tuple(value.shape) == ():
                yield name, value.to_local().cpu(), (), tuple(value.shape)
                continue
            shape, offset = compute_local_shape_and_global_offset(
                value.shape, value.device_mesh, value.placements
            )
            yield name, value.to_local().cpu(), tuple(
                (int(start), int(start + size)) for start, size in zip(offset, shape)
            ), tuple(value.shape)
        elif hasattr(value, "local_shards"):
            for shard in value.local_shards():
                shape = shard.metadata.shard_sizes
                offset = shard.metadata.shard_offsets
                yield name, shard.tensor.cpu(), tuple(
                    (int(start), int(start + size)) for start, size in zip(offset, shape)
                ), tuple(value.size())
        elif torch.is_tensor(value):
            yield name, value.cpu(), tuple((0, int(x)) for x in value.shape), tuple(value.shape)
        else:
            raise TypeError(f"Unknown saved parameter kind {name}: {type(value)}")


def run(args):
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in {2, 8}:
        raise ValueError("Only audited 2-rank tiny test or real 8-rank snapshot")
    source, initializer, output = map(Path, (args.checkpoint, args.initializer, args.output))
    for i in range(world):
        if not (source / f"model_world_size_{world}_rank_{i}.pt").is_file():
            raise FileNotFoundError(f"Incomplete source FSDP checkpoint, rank{i}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "parts").mkdir()
    dist.barrier()
    index_path = initializer / "model.safetensors.index.json"
    if index_path.exists():
        base_index = json.loads(index_path.read_text())["weight_map"]
    else:
        single = initializer / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError("Missing HF safetensors initializer")
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            base_index = {key: single.name for key in handle.keys()}
    from contextlib import ExitStack

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(safe_open(str(initializer / shard), framework="pt", device="cpu"))
            for shard in sorted(set(base_index.values()))
        }
        saved = torch.load(source / f"model_world_size_{world}_rank_{rank}.pt",
                           map_location="cpu", weights_only=False)
        tensors, metadata = {}, []
        for name, local, ranges, full_shape in entries(saved, world, rank):
            if name not in base_index:
                raise ValueError(f"FSDP checkpoint contains unexpected tensor {name}")
            slices = tuple(slice(a, b) for a, b in ranges)
            expected = handles[base_index[name]].get_slice(name)[slices]
            if list(local.shape) != list(expected.shape) or not torch.isfinite(local).all():
                raise ValueError(f"Invalid shard for {name}, rank {rank}")
            is_decoder = name.startswith(("model.layers.", "model.norm."))
            if is_decoder:
                # FP32 master weights -> actual BF16 inference; full network is
                # trained, whereas perception/embedding/head are frozen.
                if local.numel():
                    tensors[name] = local.to(expected.dtype).contiguous()
                metadata.append({"name": name, "ranges": ranges, "shape": full_shape,
                                 "rank": rank, "trainable": True})
            elif not torch.equal(local.to(expected.dtype), expected):
                raise ValueError(f"Frozen tensor {name} differs from initializer on rank{rank}")
            else:
                metadata.append({"name": name, "ranges": ranges, "shape": full_shape,
                                 "rank": rank, "trainable": False})
        save_file(tensors, str(output / "parts" / f"rank{rank}.safetensors"))
        (output / "parts" / f"rank{rank}.json").write_text(json.dumps(metadata))
        del saved
    dist.barrier()
    if rank == 0:
        parts = {i: json.loads((output / "parts" / f"rank{i}.json").read_text()) for i in range(world)}
        names = set(base_index)
        for i in range(world):
            if {x["name"] for x in parts[i]} != names:
                raise ValueError(f"rank{i} is missing saved weights")
        part_state = {i: load_file(output / "parts" / f"rank{i}.safetensors") for i in range(world)}
        final_index = {}
        changed = 0
        for shard in sorted(set(base_index.values())):
            state = load_file(initializer / shard)
            for name, tensor in state.items():
                if not name.startswith(("model.layers.", "model.norm.")):
                    continue
                merged = torch.empty_like(tensor)
                occupied = torch.zeros(tensor.shape, dtype=torch.bool)
                for i in range(world):
                    item = next(x for x in parts[i] if x["name"] == name)
                    slices = tuple(slice(a, b) for a, b in item["ranges"])
                    if not slices or not all(a < b for a, b in item["ranges"]):
                        continue
                    if occupied[slices].any():
                        raise ValueError(f"Overlap while reconstructing {name}")
                    occupied[slices] = True
                    merged[slices] = part_state[i][name]
                if not occupied.all():
                    raise ValueError(f"Incomplete reconstruction of {name}")
                changed += int(not torch.equal(merged, tensor))
                state[name] = merged
            save_file(state, output / shard)
            final_index.update({k: shard for k in state})
        if set(final_index) != names or not changed:
            raise ValueError("Output does not cover model or contains no language update")
        for file in initializer.iterdir():
            if file.is_file() and file.suffix != ".safetensors" and file.name != "model.safetensors.index.json":
                shutil.copy2(file, output / file.name)
        if index_path.is_file():
            shutil.copy2(index_path, output / index_path.name)
        manifest = {
            "source_checkpoint": str(source.resolve()),
            "initializer": str(initializer.resolve()),
            "world_size": world,
            "source_step": int(source.parent.name.split("_")[-1]),
            "complete": True,
            "trainable_language_tensors_changed": changed,
            "all_frozen_tensors_exact_after_dtype_cast": True,
            "training_core": "original RayPPOTrainer FSDP1; language decoder only",
            "dtype": "bfloat16",
            "segmentation_loaded": False,
        }
        (output / "fsdp_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
