"""CPU two-rank audit of native FSDP smoke checkpoint vs its initial export."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initializer", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    original = {}
    for path in Path(a.initializer).glob("model*.safetensors"):
        original.update(load_file(path))
    folder = Path(a.checkpoint)
    state = torch.load(folder / f"model_world_size_{world}_rank_{rank}.pt",
                       weights_only=False, map_location="cpu")
    changed, frozen, checked = [], [], 0
    for key, value in state.items():
        if isinstance(value, torch.distributed.tensor.DTensor):
            from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

            shape, offset = compute_local_shape_and_global_offset(
                value.shape, value.device_mesh, value.placements
            )
            slices = tuple(slice(start, start + size) for start, size in zip(offset, shape))
            pairs = [(value.to_local(), original[key][slices])]
        elif not hasattr(value, "local_shards"):
            pairs = [(value, original[key])]
        else:
            pairs = []
            for part in value.local_shards():
                slices = tuple(slice(start, start + size) for start, size in zip(
                    part.metadata.shard_offsets, part.metadata.shard_sizes
                ))
                pairs.append((part.tensor, original[key][slices]))
        for actual, expected in pairs:
            checked += 1
            if not torch.isfinite(actual).all():
                raise ValueError(f"Nonfinite saved parameter {key}")
            is_language = key.startswith(("model.layers.", "model.norm."))
            if not torch.equal(actual, expected.to(actual.dtype)):
                if not is_language:
                    raise ValueError(f"Frozen tensor changed: {key}")
                changed.append(key)
            elif not is_language:
                frozen.append(key)
    optim = torch.load(folder / f"optim_world_size_{world}_rank_{rank}.pt",
                       weights_only=False, map_location="cpu")
    local = {
        "rank": rank, "checked_shards": checked, "changed_language_tensors": changed,
        "unchanged_frozen_tensors": frozen, "optimizer_state_keys": len(optim.get("state", {})),
    }
    results = [None] * world
    dist.all_gather_object(results, local)
    if rank == 0:
        changed = sorted(set(k for result in results for k in result["changed_language_tensors"]))
        if not changed:
            raise ValueError("Native FSDP smoke did not update any language weights")
        report = {
            "checkpoint": str(folder.resolve()), "world_size": world,
            "language_weights_really_changed": changed,
            "all_checked_nonlanguage_weights_exactly_unchanged": True,
            "saved_model_shards_reloaded": True, "saved_optimizer_shards_reloaded": True,
            "ranks": results,
            "scope": "random tiny fixture only, not a 7B model-benefit evaluation",
        }
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
