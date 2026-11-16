"""HF media-aware rollout over the *same* native FSDP actor, not a new trainer."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

from .modeling import extend_positions


class OmniHFRollout:
    def __init__(self, module, config):
        self.module, self.config = module, config
        self.shared_actor = True

    @torch.no_grad()
    def generate_sequences(self, prompts):
        # Per-item generation avoids accidentally concatenating video frames or
        # dropping audio. Each rank receives the same number of items in Ray.
        was_training = self.module.training
        self.module.eval()
        outputs = []
        try:
            ctx = FSDP.summon_full_params(self.module, writeback=False) if isinstance(
                self.module, FSDP
            ) else nullcontext()
            with ctx:
                for i in range(len(prompts)):
                    sample = prompts[i:i + 1]
                    outputs.append(self._one(sample))
        finally:
            self.module.train(was_training)
        return DataProto.concat(outputs)

    def _one(self, prompts):
        ids = prompts.batch["input_ids"]
        attention = prompts.batch["attention_mask"]
        positions = prompts.batch["position_ids"]
        if "multi_modal_inputs" not in prompts.non_tensor_batch:
            raise ValueError("Omni rollout refuses to discard audio/video inputs")
        media = {
            k: v.to(ids.device) for k, v in prompts.non_tensor_batch["multi_modal_inputs"][0].items()
        }
        if not {"input_features", "feature_attention_mask", "pixel_values_videos",
                "video_grid_thw", "video_second_per_grid"} <= media.keys():
            raise ValueError("Incomplete Omni media tensor bundle")
        validate = prompts.meta_info.get("validate", False)
        sampling = self.config.val_kwargs if validate else self.config
        do_sample = prompts.meta_info.get("do_sample", sampling.do_sample)
        limit = int(self.config.response_length)
        eos, pad = prompts.meta_info["eos_token_id"], prompts.meta_info["pad_token_id"]
        generation = GenerationConfig(
            do_sample=do_sample, temperature=sampling.temperature,
            top_p=sampling.top_p, top_k=max(0, sampling.top_k),
            eos_token_id=eos, pad_token_id=pad, max_new_tokens=limit,
        )
        with torch.autocast(
            device_type=ids.device.type, dtype=torch.bfloat16,
            enabled=ids.device.type == "cuda",
        ):
            seq = self.module.generate(
                input_ids=ids, attention_mask=attention,
                # Let HF generate canonical text+3D positions and its internal
                # rope_delta cache. Explicit 3D positions are used for replay.
                **media, use_audio_in_video=False, generation_config=generation,
                use_cache=True, synced_gpus=False, return_dict_in_generate=False,
            )
        if seq.shape[1] <= ids.shape[1] or seq.shape[1] > ids.shape[1] + limit:
            raise ValueError("Unexpected generation length")
        seq = torch.nn.functional.pad(seq, (0, ids.shape[1] + limit - seq.shape[1]), value=pad)
        response = seq[:, ids.shape[1]:]
        response_mask = get_response_mask(response, eos, dtype=attention.dtype)
        return DataProto(
            batch=TensorDict({
                "prompts": ids, "responses": response, "input_ids": seq,
                "attention_mask": torch.cat([attention, response_mask], dim=-1),
                "position_ids": extend_positions(positions, limit),
            }, batch_size=[1]),
            non_tensor_batch=prompts.non_tensor_batch,
            meta_info=prompts.meta_info,
        )
