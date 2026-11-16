"""One-time export of Stage-1 trained SAM-side weights for the SAM IoU service.

Reads the Stage-1 SFT adapter checkpoint (ms-swift `--modules_to_save` layout)
and writes plain state_dicts the SAM server can load:

    text_hidden_fcs.pt   -> projector (2048 -> 2048 -> 256), 4 tensors
    mask_decoder.pt      -> SAM mask decoder, 120 tensors (keys already
                            unwrapped: 'decoder.*' for the inner SAM decoder)

Verified layout of checkpoint-5262:
    adapter_model.safetensors   base_model.model.text_hidden_fcs.{0,2}.{weight,bias}
                                base_model.model.mask_decoder.* (120 tensors)
    sam_mask_decoder.safetensors  duplicate of the decoder tensors (bare names)

The image/prompt encoders are frozen in Stage-1, so the plain ViT-H
checkpoint on the server already matches them.

Usage (on any machine that can read the checkpoint):
    python scripts/utils/export_sam_weights.py \
        --checkpoint outputs/train/qwen2_5_omni_sft_full/v1-20260827-141702/checkpoint-5262 \
        --out-dir outputs/artifacts/sam_service_weights
"""
from __future__ import annotations

import argparse
import os

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Stage-1 ms-swift checkpoint dir (checkpoint-XXXX)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    adapter = load_file(os.path.join(args.checkpoint, "adapter_model.safetensors"))

    projector = {}
    for k, v in adapter.items():
        if "text_hidden_fcs" not in k:
            continue
        # 'base_model.model.text_hidden_fcs.0.weight' -> '0.weight'
        name = k.split("text_hidden_fcs.")[-1]
        projector[name] = v.contiguous()

    decoder = {}
    for k, v in adapter.items():
        if "mask_decoder" not in k:
            continue
        # 'base_model.model.mask_decoder.decoder.iou_token.weight'
        #   -> 'iou_token.weight'  (unwraps both the proxy and the prefix)
        if ".mask_decoder.decoder." in k:
            name = k.split(".mask_decoder.decoder.")[-1]
        elif ".mask_decoder." in k:
            name = k.split(".mask_decoder.")[-1]
        else:
            continue
        decoder[name] = v.contiguous()

    if len(projector) != 4:
        raise SystemExit(f"expected 4 projector tensors, found {len(projector)}: "
                         f"{sorted(projector)}")
    if not decoder:
        raise SystemExit("no mask_decoder tensors found in adapter_model.safetensors")

    os.makedirs(args.out_dir, exist_ok=True)
    torch_save = __import__("torch").save
    torch_save(projector, os.path.join(args.out_dir, "text_hidden_fcs.pt"))
    torch_save(decoder, os.path.join(args.out_dir, "mask_decoder.pt"))
    print(f"projector: {len(projector)} tensors, decoder: {len(decoder)} tensors")
    print(f"written to {args.out_dir}/text_hidden_fcs.pt and mask_decoder.pt")


if __name__ == "__main__":
    main()
