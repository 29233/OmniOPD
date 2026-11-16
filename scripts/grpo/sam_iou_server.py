"""AURORA SAM IoU microservice — HTTP server side.

Runs on the dedicated SAM node (another V100 group). The GRPO training
cluster sends, per completion: the [SEG] hidden state (2048 floats) plus the
frame/mask paths. This service loads SAM ViT-H + the Stage-1-trained
``text_hidden_fcs`` projector + ``mask_decoder`` weights, runs
projector -> SAM -> postprocess, and returns the frame-averaged IoU against
the GT masks.

Path contract: frames/GT masks live on shared NFS (/mnt/tbo), so the service
resolves the same absolute paths the training side sends. No image bytes
cross the wire — only the 8KB hidden state per request.

Endpoints:
    POST /iou        {"hidden": [2048 floats],
                      "sam_frame_paths": [...], "mask_paths": [...],
                      "num_frames": 10}
    GET  /health

Launch (on the SAM node):
    AURORA_SAM_CHECKPOINT=... AURORA_SAM_DEVICE=cuda:0 \
    AURORA_PROJECTOR_WEIGHTS=<stage1 ckpt>/text_hidden_fcs.pt \
    AURORA_MASK_DECODER_WEIGHTS=<stage1 ckpt>/mask_decoder.pt \
    python scripts/grpo/sam_iou_server.py --port 8877

The Stage-1 checkpoint must first be exported with
scripts/utils/export_sam_weights.py (one-time, on the training side).
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in os.path.join(os.pathsep, os.environ.get("PYTHONPATH", "")):
    os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

from swift_plugin.segmentation import (  # noqa: E402
    load_gt_masks,
    preprocess_sam_frames,
)

_DEFAULT_NUM_FRAMES = int(os.environ.get("AURORA_NUM_SAM_FRAMES", "10"))


def _load_sam_stack(device: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """SAM ViT-H + projector; Stage-1 trained weights if provided."""
    sam_checkpoint = os.environ.get(
        "AURORA_SAM_CHECKPOINT",
        "models/sam_vit_h_4b8939.pth",
    )
    from model.segment_anything import build_sam_vit_h

    sam = build_sam_vit_h(sam_checkpoint)
    sam.requires_grad_(False)
    sam.eval()
    sam = sam.to(device)

    hidden_size = 2048  # Qwen2.5-Omni-3B/7B thinker text hidden size
    projector = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(hidden_size, 256),
        torch.nn.Dropout(0.0),
    )
    projector.requires_grad_(False)
    projector.eval()

    proj_w = os.environ.get("AURORA_PROJECTOR_WEIGHTS", "")
    if proj_w and os.path.isfile(proj_w):
        projector.load_state_dict(torch.load(proj_w, map_location="cpu"))
        print(f"[sam-server] loaded projector weights: {proj_w}")
    else:
        raise FileNotFoundError(
            "AURORA_PROJECTOR_WEIGHTS not set or missing — the projector must "
            "carry Stage-1 trained weights or IoU rewards are meaningless. "
            "Export with scripts/utils/export_sam_weights.py first."
        )

    dec_w = os.environ.get("AURORA_MASK_DECODER_WEIGHTS", "")
    if dec_w and os.path.isfile(dec_w):
        # mask_decoder on disk is the raw SAM decoder (MaskDecoderSaveProxy
        # stores it as .decoder in modules_to_save; the export script unwraps).
        sam.mask_decoder.load_state_dict(torch.load(dec_w, map_location="cpu"))
        print(f"[sam-server] loaded mask_decoder weights: {dec_w}")
    else:
        print("[sam-server] WARNING: no mask_decoder weights given; using "
              "the ViT-H checkpoint's decoder (Stage-0). Pass "
              "AURORA_MASK_DECODER_WEIGHTS if Stage-1 trained it.")
    projector = projector.to(device)
    return sam, projector


_SAM: Optional[torch.nn.Module] = None
_PROJECTOR: Optional[torch.nn.Module] = None
_DEVICE = os.environ.get("AURORA_SAM_DEVICE", "cuda:0")
# image encoder runs fp32 on V100 (no bf16); decoder follows the trainer's dtype
_ENC_DTYPE = torch.float32
_DEC_DTYPE = torch.float32


@torch.no_grad()
def compute_iou(hidden: Sequence[float], sam_frame_paths: List[str],
                mask_paths: List[str], num_frames: int) -> float:
    sam, projector = _SAM, _PROJECTOR
    seg_hidden = torch.tensor([hidden], dtype=torch.float32, device=_DEVICE)
    prompt = projector(seg_hidden).to(_DEC_DTYPE).view(1, 1, -1)

    images, resize_sizes, original_sizes = preprocess_sam_frames(sam_frame_paths, num_frames)
    images = images.to(device=_DEVICE, dtype=_ENC_DTYPE)
    gt = load_gt_masks(mask_paths, original_sizes[0], num_frames).to(_DEVICE)

    image_embeddings = sam.image_encoder(images)
    image_embeddings = image_embeddings.to(_DEC_DTYPE)
    pe = sam.prompt_encoder.get_dense_pe().to(_DEC_DTYPE)
    sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None, text_embeds=prompt)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings,
        image_pe=pe,
        sparse_prompt_embeddings=sparse.to(_DEC_DTYPE),
        dense_prompt_embeddings=dense.to(_DEC_DTYPE),
        multimask_output=False,
    )
    pred_mask = sam.postprocess_masks(
        low_res_masks,
        input_size=tuple(resize_sizes[0]),
        original_size=tuple(original_sizes[0]),
    )[:, 0]
    pred_bin = (torch.sigmoid(pred_mask) > 0.5).to(torch.float32)
    inter = (pred_bin * gt).sum(dim=(-1, -2))
    union = torch.maximum(pred_bin, gt).sum(dim=(-1, -2))
    return float((inter / (union + 1e-7)).mean().item())


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "device": _DEVICE})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/iou":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            iou = compute_iou(
                req["hidden"],
                req["sam_frame_paths"],
                req["mask_paths"],
                int(req.get("num_frames", _DEFAULT_NUM_FRAMES)),
            )
            self._json(200, {"iou": iou})
        except FileNotFoundError as exc:
            self._json(422, {"error": f"file not found: {exc}"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": repr(exc)})
            import traceback
            traceback.print_exc()


def main():
    global _SAM, _PROJECTOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()

    _SAM, _PROJECTOR = _load_sam_stack(_DEVICE)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[sam-server] listening on {args.host}:{args.port} (device={_DEVICE})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
