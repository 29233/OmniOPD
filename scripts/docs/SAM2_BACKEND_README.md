# AURORA SAM2 video-memory backend

## Source and environment

- Upstream: `https://github.com/facebookresearch/sam2.git`
- Pinned commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Checkout: `third_party/sam2`
- Runtime dependencies for remote pods: `third_party/sam2_runtime_deps`
- Hiera-L checkpoint: `models/sam2_hiera_large.pt`
- Config: `configs/sam2/sam2_hiera_l.yaml`

The optional SAM2 connected-components CUDA extension is not required. The
training and inference path uses the official PyTorch image encoder, mask
decoder, memory encoder and memory attention implementation.

## AURORA integration

Set `AURORA_SEG_BACKEND=sam2`. The Qwen2.5-Omni template then preprocesses ten
stored frames using SAM2's official 1024x1024 resize and ImageNet normalization.

The single AURORA `[SEG]` hidden state is projected to 256 dimensions and added
as a sparse SAM2 prompt on every frame. Unlike the original AURORA SAM path,
frames are decoded sequentially:

1. The first frame produces a mask, object pointer and memory feature.
2. Each later frame attends to the conditioning frame and recent mask memories.
3. Each prediction is encoded into memory for subsequent frames.
4. BCE and Dice supervise all ten output masks.

The Hiera image encoder and native point/mask prompt encoder remain frozen. Full
SFT updates the q/v LoRA, `embed_tokens`, `lm_head`, `text_hidden_fcs`,
`sam_mask_decoder`, `memory_attention`, `memory_encoder`, `obj_ptr_proj` and
`mask_downsample`.
The pretrained object-presence hard gate is disabled because it has not seen
AURORA text prompts and otherwise replaces early masks with a constant,
zero-gradient no-object mask.

## Checkpoint conversion

`prepare_sam2_sft_initializer.py` retains the trained language LoRA, `[SEG]`
embedding, LM head and AURORA projector from the Answer-CoT 7B SFT checkpoint.
It removes the old SAM decoder and adds the official SAM2 decoder/memory weights.

Every SAM2 training checkpoint contains:

- `adapter_model.safetensors`
- `sam2_video_decoder.safetensors`
- `aurora_projector_config.json`

The sidecar is normalized independently of PEFT's runtime compatibility proxies
and is restored by the AURORA model loader.

## Launch

Two-GPU smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 SMOKE=1 \
  bash scripts/sft/train_qwen2_5_omni_7b_sam2_seg_sft.sh
```

Eight-GPU three-epoch joint Answer-CoT SFT and automatic evaluation:

```bash
bash scripts/sft/train_qwen2_5_omni_7b_sam2_seg_sft.sh
```

Merged structured Audio / Non-Audio SAM2 SFT (3 epochs, 8 GPUs):

```bash
bash scripts/sft/train_merged_sam2_sft_3epoch_8gpu.sh
```

The full run uses global batch 8, 3 epochs, 1 FPS/10 frames, 5120 context,
FP16 SDPA, CE/BCE/Dice weights 1/2/0.5, `lr=1e-5`, cosine scheduling,
`weight_decay=0.1` and gradient checkpointing. After the final checkpoint it
runs greedy `test_s` and `test_u` evaluation on all eight GPUs, saves all ten
predicted masks per sample, and writes aggregate plus per-frame metrics.
