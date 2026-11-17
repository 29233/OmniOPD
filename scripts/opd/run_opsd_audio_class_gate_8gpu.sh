#!/usr/bin/env bash
# Eight DDP Student replicas, each scoring two views with an independent frozen SFT teacher.
# Platform entrypoint; dataset, output directories and logging are prepared internally.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/opsd_audio_class_gate_mute50_8gpu_$(date +%Y%m%d-%H%M%S)}"
exec bash "$ROOT_DIR/scripts/opd/run_opsd_audio_class_gate_4gpu.sh" "$@"
