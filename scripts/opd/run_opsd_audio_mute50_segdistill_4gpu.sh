#!/usr/bin/env bash
# Matched mute50 audio OPSD run, adding projected [SEG] feature alignment.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUN_TAG="${RUN_TAG:-opsd_audio_mute50_segdistill_frozen_r2}"
export AURORA_OPSD_SEG_DISTILL_WEIGHT="${AURORA_OPSD_SEG_DISTILL_WEIGHT:-1.0}"
export AURORA_OPSD_SEG_DISTILL_LOSS="${AURORA_OPSD_SEG_DISTILL_LOSS:-mse}"
exec bash "${ROOT_DIR}/scripts/opd/run_opsd_audio_mute50_4gpu.sh"
