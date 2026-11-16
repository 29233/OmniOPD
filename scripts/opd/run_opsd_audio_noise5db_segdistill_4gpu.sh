#!/usr/bin/env bash
# Matched Gaussian-noise audio OPSD run, adding projected [SEG] alignment.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SNR_DB="${SNR_DB:-5}"
export RUN_TAG="${RUN_TAG:-opsd_audio_noise${SNR_DB}db_segdistill_frozen_r2}"
export AURORA_OPSD_SEG_DISTILL_WEIGHT="${AURORA_OPSD_SEG_DISTILL_WEIGHT:-1.0}"
export AURORA_OPSD_SEG_DISTILL_LOSS="${AURORA_OPSD_SEG_DISTILL_LOSS:-mse}"
exec bash "${ROOT_DIR}/scripts/opd/run_opsd_audio_noise5db_4gpu.sh"
