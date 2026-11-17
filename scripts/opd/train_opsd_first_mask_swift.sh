#!/usr/bin/env bash
# OPSD ablation: the frozen teacher receives mask_paths[0] as an additional
# image; the student receives only the original 10-frame video and audio.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export AURORA_OPSD_TEACHER_MASK_IMAGE=1
export TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_full_opsd_first_mask.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${ROOT_DIR}/outputs/data/refavs/refavs_val_opsd_first_mask.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/train/opsd_first_mask_r1}"

exec bash "${ROOT_DIR}/scripts/opd/train_opsd_swift.sh" "$@"
