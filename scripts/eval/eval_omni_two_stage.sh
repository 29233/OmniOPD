#!/usr/bin/env bash
# Run independent GPU replicas, finish generation, then run SAM and aggregate.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
if [[ "${OMNIOPD_LOAD_ENV:-0}" == "1" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi
if [[ -z "${SWIFT_ENV:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    SWIFT_ENV="${VIRTUAL_ENV}"
  elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != "base" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    SWIFT_ENV="${CONDA_PREFIX}"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    SWIFT_ENV="${ROOT_DIR}/.venv"
  fi
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${SWIFT_ENV:-}" && -x "${SWIFT_ENV}/bin/python" ]]; then
    PYTHON_BIN="${SWIFT_ENV}/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
elif [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi
export SWIFT_ENV PYTHON_BIN
[[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]] || { echo "Missing executable PYTHON_BIN: ${PYTHON_BIN:-<empty>}" >&2; exit 1; }
FAMILY="${FAMILY:-qwen3}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained adapter directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new output directory}"
case "${FAMILY}" in
  qwen3)
    DEFAULT_MAX_NEW_TOKENS=256
    MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/Qwen3-Omni-30B-A3B-Instruct}"
    GPU_GROUPS="${GPU_GROUPS:-0,1;2,3}"
    export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
    if [[ -n "${MS_SWIFT_ROOT:-}" ]]; then
      export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH}"
    fi
    ;;
  qwen2_5)
    DEFAULT_MAX_NEW_TOKENS=512
    MODEL_PATH="${MODEL_PATH:-${MODEL:-models/Qwen2___5-Omni-7B}}"
    GPU_GROUPS="${GPU_GROUPS:-0;1;2;3}"
    export PYTHONPATH="${ROOT_DIR}/third_party/sam2:${ROOT_DIR}:${PYTHONPATH:-}"
    ;;
  *) echo "FAMILY must be qwen3 or qwen2_5" >&2; exit 1 ;;
esac
STAGE="${STAGE:-all}"
case "${STAGE}" in all|generate|segment) ;; *) echo "STAGE must be all, generate or segment" >&2; exit 1 ;; esac
IFS=';' read -r -a GROUPS_ARRAY <<< "${GPU_GROUPS}"
read -r -a SPLIT_ARGS <<< "${SPLITS:-test_s test_u}"
COMMON_ARGS=(--data-dir "${DATA_DIR:-${ROOT_DIR}/outputs/data/refavs}"
  --sam-checkpoint "${SAM_CHECKPOINT:-models/sam_vit_h_4b8939.pth}")
RESTART_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then RESTART_ARGS+=(--resume); fi
if [[ "${RETRY_ERRORS:-0}" == "1" ]]; then RESTART_ARGS+=(--retry-errors); fi
GENERATE_ARGS=(--family "${FAMILY}" --model "${MODEL_PATH}" --checkpoint "${CHECKPOINT}"
  --splits "${SPLIT_ARGS[@]}" --batch-size "${BATCH_SIZE:-1}" --max-samples "${MAX_SAMPLES:--1}"
  --max-new-tokens "${MAX_NEW_TOKENS:-${DEFAULT_MAX_NEW_TOKENS}}" --max-memory-per-gpu "${MAX_MEMORY_PER_GPU:-52GiB}"
  --attn-impl "${ATTN_IMPL:-sdpa}" --num-frames "${NUM_FRAMES:-10}")
if [[ -n "${VIDEO_FPS:-}" ]]; then GENERATE_ARGS+=(--video-fps "${VIDEO_FPS}"); fi
SEGMENT_ARGS=(--checkpoint "${CHECKPOINT}" --splits "${SPLIT_ARGS[@]}"
  --device cuda:0 --cache-size "${CACHE_SIZE:-32}")
SEGMENT_ARGS+=(--save-masks)
SHARD_DIRS=()
for ((rank=0; rank<${#GROUPS_ARRAY[@]}; rank++)); do
  SHARD_DIRS+=("${OUTPUT_DIR}/shard-${rank}")
done
mkdir -p "${OUTPUT_DIR}"
run_phase() {
  local phase="$1" rank gpu_group failed=0
  local -a pids=() phase_args=()
  if [[ "${phase}" == "generate" ]]; then phase_args=("${GENERATE_ARGS[@]}"); else phase_args=("${SEGMENT_ARGS[@]}"); fi
  for ((rank=0; rank<${#GROUPS_ARRAY[@]}; rank++)); do
    gpu_group="${GROUPS_ARRAY[rank]}"
    if [[ "${phase}" == "segment" ]]; then gpu_group="${gpu_group%%,*}"; fi
    local -a args=("${ROOT_DIR}/scripts/eval/eval_omni_two_stage.py" "${phase}" "${COMMON_ARGS[@]}"
      "${phase_args[@]}" "${RESTART_ARGS[@]}" --output "${SHARD_DIRS[rank]}")
    if [[ "${phase}" == "generate" ]]; then args+=(--shard "${rank}" --shards "${#GROUPS_ARRAY[@]}"); fi
    echo "[${phase}] shard=${rank} GPUs=${gpu_group} output=${SHARD_DIRS[rank]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_group}"
      printf '%q ' "${PYTHON_BIN}" "${args[@]}"
      printf '\n'
    else
      CUDA_VISIBLE_DEVICES="${gpu_group}" "${PYTHON_BIN}" "${args[@]}" > "${OUTPUT_DIR}/${phase}-shard-${rank}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  if [[ "${failed}" == "1" ]]; then echo "${phase} failed; inspect ${OUTPUT_DIR}/${phase}-shard-*.log" >&2; return 1; fi
}
if [[ "${STAGE}" != "segment" ]]; then run_phase generate; fi
if [[ "${STAGE}" != "generate" ]]; then
  run_phase segment
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval/eval_omni_two_stage.py" summarize --inputs "${SHARD_DIRS[@]}" --output "${OUTPUT_DIR}"
  fi
fi
