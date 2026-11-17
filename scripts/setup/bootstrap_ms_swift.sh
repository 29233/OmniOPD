#!/usr/bin/env bash
# Fetch and patch ms-swift v4.5.2 locally for frozen-adapter OPD, and install
# it into an isolated Python environment.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${MS_SWIFT_DEST:-$ROOT/third_party/ms-swift}"
PATCH="$ROOT/scripts/setup/ms-swift-v4.5.2-teacher-adapters.patch"
SWIFT_REPO="${MS_SWIFT_REPO:-https://github.com/modelscope/ms-swift.git}"
SWIFT_OBS_REQ="$ROOT/scripts/setup/requirements_swift_observed_20261117.txt"
SAM2_REQ="$ROOT/scripts/setup/requirements_sam2_runtime.txt"

INSTALL_DEPS="${INSTALL_DEPS:-0}"
usage() {
  cat <<'EOF'
Usage: bash scripts/setup/bootstrap_ms_swift.sh [--install-deps | --no-deps] [-h|--help]

Options:
  --install-deps   Verify CUDA PyTorch (torch, torchvision, torchaudio) is present,
                   clone/patch ms-swift v4.5.2, install observed Swift packages,
                   ms-swift framework dependencies (constrained by observed pins),
                   and SAM2 runtime dependencies, then editable-install ms-swift.
  --no-deps        Verify PyTorch, Transformers, and framework/SAM2 prerequisites
                   are already present, then clone/patch/editable-install ms-swift
                   with --no-deps. (Default)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-deps) INSTALL_DEPS=1; shift ;;
    --no-deps) INSTALL_DEPS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-base}" != "base" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    echo "No isolated environment Python detected." >&2
    echo "Create and activate a Python 3.10 venv (python3 -m venv .venv && source .venv/bin/activate)" >&2
    echo "or set PYTHON_BIN=/path/to/isolated-env/bin/python." >&2
    exit 2
  fi
elif [[ "${PYTHON_BIN}" != */* ]]; then
  PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN:-<empty>}" >&2
  exit 2
fi

# Fail before cloning or installing if Python < 3.10 or PyTorch/Transformers base prerequisites are missing.
"${PYTHON_BIN}" - "${INSTALL_DEPS}" <<'PY'
import importlib
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Expected Python >= 3.10 (tested on 3.10), got {sys.version.split()[0]} at {sys.executable}."
    )

install_deps = sys.argv[1] == "1"
missing_torch = []
for mod in ("torch", "torchvision", "torchaudio"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing_torch.append(mod)
if missing_torch:
    raise SystemExit(
        "Missing PyTorch base prerequisites (" + ", ".join(missing_torch) + ") in "
        f"{sys.executable}.\nInstall CUDA-matched torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
        "for your GPU platform first, then re-run bootstrap_ms_swift.sh."
    )

if not install_deps:
    missing_base = []
    for mod in (
        "transformers",
        "tokenizers",
        "accelerate",
        "peft",
        "trl",
        "datasets",
        "safetensors",
        "hydra",
        "omegaconf",
        "iopath",
        "addict",
        "dacite",
        "json_repair",
        "qwen_omni_utils",
        "decord",
        "soundfile",
        "librosa",
        "einops",
    ):
        try:
            importlib.import_module(mod)
        except Exception:
            missing_base.append(mod)
    if missing_base:
        raise SystemExit(
            "Missing Swift/SAM2 prerequisites (" + ", ".join(missing_base) + ") in "
            f"{sys.executable}.\nRun with --install-deps:\n"
            f"  PYTHON_BIN={sys.executable} bash scripts/setup/bootstrap_ms_swift.sh --install-deps"
        )
PY

if [[ ! -d "$DEST/.git" ]]; then
  if [[ -e "$DEST" ]]; then echo "Refusing non-Git checkout: $DEST" >&2; exit 2; fi
  git clone --depth=1 --branch v4.5.2 "$SWIFT_REPO" "$DEST"
fi
if [[ "$(git -C "$DEST" describe --tags --exact-match 2>/dev/null)" != "v4.5.2" ]]; then
  echo "Expected unmodified v4.5.2 ms-swift checkout" >&2; exit 2
fi
if [[ "$(git -C "$DEST" rev-parse HEAD)" != "ff5128777dd3dc5538f1a95fbcc29442c327d7d2" ]]; then
  echo "Unexpected v4.5.2 commit; refusing to apply a patch to different upstream" >&2; exit 2
fi
if git -C "$DEST" apply --check "$PATCH"; then
  git -C "$DEST" apply "$PATCH"
elif ! git -C "$DEST" apply --reverse --check "$PATCH"; then
  echo "ms-swift patch neither cleanly applicable nor already applied" >&2; exit 2
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$SWIFT_OBS_REQ"
  if [[ -f "$DEST/requirements/framework.txt" ]]; then
    "$PYTHON_BIN" -m pip install -c "$SWIFT_OBS_REQ" -r "$DEST/requirements/framework.txt"
  fi
  "$PYTHON_BIN" -m pip install -c "$SWIFT_OBS_REQ" -r "$SAM2_REQ"
fi

"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$DEST"
"$PYTHON_BIN" - <<'PY'
import numpy
import peft
import torch
import transformers
import trl
import hydra
import omegaconf
import iopath
from pathlib import Path
from swift.pipelines.train import rlhf
from swift.arguments import rlhf_args
root = Path(rlhf.__file__).resolve().parents[3]
assert root.name == "ms-swift", (root, rlhf_args.__file__)
assert "args.teacher_adapters" in Path(rlhf.__file__).read_text()
assert "not self.teacher_adapters" in Path(rlhf_args.__file__).read_text()
print(
    "isolated ms-swift v4.5.2 teacher-adapter patch active:",
    root,
    f"(torch={torch.__version__}, transformers={transformers.__version__}, "
    f"numpy={numpy.__version__}, peft={peft.__version__}, trl={trl.__version__})",
)
PY
