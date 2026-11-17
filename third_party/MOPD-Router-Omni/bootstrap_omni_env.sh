#!/usr/bin/env bash
# Provision a SEPARATE venv overlay for native MOPD-Router-Omni.
# NOTE: requirements-omni.txt pins numpy==1.26.4 and opencv-python-headless==4.11.0.86,
# which conflict with the ms-swift runtime (numpy==2.1.3, opencv-python-headless==5.0.0.93).
# Always keep OMNI_ENV separate from your ms-swift .venv / SWIFT_ENV.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$REPO/../.." && pwd)"
BASE="${BASE_PYTHON:-python3}"
if [[ "$BASE" != */* ]]; then
  BASE="$(command -v "$BASE" || true)"
fi
if [[ -z "$BASE" || ! -x "$BASE" ]]; then
  echo "BASE_PYTHON is not executable: ${BASE_PYTHON:-python3}" >&2
  exit 2
fi
ENV="${OMNI_ENV:-$PROJECT/outputs/envs/mopd-router-omni-v1}"

# Guard against overwriting the ms-swift environment with numpy==1.26.4.
for swift_candidate in "$PROJECT/.venv" "${SWIFT_ENV:-}" "${VIRTUAL_ENV:-}"; do
  if [[ -n "$swift_candidate" && "$(cd -- "$(dirname "$ENV")" 2>/dev/null && pwd)/$(basename "$ENV")" == "$(cd -- "$(dirname "$swift_candidate")" 2>/dev/null && pwd)/$(basename "$swift_candidate")" ]]; then
    echo "Refusing to install MOPD overlay (numpy==1.26.4) into Swift/active venv: $ENV" >&2
    echo "Keep OMNI_ENV separate (default: $PROJECT/outputs/envs/mopd-router-omni-v1)." >&2
    exit 2
  fi
done

# Fail fast if BASE_PYTHON does not provide the shared torch/transformers base.
"$BASE" - <<'PY'
import sys
try:
    import torch, transformers
except Exception as exc:
    raise SystemExit(
        f"BASE_PYTHON ({sys.executable}) lacks torch/transformers base prerequisites: {exc}\n"
        "Point BASE_PYTHON to a Python 3.10 interpreter with torch==2.7.1 and transformers==5.16.1 installed."
    )
if torch.__version__.split("+")[0] != "2.7.1" or transformers.__version__ != "5.16.1":
    raise SystemExit(
        f"Expected BASE_PYTHON with torch==2.7.1 and transformers==5.16.1, got "
        f"torch={torch.__version__}, transformers={transformers.__version__}."
    )
PY

if [[ ! -x "$ENV/bin/python" ]]; then "$BASE" -m venv --system-site-packages "$ENV"; fi
"$ENV/bin/python" -m pip install -r "$REPO/requirements-omni.txt"
"$ENV/bin/python" -m pip freeze > "$ENV/omni-environment.freeze.txt"
PYTHONPATH="$REPO:$REPO/verl" "$ENV/bin/python" - <<'PY'
import sys, numpy, ray, tensordict, torch, transformers, verl
assert torch.__version__.split("+")[0] == "2.7.1"
assert transformers.__version__ == "5.16.1"
assert numpy.__version__ == "1.26.4", f"Expected numpy==1.26.4 in MOPD overlay, got {numpy.__version__}"
print({
    "python": sys.executable,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "ray": ray.__version__,
    "tensordict": tensordict.__version__,
    "verl": verl.__file__,
})
PY
