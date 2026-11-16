#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$REPO/../.." && pwd)"
BASE="${BASE_PYTHON:-python3}"
ENV="${OMNI_ENV:-$PROJECT/outputs/envs/mopd-router-omni-v1}"
if [[ ! -x "$ENV/bin/python" ]]; then "$BASE" -m venv --system-site-packages "$ENV"; fi
"$ENV/bin/python" -m pip install -r "$REPO/requirements-omni.txt"
"$ENV/bin/python" -m pip freeze > "$ENV/omni-environment.freeze.txt"
PYTHONPATH="$REPO:$REPO/verl" "$ENV/bin/python" - <<'PY'
import sys, torch, transformers, verl
assert torch.__version__.split("+")[0] == "2.7.1"
assert transformers.__version__ == "5.16.1"
print({"python": sys.executable, "torch": torch.__version__,
       "transformers": transformers.__version__, "verl": verl.__file__})
PY
