#!/usr/bin/env bash
# Install the specific ms-swift sources needed by frozen-adapter OPD, locally.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/third_party/ms-swift"
PATCH="$ROOT/scripts/setup/ms-swift-v4.5.2-teacher-adapters.patch"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to an isolated environment Python executable}"
if [[ ! -d "$DEST/.git" ]]; then
  if [[ -e "$DEST" ]]; then echo "Refusing non-Git checkout: $DEST" >&2; exit 2; fi
  git clone --depth=1 --branch v4.5.2 https://github.com/modelscope/ms-swift.git "$DEST"
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
"$PYTHON_BIN" -m pip install --no-deps --no-build-isolation -e "$DEST"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from swift.pipelines.train import rlhf
from swift.arguments import rlhf_args
root = Path(rlhf.__file__).resolve().parents[3]
assert root.name == "ms-swift", (root, rlhf_args.__file__)
assert "args.teacher_adapters" in Path(rlhf.__file__).read_text()
assert "not self.teacher_adapters" in Path(rlhf_args.__file__).read_text()
print("isolated ms-swift v4.5.2 teacher-adapter patch active:", root)
PY
