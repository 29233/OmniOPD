#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Build the 60K Nemotron + NuminaMath-TIR prompt mixture.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
ROOT="${ROOT:-./data/mopd_router}"

exec "${PYTHON}" "${SCRIPT_DIR}/build_nemotron_numina_mix.py" \
    --output_dir "${ROOT}/nemotron_numina_prompt_60k" \
    "$@"
