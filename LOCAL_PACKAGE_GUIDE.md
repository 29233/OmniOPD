# AURORA Ref-AVS Core

Source-only repository of the project's model and training core. It combines the
AURORA `ms-swift` Omni/SAM/SAM2 plugins (SFT, GRPO, OPSD, and Verified Routed OPD)
with the clean native `MOPD-Router-Omni` reasoning-only migration.

## Included

- `swift_plugin/`: Omni, SAM/SAM2 segmentation, SFT/GRPO/OPD/joint rewards and loss plugins.
- `scripts/`: categorized into `data/`, `sft/`, `grpo/`, `opd/`, `eval/`, `utils/`, and `docs/` (see [scripts/README.md](scripts/README.md)).
- `configs/`: DeepSpeed configurations and settled structured CoT / privilege prompt templates.
- `third_party/MOPD-Router-Omni/`: clean upstream-based native MOPD-Router Omni
  reasoning-only migration, including its vendored `verl` source and migration guide.
- Project documentation and dependency declarations.

## Deliberately excluded

Model weights/checkpoints, datasets and media, generated evaluation reports,
TensorBoard/W&B logs, Python environments/caches, and credentials.
The upstream `ms-swift` and `SAM2` checkouts are not vendored here; install the
versions required by `requirements_ms_swift.txt` and documentation, or obtain them
from their upstream repositories. The clean MOPD migration has its own `verl/` source included.

## Local use

1. For the `ms-swift` / SAM / SAM2 path, start with `README.md`,
   `MS_SWIFT_SFT.md`, and `requirements_ms_swift.txt`; set model/data/output
   paths via environment variables when invoking launchers.
2. For native `MOPD-Router-Omni` reasoning-only training, follow
   `third_party/MOPD-Router-Omni/OMNI_MIGRATION.md`.
   That path requires a compatible CUDA/PyTorch/Transformers/Ray environment, local
   merged Thinker exports, three independent frozen teachers, training data, and timing audit.
3. Evaluation scripts validate each selected checkpoint's saved training arguments and
   manifest to ensure protocol-matched inference and mask saving.

The native MOPD code implements the clean reasoning-only migration; the AURORA
`swift_plugin` modules cover segmentation-aware SFT/GRPO/OPD training. They are
separate training paths and should not be treated as one combined algorithm.
