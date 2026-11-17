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
The SAM and SAM2 Python sources are vendored here (weights are not). The
ms-swift checkout is not bundled: create a Python 3.10 venv with CUDA-matched
PyTorch (`torch==2.7.1`, `torchvision==0.22.1`, `torchaudio==2.7.1`) and run
`bash scripts/setup/bootstrap_ms_swift.sh --install-deps` (or `--no-deps` when
dependencies are pre-installed) to fetch v4.5.2, apply the two teacher-adapter
fixes, and install observed Swift/framework/SAM2 dependencies. Its extra SAM2
dependencies are listed in `scripts/setup/requirements_sam2_runtime.txt`. The
clean MOPD migration contains its own `verl/` Python sources.

## Local use

1. For the `ms-swift` / SAM / SAM2 path, start with `README.md`,
   `MS_SWIFT_SFT.md`, and `scripts/docs/INDEPENDENT_VERIFY.md`. The historical
   `requirements_ms_swift.txt` is legacy-only (archived Transformers 4.x pins)
   and is NOT the tested 2026-11-17 Transformers 5.x environment. The *observed*
   Swift and native MOPD environments have separate files under
   `scripts/setup/` (`numpy==2.1.3` vs MOPD `numpy==1.26.4`), and neither is a
   complete transitive lockfile.
   Set model/data/output paths via `.env` (`OMNIOPD_LOAD_ENV=1` or
   `set -a && source .env && set +a`) or inline environment variables when
   invoking launchers. Launchers auto-detect `.venv` and `PYTHONPATH`.
2. For native `MOPD-Router-Omni` reasoning-only training, follow
   `third_party/MOPD-Router-Omni/OMNI_MIGRATION.md`.
   That path requires a compatible CUDA/PyTorch/Transformers/Ray environment, local
   merged Thinker exports, three independent frozen teachers, training data, and timing audit.
3. Evaluation scripts validate each selected checkpoint's saved training arguments and
   manifest to ensure protocol-matched inference and mask saving.

See [scripts/docs/INDEPENDENT_VERIFY.md](scripts/docs/INDEPENDENT_VERIFY.md) for
the verified launch/preflight steps and the difference between a dry run and a
completed GPU optimizer step.

The native MOPD code implements the clean reasoning-only migration; the AURORA
`swift_plugin` modules cover segmentation-aware SFT/GRPO/OPD training. They are
separate training paths and should not be treated as one combined algorithm.
