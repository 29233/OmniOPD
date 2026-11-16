# Standalone checkout: reproducibility and verification

The source is cloned into its own directory. Model weights, REF-AVS manifests,
media and trained adapters are deliberately external inputs, **not** bundled
with the repository. A working preflight does not establish that an optimizer
step or a complete official-split evaluation has succeeded.

## Dependencies

Use a Python 3.10 CUDA environment. The historical
`requirements_ms_swift.txt` pins **Transformers 4.57.6** and is NOT the
environment used for this checkout's 2026-11-16 smoke. Versions actually
observed for the ms-swift runtime are in
`scripts/setup/requirements_swift_observed_20261117.txt`; the separate
`scripts/setup/requirements_mopd_overlay_observed_20261116.txt` records
native MOPD's overlay. Install
matching CUDA torch/vision/audio wheels for your GPU and the SAM2 dependencies
from `scripts/setup/requirements_sam2_runtime.txt`. That tested-version list
documents core versions; it is not a complete lockfile for every transitive
media dependency. Then run
`PYTHON_BIN=/absolute/path/to/venv/bin/python bash scripts/setup/bootstrap_ms_swift.sh`
to obtain a local v4.5.2 ms-swift checkout patched for frozen teacher adapters.
Do not use a mixture of ms-swift 3.x and 4.x on `PYTHONPATH`.

Export `PYTHONPATH="$PWD/third_party/sam2:$PWD"`. This makes both `sam2`
and `model.segment_anything` resolve to code **inside this checkout**. Supply
`MODEL`, `SAM2_CHECKPOINT`, `SFT_CHECKPOINT`, `DATA_ROOT`, and `OUTPUT_DIR`
explicitly for the recipe chosen. `MODEL` and `SAM2_CHECKPOINT` point to
external weights, not Python source. For native MOPD, use its *separate*
requirements, launcher and three independent teachers documented in
`third_party/MOPD-Router-Omni/OMNI_MIGRATION.md`.

## CPU/source checks

```bash
PYTHONPATH="$PWD/third_party/sam2:$PWD" python -m unittest discover \
  -s tests -p 'test_independent_core.py' -v
cd third_party/MOPD-Router-Omni
PYTHONPATH="$PWD:$PWD/verl" python -m unittest \
  omni_mopd.test_migration omni_mopd.test_train_pipeline -q
```

The MOPD CPU test exercises a tiny actor optimizer update and signed reverse-KL
direction, **not** full-parameter 7B training, segmentation or test-set gains.
The independent tests check plugin import provenance, class reward, mask I/O
and frame metric. Shell syntax may be checked via `bash -n scripts/{sft,grpo,opd,eval}/*.sh`.

## Three training preflights

Export **real external paths** appropriate to the launcher; e.g.:

- SFT: `INIT_ADAPTER`, `DATA_DIR`, `PYTHON_BIN`, `OUTPUT_DIR`, then
  `DRY_RUN=1 bash scripts/sft/train_merged_sam2_sft_3epoch_8gpu.sh`.
  This checks the SFT manifest, pads 12820 distinct rows to 12824 for eight
  ranks and records settings. It does not load weights or train.
- GRPO: `MODEL`, `SFT_CHECKPOINT`, `SOURCE_ROOT`, `SAM2_CHECKPOINT`,
  `DATASET_AUDIT`, `DATA_DIR`, `OUTPUT_DIR`, `SWIFT_ENV`, then
  `DRY_RUN=1 bash scripts/grpo/train_qwen2_5_omni_7b_sam2_grpo_8gpu.sh`.
  This checks checkpoint settings and produces prompt-only data. It does
  not call `swift rlhf`.
- Online OPD: `ARM=opd`, `MODEL`, `SFT_CHECKPOINT`, `SAM2_CHECKPOINT`,
  `DATA_ROOT`, `OUTPUT_DIR`, `SWIFT_ENV`, then
  `DRY_RUN=1 bash scripts/opd/run_current_policy_opd_8gpu.sh`.
  This checks the initializer and training-data contract, **not** routing
  efficacy or the first real update.

These three launchers enforce eight visible GPU IDs. Setting
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` on a two-card machine only simulates
the preflight's rank-count check: **never treat that as eight GPUs or start a
non-dry training run on that machine**. A full training claim requires a real
step, finite losses and a loadable checkpoint with the trained SAM2/projector.

## Evaluation

`scripts/eval/test_qwen2_5_omni_refavs.py` supports `--splits test_s`,
`--max-samples 1` for a connectivity diagnostic, `--save-masks` and
`--fail-fast`. Inspect the checkpoint `args.json`, training manifest and
projector config first. Use the checkpoint's FPS, max_pixels, dtype, token
limits and SAM2 weights; media manifests have a nominal density of **1 FPS**
but no verified frame timestamps. The full official `test_s`/`test_u` splits
are required for benefit claims. A subset is for debugging only.

`scripts/eval/eval_omni_two_stage.py` is the legacy **SAM ViT-H** two-stage
evaluator and is not interchangeable with the SAM2 evaluator above. When SAM
segmentation runs it saves predicted masks by default and records per-frame
IoU/Dice. Its Qwen2.5 default `video_fps=2` must not silently be used for a
1-FPS-trained SAM2 checkpoint. Its direct evaluation path has no general
checkpoint-settings parity guard; do not report gains on a mismatched run.

## Verification performed on 2026-11-16 (UTC)

The checked-out OmniOPD revision is
`8d81ffafbad8b7b240a388ac408ef7cad0846251` **plus the local fixes in
this working tree**; this verification is not evidence that the unmodified
GitHub revision works by itself.

1. Main SFT/GRPO/online OPD 8-GPU recipes passed `DRY_RUN=1` using the
   matching existing model/data/checkpoint inputs, with **GPU IDs simulated**
   for preflight. SFT padding gave 12820 unique/12824 physical rows, GRPO
   prepared 10201 train/1277 validation rows, online OPD verified its
   Student-only dataset manifest. No actual eight-card optimization occurred.
2. Local `ms-swift` v4.5.2 was fetched at
   `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`, patched, and installed
   into a new `.venv-verify` that reuses existing site packages for CUDA
   libraries. Source-import checks and 4 core unit tests passed. Vendored
   MOPD-Router's **17 tiny CPU** tests also passed (using its native dependency
   overlay). These are engineering checks, not learned-model quality tests.
3. A real **one-GPU, one-example, one-optimizer-step 3B SAM ViT-H SFT** finished
   with finite loss `2.802` and grad norm `47.84`, and wrote checkpoint-1
   (adapter, trained SAM decoder, projector config and trainer state).
   The one-sample validation `rouge-1=5.882` is not an evaluation claim.
   The `test_s` single-case inference on this checkpoint finished with zero
   runtime errors, but generated **no [SEG]** on this sample, hence no
   segmentation masks. A one-step model is not expected to be useful.
4. The standalone new-venv evaluator also loaded the *external* trained 7B
   SAM2 SFT checkpoint-4809 and ran one official `test_s` sample: 10 saved
   masks and 10 IoU/Dice values, with source hash/settings in
   `evaluation_manifest.json`. This particular sample predicted `woman` for
   GT `piano`; its per-frame mean IoU `0.00025` is a **badcase**, not a
   checkpoint score. A prior evaluator run without explicitly replaying the
   SAM2 sidecar generated identical masks: the near-zero IoU cannot be
   attributed to a random SAM2 head on this evidence.
5. A real **one-GPU, one-example, one-step 3B current-policy class-privilege
   OPSD** run completed using the one-step SFT checkpoint as initializer.
   Logged `aurora/opsd_jsd_loss=0.0977`, finite grad norm `9.053`, and
   checkpoint-1 with SAM decoder/projector. This historical GKD recipe uses
   `beta=0` and is **not the native three-teacher MOPD reverse-KL recipe**.
   The sampled response had no `[SEG]`: `aurora/seg_rows=0` and mask loss
   zero, so this tests the text KD update path only. Its single `test_s`
   inference completed without runtime errors, but likewise had no `[SEG]`
   and cannot evaluate segmentation quality.
   Tensor audit: the first q/v LoRA-B tensors differ from their SFT
   initializer by about `1e-5` maximum absolute value.
6. The legacy 3B GRPO path completed a real **one-GPU, one-example, one-step**
   run with its trained-weight SAM IoU server reachable, saving an adapter,
   trained SAM sidecar and projector config. An independent synthetic 2048D
   `[SEG]` feature reached `POST /iou` and returned a finite value (zero for
   the uninformative zero vector); this checks the service's decode path,
   not the student's sampled segmentation reward.
   The weak one-step SFT initializer generated no valid target/SEG in two
   sampled completions: all reward columns and reward standard deviation
   were zero. The nonzero logged loss came from the KL/optimization path;
   **this does not demonstrate useful verifiable reward or a GRPO gain**.
   Its checkpoint's one-sample `test_s` inference completed without errors
   but did not generate `[SEG]`, so no segmentation masks were scored.
   The first q/v LoRA-B tensors differ from the SFT initializer by about
   `1e-6`, consistent with an optimizer update despite zero task reward.
   The revised launcher now fails when `/health` is unavailable, and the IoU
   reward raises by default on service errors; setting
   `AURORA_GRPO_REWARD_FAIL_OPEN=1` is an explicitly unsafe diagnostic only.

This host exposes only two A100 GPUs, one already used by another workload.
**Real eight-GPU SAM2 GRPO and online OPD steps, full `test_s`/`test_u`
performance, and matched no-OPD gain have NOT been verified**. The classic
GRPO recipe also needs a functioning trained SAM IoU service, not merely a
running optimizer.
The 3B one-step tests and
one 7B test sample must never be reported as model-benefit measurements.
