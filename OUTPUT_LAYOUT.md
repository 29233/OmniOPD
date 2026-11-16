# Output Layout

All generated artifacts use the following layout under `outputs/`. Do not
write datasets, checkpoints, or evaluation results directly under `outputs/`.

```text
outputs/
  data/
    refavs/                         # REF-AVS JSONL manifests and variants
    refavs_hwnas/                   # Manifests generated on the HWNAS mount
    cot/                            # Teacher-generated CoT JSONL/JSON files
    audio_degraded/<condition>/     # Derived audio inputs, e.g. snr5
  train/
    <model>_<gpu>gpu_<frames>f/     # One ms-swift run root, including logs
      vN-YYYYMMDD-HHMMSS/
        checkpoint-<step>/
  eval/
    <model>/<checkpoint-or-run>/    # Standalone evaluation outputs
      *.jsonl
      summary*.json
      masks/                         # Optional saved mask visualizations
        <split>/<sample-id>/frame_000.png
    probes/<probe-name>/            # Teacher-privilege and diagnostics
    analysis/                       # HTML or cross-run comparisons
  artifacts/
    sam_service_weights/            # Exported reusable service weights
```

Naming rules:

- `data/` contains inputs and manifests only; it must not contain model checkpoints.
- `train/` contains training logs, TensorBoard files, arguments, optimizer state, and checkpoints.
- `eval/` contains independent test output, summaries, probes, and optional mask images; it must not be used as a training `output_dir`.
- `artifacts/` contains exported weights intended for external services or evaluators.
- Qwen REF-AVS evaluation saves thresholded predicted masks by default under
  `masks/<split>/<sample-id>/frame_XXX.png`. Each successful SAM-scored JSONL
  record includes `mask_dir` and `mask_paths`; pass `--no-save-masks` to keep
  IoU/Dice scoring while disabling image files.
- Use lowercase model identifiers: `videollama2`, `qwen2_5_omni_3b`, or `qwen2_5_omni_7b`.
- Include the GPU count and sampled frame count in wrapper-created run roots, for example `videollama2_2gpu_10f`.
- Keep the checkpoint name (`checkpoint-500`) supplied by ms-swift unchanged below the run root.
- Use `AURORA_DATA_JSONL`, `AURORA_VAL_JSONL`, `OUTPUT_DIR`, `LOG_FILE`, `--data-dir`, and `--output-dir` to override these defaults for legacy or special runs.

Historical root-level datasets, runs, probes, and exported weights have been
migrated into this layout. The migration preserves file contents and
checkpoint names; scripts and documentation should use the new locations.
