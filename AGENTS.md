# Evaluation settings

- All future claims about model benefit must be evaluated on the official
  `test_s` and/or `test_u` splits. If a rapid iteration permits only one split,
  evaluate `test_s` first. Validation subsets may be used for debugging,
  checkpoint inspection and hypothesis generation, but must not be reported as
  final gain evidence or used interchangeably with test metrics.
- Attribute an OPD gain only through a protocol-matched test comparison against
  the appropriate no-OPD control (same initializer, data mixture, optimizer,
  training budget and evaluation settings). An absolute test score or a
  comparison with an unmatched SFT checkpoint does not isolate OPD.
- For every generated training or evaluation script, inspect the dataset's media
  timing/preparation settings before choosing FPS. Do not record a dataset FPS
  merely because a launcher or conversation used that number.
- Actual audit on 2026-10-25: all 3566 distinct clips referenced by the current
  train/val/test_s/test_u manifests have 10 stored frames and 10-second clip IDs;
  audio duration is approximately 10 seconds. Nominal frame density is 1 FPS,
  consistent with `scripts/run_sft_fps1.sh`. JSONL and metadata.csv have no
  per-frame timestamps/FPS column, so exact sampling instants are not verified.
- The user requires evaluation preprocessing to match the selected checkpoint's
  actual training settings. Read its saved `args.json`, launch manifest and
  training launcher before writing or changing an evaluation entrypoint.
- The completed Qwen2.5-Omni-3B audio/class MOPD training used model timeline
  2 FPS (`FPS=2.0`, `AURORA_VIDEO_FPS=2.0`). This is a known mismatch with the
  nominal data rate, not evidence that the dataset is 2 FPS. Never label the two
  rates as identical. For dataset-aligned evaluation use the audited data rate
  and explicitly record the historical training difference. Do not silently
  change a running/resumed training job or claim changing evaluation FPS repairs
  the already-trained checkpoint.
- Verify FPS, visual token/frame limits, max_pixels, maximum sequence and response
  lengths, compute dtype, SAM frame count, template and trained adapter components.
  A historical evaluation score is not evidence that its settings match this run.
- Record settings and their source in the evaluation manifest. Reject known
  mismatches before loading models. List intended inference differences (e.g.
  greedy decoding and original test audio versus sampled mute50 training rollouts)
  explicitly instead of claiming every train/eval option is identical.
- Use the trained checkpoint's projector, SEG embeddings and SAM mask decoder;
  the original SAM checkpoint only initializes the base SAM components.
- Evaluations that run the SAM segmentation branch must save every predicted
  frame mask by default and record the mask directory in the evaluation
  manifest. Also retain per-frame IoU/Dice in each JSONL record so first-frame,
  later-frame and temporal-consistency analyses do not require model replay.
  Disable mask saving only for an explicitly language-only diagnostic or when
  the user explicitly requests it; record that exception in the manifest.
- Details and the 2026-10-25 correction: `scripts/docs/OPSD_AUDIO_CLASS_GATE_README.md`.
