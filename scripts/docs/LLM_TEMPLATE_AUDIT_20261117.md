# Omni checkpoint loading and LLM-template audit (2026-11-17, Asia/Shanghai)

This is an **engineering/selected-case audit**, not an official test-set
benefit measurement. Checked repo: `third_party/OmniOPD-independent`, source
commit `8d81ffafbad8b7b240a388ac408ef7cad0846251` plus local changes.

## What loaded and generated

- **7B SAM2 SFT checkpoint-4809**: loaded the trained language adapter, 4
  projector tensors, 285 SAM2 decoder/memory tensors, thinker embeddings and
  LM head. On four *hand-picked* `test_s` examples spanning Audio,
  Audio-Visual, Visual, Semantic: 4/4 produced CoT + one `[SEG]`; each saved
  ten frame masks and ten IoU/Dice values. A fifth Audio example also produced
  `[SEG]` and ten saved masks. Four selected cases predicted the GT category;
  this is not a representative accuracy estimate. A separate Audio example
  predicted `woman` for GT `piano` and was a segmentation badcase.
- **One-step 3B SFT, GRPO and class-privilege OPSD checkpoints**: all load,
  generate and finish on the first four `test_s` examples under their
  **original raw task prompt** with no runtime errors; none of the twelve
  outputs contained `[SEG]`, so no masks were scored. These tiny one-example
  training checkpoints are not proof that meaningful segmentation works.
- **Native MOPD step500 merged Thinker**: all 1346 exported weight tensors
  loaded, and one `test_s` Audio case produced a complete reasoning response,
  one `[SEG]` and `<|im_end|>` under both historical and corrected video input.
  This is **language only**; no projector/SAM2 segmentation was run. The
  decision was `woman` for GT `piano` in both modes. Greedy 128-token
  diagnostic is not the sampled 256-token training rollout.
  The actual `omni_mopd.eval_official_language` entrypoint was also tested
  with **one** `test_s` case. It loaded the weights, generated `[SEG]`, recorded
  `video_preprocessing=historical_native_capped` from the saved training
  manifest, and marked `full_official_split=false`.

## Low-level template contract

On an identical `test_s` record with the trained 7B processor:

| Contract | Result |
|---|---|
| Train vs inference media-expanded prompt token IDs | **exactly equal** (4049) |
| `<video><audio>` | one each, separate audio waveform |
| Audio features | `[1, 128, 30000]` |
| Stored video | 10 frames; video grid `[5,40,72]` |
| Model timeline | 1 FPS; `video_second_per_grid=2` (temporal patch=2) |
| `[SEG]` | ID 151665; saved embedding/head each have 152064 rows |
| 7B SFT train vs `test_s` task instruction header | exactly equal (856 characters) |
| Native Omni encoder **before** local media fix | video grid `[5,24,40]`, prompt 1649 tokens |
| Native Omni encoder **after** fix | grid `[5,40,72]`, prompt 4049 tokens; IDs exactly match ms-swift |
| Native vs ms-swift media values | audio/video values **exactly equal after BF16 cast**; RoPE positions consistent |

Cause: the old native `omni_mopd/data.py` passed
`max_pixels=200704` **into `fetch_video`** for each frame. The initializer's
ms-swift `Qwen2_5OmniTemplate.replace_tag` does not: for video lists it uses
the video token limit, not the template's per-image `max_pixels`. This
was a 2400-prompt-token visual distribution shift, not an FPS difference.
The fix removes the extra cap for **new native training**. On 12 sampled
train clips spanning four modalities, corrected native prompt length
was 4043–4154, with no prompt over the native 4864 limit; this is **not**
a full 12820-row overlength audit.

### Historical checkpoint safety

The real native MOPD step500 training manifest records old `data.py` SHA256
`a6f671f54a1a8e45f0b1a35a59e0334ab19cda6edf7a5563372496609898dd21`.
**The existing checkpoint cannot be retroactively repaired.**
The native language/SAM2 official evaluators now select the old capped
preprocessing for that checksum and record it in their manifests. Future
runs' manifests explicitly say `swift_training_parity`. Unknown protocols
fail closed. Do not report a performance change from evaluating step500
with corrected video input as an OPD gain.

### Task-prompt style is a separate issue

- Old 3B smoke SFT/GRPO/OPD train prompts start immediately with
  `<video><audio>`; the old raw `test_s` does too. Input prefix length on
  the first raw case: 3885 of max 4096, leaving **211** tokens before the
  trained length limit.
- The *new structured-CoT* `test_s` has an 856-character task header and
  is appropriate for the **7B structured SFT**. Giving it to those old 3B
  checkpoints raised the prompt to 4049 tokens, leaving only **47** before
  the 3B length limit. Earlier 3B structured-prompt smoke was therefore
  not a matched template test.
- Even after switching back to the correct raw 3B prompt, all four selected
  cases per one-step model lacked `[SEG]`: this cannot be blamed solely on
  that prompt mismatch. The one-example initializer is not a useful baseline.
- Direct evaluator now checks checkpoint `args.json` dataset's task header
  against **every row** in the requested source split before loading model
  weights. It rejects raw-3B vs structured-test swaps; relocated datasets
  can be supplied with `--train-prompt-source`.

The *model's* CoT format compliance is not perfect despite matching token
encoding: among four hand-picked 7B cases, one Audio case ordered Visual
evidence before Audio evidence (against the stated stage order), while both
selected non-audio cases omitted unnecessary Audio/Cross-modal evidence.
This says nothing about rates on the full split.

## Reproduction artifacts

- `outputs/verify/template_audit/7b_native_swift_after_fix.json`
- `outputs/verify/template_audit/3b_raw_contract.json`
- `outputs/verify/template_audit/3b_structured_mismatch.json` (intentional failure)
- `outputs/verify/template_audit/native_vs_swift.json` (before fix)
- `outputs/verify/template_audit/native_vs_swift_after_fix.json`
- `outputs/verify/template_audit/native_mopd_step500_input_pair.json`
- `outputs/verify/template_audit/native_mopd_step500_input_pair_v2.json`
- `outputs/verify/template_audit/native_mopd_step500_official_language_test_s1/evaluation_manifest.json`
- `outputs/verify/template_audit/sft7b_test_s_4modalities/test_s.jsonl`
- `outputs/verify/template_audit/cot_format_audit.json`

`scripts/eval/audit_llm_template.py` performs reproducible CPU token/media
parity assertions; `scripts/eval/audit_native_generation.py` loads the native
Thinker and generates both input variants. MOPD's **19 CPU** tests pass.
No full official `test_s`/`test_u` or matched no-OPD control was run.
