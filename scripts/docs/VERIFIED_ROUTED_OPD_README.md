# Verified routed OPD for REF-AVS

## Why this recipe exists

The original sparse audio/visual OPD run localized Teacher disagreement but
licensed both Teachers on every sample.  On the fixed 500-case diagnostic,
55.7% of Teacher-token terms came from a Teacher whose direct privileged
prompt did not improve that sample's video mIoU.  Token localization therefore
answered *where* while leaving *whether this Teacher is correct here*
unverified.

The new recipe combines:

1. cached privileged Teacher probe trajectories;
2. category-first, IoU-second licensing and routing;
3. verifiable class/mask reward for the Student;
4. one-Teacher-per-token hard routing;
5. sampled-token reverse-KL (`KL(Student || Teacher)`) through ms-swift's
   native signed Teacher/Student log-ratio GRPO token advantage;
6. an empirical difficulty mixture that keeps both ordinary RL-active samples
   and hard Student-all-wrong samples that a Teacher can rescue.

## Paper findings used

The attached VG-OPD paper uses a cached greedy probe from each expert, scores
Student and Teacher under the same criterion verifier, and licenses a Teacher
only when its counterfactual verifier gain is positive.  For policy-gradient
OPD, this implementation now follows ms-swift's native signed sampled-token
log-ratio:

```text
delta = log p_teacher(y_t) - log p_student(y_t)
A_KD  = verified_weight * delta
```

Positive deltas increase Teacher-preferred sampled tokens and negative deltas
suppress Teacher-disfavored sampled tokens.  The non-negative k3 estimator

```text
k3 = exp(delta) - delta - 1
```

is retained only for monitoring `KL(Student || Teacher)`; it is not detached
and reused as a PPO advantage.

The paper does **not** prescribe an easy/medium/hard mixture.  It mixes several
datasets, drops rows without a positive criterion, trains capability experts on
axis-filtered slices, and selects checkpoints by offline validation.  For
difficulty filtering we additionally use DAPO's useful observation that
all-equal reward groups have zero GRPO advantage.  We do not enable ms-swift's
native `dynamic_sample`, because it would also discard all-zero Student groups
that contain useful verified Teacher rescue signal.

## License and routing

For Student trajectory `s` and Teacher `k`:

```text
class_license_k =
    teacher_class_correct and not student_class_correct

iou_license_k =
    teacher_class_correct and student_class_correct
    and teacher_iou - student_iou > iou_margin

licensed_k = class_license_k or iou_license_k
```

The routing score is:

```text
score_k =
    2.0 * class_license_k
    + 0.75 * max(teacher_iou - student_iou, 0)
    + 0.25 * teacher_iou
```

The category-rescue bonus is larger than the maximum possible IoU contribution,
so a GT-correct Teacher always outranks a category-wrong Teacher.  Among two
category-rescuing Teachers, all three IoUs affect routing through gain and
absolute Teacher quality.  Licensed scores are softmaxed, converted into a
verified strength, and overlapping token masks are assigned to one Teacher.

Implementation:

```text
swift_plugin/verified_opd_routing.py
```

## Verifiable task reward

Student GRPO uses:

```text
R = 0.5 * class_exact + 0.5 * class_exact * mean_10_frame_iou
```

Mask reward is gated by class correctness.  This prevents a trajectory whose
text names the wrong target from receiving a large reward only because its
`[SEG]` hidden state overlaps the GT object.

Teacher probe scores remain separate:

* the cached Teacher trajectory decides licensing/routing;
* the frozen Teacher then scores the Student's own sampled token sequence under
  its privileged prompt;
* the Student never trains on Teacher-generated response tokens directly.

## Empirical difficulty refresh

Run `K` stochastic Student probes and one greedy probe per privileged Teacher.
The refresh script defines:

| Bucket | Definition |
|---|---|
| `rl_active` | Student reward std is non-zero |
| `kd_rescue` | Student group is uniformly poor and at least one Teacher predicts the GT class |
| `mastered` | Student reward is uniformly high |
| `hard_unverified` | Student uniformly fails and neither Teacher predicts GT |
| `intermediate` | Remaining samples |

Default materialized mixture:

```text
rl_active       45.0%
kd_rescue       30.0%
intermediate    15.0%
mastered         7.5%
hard_unverified  2.5%
```

The manifest records source and realized proportions, and sampling uses
inverse-square-root class frequency to reduce class imbalance.

Prepare refreshed data from Student and Teacher probes:

```bash
python scripts/data/prepare_verified_routed_opd_data.py \
  --source /path/to/privileged/train.jsonl \
  --student-probes /path/to/student_probes.jsonl \
  --audio-teacher-probes /path/to/audio_teacher_probes.jsonl \
  --visual-teacher-probes /path/to/visual_teacher_probes.jsonl \
  --output-dir /path/to/refreshed/data
```

## Training

```bash
DATA_ROOT=/path/to/refreshed/data \
RUN_ID=verified_routed_v1 \
MAX_STEPS=100 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/opd/run_sam2_verified_routed_opd_grpo_8gpu.sh
```

Defaults:

* 8 Student rollouts per prompt;
* BF16 compute;
* q/v LoRA only;
* frozen embedding, LM head, SEG projector and SAM2 decoder/memory;
* reference-policy KL beta `0.01`;
* verified KD coefficient `0.25`;
* bottom-10% Teacher likelihood within expected Teacher stages, dilated ±1;
* checkpoint every 25 steps;
* native DAPO resampling disabled in favor of the offline mixture.

## KL direction

Three related objectives must not be conflated:

1. ms-swift full-vocabulary GKD:
   * `beta=0`: `KL(Teacher || Student)` (forward KL);
   * `beta=1`: `KL(Student || Teacher)` (reverse KL).
2. The project's original sparse OPD used `beta=0` / full-vocabulary Forward
   KL.
3. The new joint trainer uses ms-swift's native signed sampled-token k1
   log-ratio as the routed reverse-KL policy advantage.  It does **not** use
   the full-vocabulary GKD loss and sets the base trainer's
   `teacher_kl_coef=0`, because routing applies the native signal explicitly.
   k3 remains an audit-only non-negative KL gauge.

## Audit outputs

Every rank writes:

```text
<output>/verified_routed_opd_audit/rank<rank>.jsonl
```

Each trajectory record contains:

* Student category correctness and IoU;
* cached Audio/Visual Teacher category correctness and IoU;
* category and IoU licenses;
* routing probabilities and verified weights;
* token pieces and CoT stages;
* seed and dilated masks for both Teachers;
* final one-Teacher-per-token assignment;
* token weights, signed Teacher/Student log-ratios, k3 monitoring values and
  actual KD advantages.

## Validation

The signed implementation satisfies three properties:

1. exact equality with ms-swift's native `compute_teacher_logratio`;
2. score-function gradient equality with the negative reverse-KL gradient on a
   categorical distribution;
3. positive and negative routed advantages and zero unlicensed advantages.

## Required ablations before a full claim

Run matched 50/100-step experiments:

1. GRPO only;
2. verified reward + ungated localized OPD;
3. category-only license;
4. category + IoU license;
5. verified license + random same-coverage tokens;
6. verified license + hard routing;
7. shuffled privileged evidence.

This separates gains from:

* verifiable task reward;
* Teacher eligibility;
* token localization;
* routing;
* privileged information content versus generic regularization.
