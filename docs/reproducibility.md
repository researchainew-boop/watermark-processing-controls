# Reproduction of the Conference Study

## Evidence versions and status

All Phase 5 analyses are complete. The current manuscript compiled to five
IEEE pages including references, with 24 references, six tables and three
figures. All five pages were rendered at 120 dpi and visually inspected;
no overlap or clipping was found. The combined numerical, formatting,
embedded-font and source/PDF fingerprint gate passed; see
[the saved Phase 5 submission check](../outputs/phase5/submission_check.json).
Public reviewer hosting remains pending. The
previous six-page source/PDF remains in `manuscript/archive/pre_phase123_20261009.*`.
The source before the new controls is preserved in
`manuscript/archive/pre_phase4_20261010.tex`.

The evidence scopes are:

- `outputs/conference_clean/`: retrospective correction, 5,823 sources and
  the original 26,880-copy/1,680-source test.
- `outputs/extension/` and `outputs/extension_controls/`: completed encoding,
  frozen 192-source sample, original published baseline, and the original
  24-AI/24-real watermark/diffusion comparison.
- `outputs/phase3/controls/`: completed fixed-head unmarked-edit control,
  46,424-copy original-source test, and five 49,158-copy repeated tests with
  another 1,000 Open Images sources.
- `outputs/phase3/native_baseline/`: completed native-resolution contrast on
  the same 192 sources, with the unchanged published detector.
- `outputs/phase3/watermarks/`: completed enlarged local comparison,
  204 AI/204 real sources, 21,216 outputs and 42,432 decoder rows.
- `outputs/phase3/regeneration/`: completed one-payload regeneration on
  204 AI sources only, with 408 covers and 1,224 outputs.
- `outputs/phase4/size_controlled/`: completed forty CPU probe fits across
  four training treatments, two backbones and five source assignments.
- `outputs/phase4/domain_generalization/`: completed six detector configurations,
  independent fusion/calibration source halves and collection-transfer reporting.
- `outputs/phase4/watermark_reporting/`: completed reanalysis of fixed decoder
  outcomes, with default and any-three-payload unmarked false positives.
- `outputs/phase4/fresh_replication/`: completed locked replication of 400
  unused items, with 2,560 copies and unchanged heads/thresholds.
- `outputs/phase5/matching_factorial/`: completed forty balanced factorial fits.
- `outputs/phase5/calibration/`: completed independent fusion/calibration and
  supervised SD-family-exclusion controls with fixed policy thresholds.
- `outputs/phase5/paired_content/`: completed 200-pair prospective test,
  400 source images and 1,200 unmarked copies, plus post hoc overlap diagnostics.
- `outputs/phase5/fair_watermarks/`: completed common nominal-FPR/distortion
  comparison on 1,004 sources and 8,912 images.

All requested Phase 3, 4 and 5 analyses have completed. Read measurements from the final
summaries linked below; partial files remain intermediate evidence.
CIFAKE/CIFAR pilots and demonstration outputs are excluded from manuscript
measurements.

The new replication's `summary.json` reports unchanged models and thresholds.
It is a new-item replication from the same five collections, not an
unseen-generator evaluation.

## Inputs and environments

`configs/study.yaml` points to the existing sibling project's
`data/raw/`. For full reproduction, adapt
paths in a separate working copy and preserve raw images, source/derived
manifests, standardized bases, feature caches, fits, thresholds, and provenance.
Collection scripts that clear destinations belong in separate data locations.

The original CPU correction records Python 3.12 and scikit-learn 1.9.1;
its dependencies include NumPy, pandas, OpenCV, SciPy, PyWavelets, PyYAML,
threadpoolctl, and the project source. The experimental environment records
NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.6.1, PyTorch 2.5.1+cu121,
torchvision 0.20.1+cu121, OpenCLIP 2.26.1, TrustMark 0.9.2, Diffusers 0.35.1,
Transformers 4.46.3, and Accelerate 1.2.1. Use each artifact's actual recorded
versions and checkpoint identities; the environments differ.

The five-seed marker identifies reused seed heads, whose training/validation
design remains unchanged when test-only copies are added. Its package versions
describe the analysis runtime. New feature extraction checks checkpoint/model
identity and comparability with original caches; backbones remain frozen.

The slot-controlled fits use the experimental NumPy 1.26.4/scikit-learn 1.6.1
CPU runtime, while the domain diagnostic records NumPy 2.5.3/scikit-learn 1.9.1.
Follow their recorded environments rather than combining the two dependency
trees. The optional new-replication archive selection also requires PyArrow
19.0.1 to read existing Midjourney parquet files. Its presence does not imply
permission to redistribute dataset images or metadata. PyArrow 19.0.1 is also
used by the optional public-source-index reconstruction.

### Audited public-item index

`outputs/phase4/reproduction_index/` identifies exact public items without
images, absolute/local paths, URLs, usernames, prompts, captions or photo-author
metadata. The index contains 7,592 unique pseudonymous sources: the original
6,000 (including 177 exclusions), 1,000 added Open Images items, the older
192-source fresh sample and the new 400-source replication. In total 7,415
records are retained. Separate CSVs contain all 6,000 original split assignments
and 34,115 assignments over the five 6,823-source retained pools. Both fresh
samples have blank original-split fields and do not enter those training pools.

    python scripts/build_reproduction_index.py

The builder replays the original ZIP member and parquet row order, minimum
side 256 and parquet label-zero filter. Each of the 4,000 eligible generated
exports must match the original audit's decoded pixels after the original RGB
PIL JPEG95 export. All 2,000 original COCO archive entries are also checked
against native decoded pixels. Every standardized source's pixel/file
fingerprint is verified, and ZIP entries, parquet bounds, unique pseudonyms and
split pools are checked. The completed run took 236.43 seconds without model
inference or network acquisition.

Local `mj-000.parquet`, `mj-001.parquet` and `mj-002.parquet` aliases correspond
to public `test-00000-of-00030.parquet`, `test-00001-of-00030.parquet` and
`test-00002-of-00030.parquet` in the dataset's `data` directory. Their entire
file SHA256 values were verified against the canonical public shard identities.
Parquet rows in the index are zero based. COCO IDs retain twelve digits; Open
Images IDs retain their public hexadecimal identifiers.

Completion requires `index_audit.json` with `all_checks_passed: true` and
`status: complete_unambiguous`, plus matching fingerprints for
`public_source_index.csv`, `original_split_assignments.csv` and
`five_seed_split_assignments.csv`. The package includes only these allowlisted
public fields after its own validation. Reacquisition still depends on the
cited collections and their terms; this unpublished local index makes no image
redistribution or licensing assertion. Raw original AI pixel fingerprints refer
to JPEG95 exports; added-source raw hashes retain their saved native/rotation
scope, which the audit describes.

## Historical correction and numerical conventions

To reconstruct the original correction in a separate working copy:

    python scripts/prepare_conference.py --samples 2000

The screen forms components from native/standardized exact-pixel matches and
8×8 average-hash pairs at Hamming distance at most two. It retains one member
by original train/validation/test priority and source ID and excludes all
its duplicates' derivatives. Original splits and feature caches are preserved.
Probes fit training; fusion and thresholds use eligible validation rows.
Midjourney and the three held-out settings remain excluded from fitting.
The correction follows inspection of earlier results and is retrospective.

Logistic ROC ranking uses saved linear decision scores evaluated in float64.
Fixed-threshold decisions use probabilities and validation thresholds.
Payload ranking uses raw agreement. Earlier probability-ranking results are
preserved in `outputs/conference_clean/metrics_probability_ranking.json`.
To refresh original statistics without refitting:

    python scripts/prepare_conference.py --refresh-statistics --samples 2000

Paired source-bootstrap draws are stratified by source collection. TPR at
5% FPR is a descriptive test-ROC point; fixed validation-threshold errors are
separate. Cache refits reproduce analysis conditional on saved features.
Independent end-to-end reproduction also requires acquisition, preprocessing,
checkpoints, and encoder inference.

## Experiment 1: controls and five source splits

The fixed-head control adds fourteen unmarked edits for each of 1,396 original
AI test sources: 19,544 rows. Its full expanded test has 46,424 rows from
1,680 sources. Corresponding marked/unmarked AI edits are compared against
identical real-edit rows and settings. Original coefficients stay unchanged.

The repeated-split pool adds 1,000 Open Images sources to the 5,823 original
screened sources. Exact-pixel and stated average-hash screens exclude overlaps
against all original 6,000 files, the frozen 192, and within the new set.
Metadata rotation precedes standardization. Five seeds, 20260920–20260924,
each allocate Open Images 700/150/150, with all 962 Midjourney sources in test.
Each has 4,102 training, 878 validation, and 1,843 test sources.

Train/validation each use thirteen original conditions. Test uses all sixteen
original conditions and fourteen unmarked AI edits: thirty copies per each
of 1,405 AI sources, sixteen per each of 438 real sources, totaling 49,158.
Every derivative follows its source split. Unmarked edited copies and the
three held-out settings stay excluded from fitting. Fusion uses validation;
the payload gate stays at 13/16.

Ordered commands:

    python scripts/phase3_prepare_controls.py --stage all
    python scripts/phase3_prepare_heldout.py
    python scripts/phase3_prepare_unmarked_union.py
    python scripts/phase3_extract_controls.py --which both --batch-size 16
    python scripts/phase3_analyze_controls.py --stage all --backbone both
    python scripts/phase3_report_controls.py

Preparation/extraction reuse locked completed files. New features include
6,939 missing held-out CLIP rows and 19,600 extra unmarked edits from 1,400
AI sources in the union of five tests; both backbones are extracted for the
latter. Matrix row identities and manifest hashes are checked.

Required final evidence includes `unmarked_frozen_results.json`,
`split_seeds/mean_std_metrics.csv`, `split_seeds/complete.json`, and
`summary.json`. The completion marker fingerprints the unchanged original
fourteen heads, new heads, source assignment, features, and results. It records
285 per-seed metric rows and 57 mean/SD summaries, including separate COCO
and Open Images results. Intermediate marked-only tests remain in
`split_seeds/marked_only_seed_archive/`.

The ± quantities are sample SD (`ddof=1`) over five seeds. Tests overlap and
reuse the Midjourney holdout, so SD measures split sensitivity rather than an
uncertainty interval or independent external replication. See
[phase3_controls.md](phase3_controls.md).

### Training-slot-controlled follow-up

The primary protocol precedes the first thirty CPU fits. A separately frozen
amendment precedes ten JPEG-only fits. The full and two equal-slot arms have
53,326 rows from the same 4,102 sources, with identical labels, class weights,
regularization, solver and iteration cap. Each arm keeps all AI rows and the
real clean/opposite-payload rows. The two controls replace eleven transformed
real slots per source with either its repeated clean vector or alternating
JPEG70/JPEG50 vectors (six/five slots). The reduced-row baseline removes those
slots and has 30,842 rows. All arms use unchanged 11,414-row validation and
49,158-row expanded test cohorts.

On a new working copy with the locked Phase 3 caches:

    python scripts/phase4_size_controlled.py --prepare-only
    python scripts/phase4_size_controlled.py
    python scripts/phase4_size_controlled.py --jpeg-only

The completed research checkout protects four-arm results from a three-arm
rerun. The JPEG-only entry point reuses unchanged saved fits when present. The
first protocol, scientific source and aggregates remain in `primary_archive/`
and `initial_script_snapshot.py`; the amendment binds their hashes.

All forty heads converged without warnings, at most 175 iterations. Full and
removal refits match earlier coefficients exactly. ResNet AUC is
0.928834 ± 0.000960 for full training, 0.882160 ± 0.011464 for clean repetition,
and 0.898495 ± 0.004031 for JPEG repetition. Full-minus-control AUC differences
are 0.046675 [0.043516, 0.050108] and 0.030340 [0.027564, 0.033118]. CLIP AUC
is 0.986326, 0.971577 and 0.974128, respectively.

Thresholds are selected at empirical validation FPR ≤5% and then fixed on
test. Report actual test FPR alongside test TPR; the separately named test-ROC
TPR5 is descriptive. ResNet test TPR at the validation threshold is
0.739004/0.607265/0.688584, with actual test FPR
0.056564/0.074772/0.085987 for full/clean/JPEG training. The 2,000 paired
source-cluster resamples preserve source overlap across all five tests and
condition on fitted heads and validation thresholds. Equal slots do not
equalize augmentation diversity or isolate the unique effect of exact matching.
See [size_controlled_ablation.md](size_controlled_ablation.md) and
`outputs/phase4/size_controlled/complete.json`.

### Equal-size collection transfer and calibration

The domain diagnostic uses seed 20260920 and cached features. COCO-only,
Open-Images-only and joint training each use 700 real sources; the joint regime
uses 350 from each collection. Synthetic training and all condition slots are
unchanged. Two backbones give six probe/fusion configurations, with probe,
fusion and gated outputs for each. Fusion fitting and threshold calibration
use separate source halves of validation; neither fitted head sees calibration
sources. Each half has 75 real sources, and the joint regime uses 37/38 and
38/37 sources per collection. Test has the same 150 COCO, 150 Open Images and
1,405 AI sources across all regimes: 46,950 copies.

    python scripts/phase4_domain_generalization.py --stage frozen-calibration
    python scripts/phase4_domain_generalization.py --stage domains --backbone both
    python scripts/phase4_domain_generalization.py --stage attack-scope

The first stage separately locks operating thresholds for the already fitted
Phase 3 seed-20260920 joint full heads. That existing validation set is disjoint
from probe training, but was used to fit existing fusion heads: fusion/gate
calibration is retrospective. These frozen thresholds are used unchanged for
the new source replication; the new domain fits are not substituted into it.

All new domain systems attain calibration FPR 0.04923 on separate sources.
On matched thirteen-condition test scope, COCO-trained CLIP has photographic
FPR 0.0600 against COCO and 0.30923 against Open Images; joint training gives
0.03949/0.09949. Adding the three held-out test conditions changes the scope;
those rates are reported separately. Equal-size joint training improves ResNet
ranking but does not eliminate calibration transfer failure. Selecting a
different regime after observing these results would not be independent
validation. Intervals use 1,000 source-cluster draws, conditional on the fits.
Required evidence includes `summary.json`, `complete.json`, `metrics.csv`,
`frozen_seed_thresholds.json`, and `attack_scope_false_positives.csv`. See
[domain_generalization.md](domain_generalization.md).

## Experiment 2: frozen sample and native baseline

The earlier ordered procedure is in [extension_run.md](extension_run.md).
Common encoding uses 600 sources under four conditions. The 192-source fresh
sample was locked before its first original-head inference. Its reported
errors remain in the record; its images/scores never enter new fits.

For the retrospective native-resolution contrast:

    python scripts/native_baseline.py --stage prepare
    python scripts/native_baseline.py --stage evaluate --device cuda --batch-size 4
    python scripts/native_baseline.py --stage analyze

Original file and decoded-pixel hashes must match the locked records. Official
unchanged UniversalFakeDetect ViT-L/14 applies CenterCrop(224), ToTensor,
and CLIP normalization, without Resize or feature normalization. Score
polarity and the published logit-zero threshold remain unchanged. Completed
inference is protected from overwriting.

Native AUC is 0.6593 versus 0.4423 after 256×256 resizing and common JPEG95
on the same sources. The paired difference is 0.2170, with source-bootstrap
95% interval [0.1473, 0.2915]. Both resolution and JPEG encoding change,
so failure attribution remains unconfirmed. Collection-specific results,
predictions, checkpoint hashes, and timings are preserved. See
[native_baseline.md](native_baseline.md).

### Prospective new-item replication

The frozen protocol selects 400 unused source items, 80 per collection: COCO,
Open Images, Stable Diffusion, DALL-E 3 and Midjourney. The four existing joint
full probe/fusion heads from seed 20260920, the gate at 13/16, and the existing
validation thresholds are bound before source selection. There is no
refitting or threshold tuning on the new sample. All items receive the same
INTER_AREA resize to 256 square and JPEG95 round trip, with lossless PNG
staging. Metadata orientation is applied before standardization for Open
Images.

The four conditions are identity, JPEG50, crop and a 32×32 round trip. Real
items have four unmarked copies; AI items have eight copies (marked/unmarked
per condition), totaling 2,560. Selection uses fixed hash order of public item
IDs and minimum side 256; original eligible archive exports and earlier source
IDs are excluded before exact-pixel/average-hash overlap screening. No source
is chosen from detector scores. Pretraining and semantic overlap remain
unexcluded.

After frozen calibration, execute each stage once:

    python scripts/phase4_fresh_replication.py --stage select
    python scripts/phase4_fresh_replication.py --stage derive
    python scripts/phase4_fresh_replication.py --stage extract --device cuda --batch-size 16
    python scripts/phase4_fresh_replication.py --stage analyze

Selection, derivation, inference and analysis are complete. Completion evidence is `selection_complete.json`,
`derivation_complete.json`, both backbone inference records, `predictions.csv`,
and `summary.json`. The source protocol specifies one scoring pass and no
test retuning. Both GPU feature runs passed the frozen-cache anchors. The
2,000 paired source-cluster draws preserve all copies of a source, stratify by
collection, and compare probe, ordinary fusion and gate at fixed thresholds.
Their intervals are conditional on the frozen heads.

For unmarked identity images, ResNet probe AUC is 0.9653125
[95% CI 0.9483307, 0.9798203], TPR 0.8083333 and actual test FPR 0.03125.
CLIP probe AUC is 0.9944531 [0.9873171, 0.9985677], TPR 0.9833333 and
actual test FPR 0.08125. Over the four conditions, unmarked probe AUC is
0.9567301/0.9884635 for ResNet/CLIP, with actual FPR 0.0515625/0.1421875.
These FPRs are observed transfer outcomes, not thresholds selected on test.
The gate approximately preserves the unmarked probe ranking, while ordinary
fusion lowers it. New items strengthen within-collection replication; they do
not establish generalization to unseen generators or remove pretraining,
scene-content and collection effects.

## Experiment 3: larger watermark and regeneration comparison

The local design selects 204 AI test sources (68 per generated-image
collection) and 204 distinct COCO controls using maximum-total-CLIP-cosine
assignment. These previously examined sources support retrospective
robustness analysis.

Payloads are `1011010010110100`, `0100101101001011`, and
`1100101000110101`. Classical strength is 12; official TrustMark Q strength
is 1 with BCH5. Classical verification accepts thirteen of sixteen expected
bits. TrustMark requires valid ECC, the expected schema, and an exact 61-bit
data message: sixteen payload bits followed by forty-five zeros. Their
capacities and criteria differ.

    python scripts/phase3_watermarks.py --stage select
    python scripts/phase3_watermarks.py --stage prepare --device cpu --threads 2
    python scripts/phase3_watermarks.py --stage attacks
    python scripts/phase3_watermarks.py --stage analyze --device cuda --batch-size 16

There are 1,632 covers and thirteen conditions: identity, JPEG75/50, two
blurs, fixed noise, crop, 32×32 round trip, grid shift with JPEG/noise,
spectral wipe, and Gaussian sigma-6 noise seeds 20261009–20261011.
The completed design has 21,216 outputs and 42,432 decoder rows. Read
[the final local summary](../outputs/phase3/watermarks/summary.json) and
the associated source-level intervals. TrustMark decoding ran on CPU for
13,840 outputs and CUDA for 7,376 outputs, with the device saved per row.
Both use the same official FP32 weights, preprocessing, and ECC, with
public-API decode anchors on each device. Resumption changes neither the
verification criterion nor tuning.

The larger regeneration uses the default payload and both marks on 204 AI
sources only: 408 covers and 1,224 completed outputs, without new real controls.

    python scripts/phase3_regeneration.py --stage run --mode auto
    python scripts/phase3_regeneration.py --stage analyze --device cuda --batch-size 16

The final `complete.json` and `summary.json` are required before manuscript
generation. [The completed regeneration summary](../outputs/phase3/regeneration/summary.json)
reports VAE recovery of 76/204 DWT–DCT and 21/204 TrustMark sources,
diffusion-0.25 recovery of 10/204 and 0/204, and diffusion-0.50 recovery of
6/204 and 0/204. Mean PSNR at strength 0.50 is 16.07/16.14 dB against the
marked cover. [phase3_regeneration.md](phase3_regeneration.md) records the
checkpoint, scheduler, precision, distortion references, and limits.
The earlier 24-pair experiment and its 288 diffusion outputs remain separate
under `outputs/extension/`, with their own unmarked AI/real controls.

### Explicit unmarked-control false positives

The reporting follow-up reuses the saved 42,432 decoder outcomes, with no new
decoder inference or test-selected threshold:

    python scripts/phase4_watermark_reporting.py

It queries each default payload and the any-of-three acceptance event on the
same unmarked controls, rather than counting payload queries as independent
samples. Source-level intervals preserve repeated conditions; per-condition
and any-condition exact binomial bounds prevent treating zero observed events
as zero population risk. DWT-DCT requires ≥13/16 expected bits; TrustMark
requires BCH5 ECC, schema 1 and the exact expected 61-bit message. These rules
are not matched-FPR operating points or a cryptographic security demonstration.

Across thirteen conditions, mean photographic acceptance is 0.010935 for the
default DWT payload and 0.031674 for any of three. Corresponding unmarked-AI
means are 0.015083 and 0.037330. For the source event of any acceptance across
all thirteen conditions, photographic counts are 19/204 for the default and
46/204 for any three; this differs from the per-condition mean. TrustMark has
0/204 sources under either event, with a two-sided 95% upper bound 0.017920.
The overall zero-event bootstrap can be degenerate, so use the exact
source-level bound when interpreting risk. The 204-source regeneration run
still has no new unmarked real controls. Required evidence is `summary.json`,
`analysis_audit.json`, `control_per_condition.csv`, `control_overall.csv`,
`cross_method_acceptance.csv`, and `table_ready.csv` under
`outputs/phase4/watermark_reporting/`.

## Public source identifiers and split assignments

`outputs/phase4/reproduction_index/` contains three audited CSVs:

- `public_source_index.csv`: 7,592 public acquisition records, including
  177 excluded original records and 7,415 retained records.
- `original_split_assignments.csv`: all 6,000 original selected records,
  with the retained flag marking 177 exclusions.
- `five_seed_split_assignments.csv`: 34,115 assignments, covering the same
  6,823 retained original/Open Images sources for each of five seeds.

The acquisition inventory comprises the 6,000 original selections, 1,000
additional Open Images photographs, the 192-source sample, and the new
400-source replication. Both fresh samples have blank original splits and
never enter repeated-split fitting. Inventory counts are not independent
sample counts for any reported experiment.

The index uses pseudonymous study IDs, collection names, public archive
basenames/member names or zero-based parquet rows, public photo IDs,
retention/split fields, and fingerprints. It contains no images, absolute
paths, URLs, prompts, captions, usernames, or photo-author metadata. Obtain
inputs through the cited public collections under their terms.

The builder replays exact original eligible-export order, including
minimum-side filtering and JPEG-95 export, and validates all 4,000 original
AI exports, 2,000 original COCO pixel fingerprints, and 7,592 standardized
pixel/file hashes. Canonical public Midjourney shard hashes are checked
against the local cache aliases. `index_audit.json` records these checks,
input/archive hashes, library versions, fingerprint conventions, and the
three CSV hashes. Rebuilding requires the locally acquired archives and
existing image files; it does not repeat model inference:

    python scripts/build_reproduction_index.py

The anonymous package stores the audited CSVs under `reproduction_index/`
and its audit under `evidence/phase4/reproduction_index/`. This provides
input identification; no independent end-to-end reproduction or permission
to redistribute images is claimed.

## Phase 5: processing matching, independent calibration and stronger controls

Use a separate output archive for independent reconstruction. The commands
below describe the original stage order; do not refit or rerun generation over
the completed research outputs. The portable snapshot excludes all feature
caches, fitted heads, private manifests and benchmark image bytes, so supplying
the recorded inputs and reacquiring the pinned models is necessary.

### Balanced processing-matching factorial (completed)

    python scripts/phase5_matching_factorial.py --prepare-only
    python scripts/phase5_matching_factorial.py --bootstrap-samples 2000

The 40 CPU fits use clean plus two unmarked edit families per source. AA/BB
match the real/AI edit sets; AB/BA mismatch them. Every cell has identical
source/row/class-weight/solver settings and is scored on the same five-view
test. Only 1,834 non-Midjourney AI sources have the required fitting caches;
all 962 Midjourney sources stay test only. The five tests overlap. Consult
[matching_factorial.md](matching_factorial.md), `protocol.json`,
`complete.json`, `convergence.json`, and the three aggregate CSVs under
`outputs/phase5/matching_factorial/`. Bootstrap uncertainty preserves all
derivatives and split memberships in the 4,238-source test union.

### Source-disjoint fusion fitting and calibration (completed)

    python scripts/phase5_calibration.py --stage prepare
    python scripts/phase5_extract_calibration.py
    python scripts/phase5_calibration.py --stage fit
    python scripts/phase5_calibration.py --stage calibrate
    python scripts/phase5_calibration.py --stage evaluate
    python scripts/phase5_calibration.py --stage loo-fit
    python scripts/phase5_calibration.py --stage loo-calibrate
    python scripts/phase5_calibration.py --stage loo-evaluate
    python scripts/phase5_calibration.py --stage source-event-intervals

The 792 missing calibration CLIP features were extracted once on GPU and
their completion and checkpoint hashes are recorded. Other stages used the
CPU environment. Each validation collection is split by
the fixed SHA ordering into fusion-fit and calibration source halves. Original
probes remain fixed. `heads_locked.json` precedes `thresholds_locked.json`,
which precedes evaluation. No test threshold tuning is allowed. All ten main
fusion fits, ninety policies, and four additional SD-exclusion fits with
eighteen policies are complete. The supervised SD-exclusion fit uses only
DALL-E synthetic training examples; pretrained representations may still have
seen SD-family imagery. The `loo-*` names are historical command names for
that specific control, not leave-one-out over all generator families.

`per_image_core13`, `conformal_core13`, and `conformal_all16` are distinct
precommitted policies. Conformal policies calibrate each photograph's maximum
over the named processing family separately within COCO and Open Images,
then deploy the larger collection cut as one global threshold with strict
ties. The finite-source bound is marginal under exchangeability, not a
high-confidence bound on every saved threshold. The all16 policy includes
three views held out from fitting and is a retrospective diagnostic. The
already examined Phase 4 fresh400 sample is a follow-up, not untouched again.
Zero-event collection intervals are reported separately and do not turn a
degenerate bootstrap into evidence of zero population error. See
[phase5_calibration.md](phase5_calibration.md).

### Caption-paired prospective sample (completed)

    python scripts/phase5_paired_content.py --stage protocol
    python scripts/phase5_paired_content.py --stage select
    python scripts/phase5_paired_runtime.py --stage lock
    python scripts/phase5_paired_runtime.py --stage generate
    python scripts/phase5_paired_runtime.py --stage derive
    python scripts/phase5_paired_runtime.py --stage extract
    python scripts/phase5_paired_runtime.py --stage analyze

Selection locks 200 previously unused COCO photographs, their lowest-ID public
caption annotation, and one deterministic generation seed each. The generated
image is not conditioned on the photograph pixels. The fixed RV/LCM SD1.5
pipeline uses eight steps, guidance one and 512-square output. GPU generation
requires PEFT 0.17.0 with Diffusers 0.35.1; the initially installed PEFT 0.13.2
is incompatible with this Diffusers LoRA loading and was superseded before
generation. The pinned LCM adapter revision and hash are recorded in
`generation_assets_lock.json`. Native output placement and numerical settings
are recorded by the runtime. Both origins receive identical area256/JPEG95
recoding. Identity, JPEG50 and lowres32 give 1,200 completed copies from 200
pairs/400 origin sources, with no watermark embedding. The locked joint and
supervised SD-excluded heads are both reported without selecting a winner on
these pairs. This is a different SD1.5 pipeline, not a new generator family.

`generation_complete.json`, `derivation_complete.json`, both inference
records, `summary.json` and `complete.json` document the finished run. The
joint ResNet/CLIP identity AUCs are 0.7292 [95% paired intervals 0.6857,
0.7731] and 0.8200 [0.7849, 0.8554]. Their image-calibrated TPR/FPR is
0.170/0.045 and 0.310/0.020. The all16 source-max cuts reduce identity TPR
to 0.020/0.025 with zero accepted photographs out of 200. Zero observed
errors do not establish zero future risk. CLIP's lowres32 image-policy FPR
is 0.325, demonstrating processing shift despite fixed calibration.

This test shows limited transfer to this particular RV/LCM pipeline. Content
pairing and generator configuration change together; neither alone is
established as the cause. A postgeneration audit detected no exact pixel
matches, but three generated images had ten prior coarse-aHash matches.
All 200 locked pairs remain primary. The post hoc 197-pair sensitivity gives
ResNet/CLIP identity AUC 0.7327/0.8227 without changing models or thresholds.
Both safe aggregate reports are preserved separately.

The audited public pair index supplies caption IDs and seeds, not
caption/prompt text. It identifies inputs; execution is established by the
completion and inference records. The superseded DDIM50 protocol remains
archived; it was replaced before selection/outcomes. See
[paired_content_control.md](paired_content_control.md).

### Common nominal FPR and distortion watermark comparison (completed)

    python scripts/phase5_fair_watermarks.py --stage select
    python scripts/phase5_fair_watermarks.py --stage prepare-cpu
    python scripts/phase5_fair_watermarks.py --stage derive-cpu
    python scripts/phase5_fair_watermarks.py --stage prepare-learned --device cuda --batch-size 16
    python scripts/phase5_fair_watermarks.py --stage derive
    python scripts/phase5_fair_watermarks.py --stage decode-calibration --device cuda --batch-size 16
    python scripts/phase5_fair_watermarks.py --stage calibrate
    python scripts/phase5_fair_watermarks.py --stage decode-test --device cuda --batch-size 16
    python scripts/phase5_fair_watermarks.py --stage analyze

There are 400 calibration photographs, 400 distinct test photographs (200 each
COCO/Open Images per split), and 204 AI test sources. Three fixed 16-bit
payloads are evaluated over four conditions. Original collection encoding
histories remain despite shared JPEG95 recoding. Both marks are adjusted using
the frozen distortion-only residual/rounding rule to PSNR40 dB ±0.1. TrustMark
residual rescaling is outside its official encoding configuration. The raw
first-sixteen information margins, default query and any-of-three query each
have separate method-specific cuts at nominal calibration FPR5%. Different
native ECC/bit rules are secondary. Test FPR is observed and need not match
between methods. All decoder/payload/derivative repetitions stay within the
source-level uncertainty calculation.

Encoding, derivation, calibration/test inference locks, thresholds and the
final summary are complete. The run produced 8,912 images, 440 rate records,
30 paired differences and eight quality aggregates. Calibration accepted
80/1,600 images for each method/query rule; thresholds were frozen before
test decoding. No candidate selection or test retuning occurred.

Across four conditions, default-payload signed-margin acceptance is 0.497549
for DWT-DCT and 0.915441 for TrustMark; their paired difference is 0.417892
[95% source interval 0.397059, 0.438725], conditional on the fixed cuts.
Actual photographic false-match rates are 0.046875 and 0.053750, so the
shared nominal calibration target does not imply equal test errors. The
default positives use the first embedded payload, while the any-of-three
positive population includes all three payloads.

Partial 16-bit signed-margin evidence must remain separate from full-message
recovery. After lowres32, TrustMark accepts 136/204 partial-margin scores,
but only 5/204 recover the exact 61-bit message with valid BCH5/schema1.
Embedding PSNR satisfies the fixed 40 dB tolerance, while SSIM still differs.
TrustMark residual rescaling is outside its official encoding configuration;
the results apply to these rescaled, quantized encoders and four conditions.
See [phase5_fair_watermarks.md](phase5_fair_watermarks.md).

### Final Phase 5 numerical audit

    python scripts/check_phase5_submission.py --component all

This checks the recorded completion/protocol/hash relationships and aggregate
numerical identities. It refuses missing or changed completion evidence.
It does not train models or replace a PDF visual review.

## Figure assets and manuscript generation

The pipeline strip uses an original project-generated image processed by the
actual DWT–DCT implementation, JPEG50, crop, 32×32 round trip, and DDIM strength
0.50. It is excluded from benchmark aggregates. Its prompt, file identities,
processing, and exclusion are in
`manuscript/assets/generated-asset-provenance.json`. The log-FPR ROC figure
uses only the completed caption-paired identity test: 200 pairs and 400
unmarked sources. It displays the saved joint ResNet probe/fusion/gate and
joint CLIP probe logits. No refit, encoder inference or test threshold
selection occurs during plotting. Its provenance binds the predictions,
locked manifest, completion report, summary, heads, plotting code and output
PDF/PNG hashes. Zero FPR is displayed at 0.001 and identified in the figure;
the smallest nonzero empirical FPR is 1/200. The previous expanded-test ROC
and its provenance are preserved under `manuscript/archive/pre_phase5_*`.
The reviewer package records this plotting environment separately in
`requirements/recorded-figures.txt` (Matplotlib 3.9.3 with the CPU correction
NumPy/scikit-learn/SciPy versions).

After all final control, baseline, local-mark, regeneration, slot/domain/control
reports, new replication summaries and
figure measurements are complete:

    python scripts/phase3_regeneration.py --stage example_prepare
    python scripts/build_phase3_figures.py --stage pipeline
    python scripts/phase5_roc_figure.py
    python scripts/build_conference_paper.py

The legacy figure builder's `roc` and `both` stages write the former
expanded-test ROC. Use its `pipeline` stage for the final figure sequence;
the Phase 5 builder then writes the approved caption-paired ROC.

The build uses the conference template/bibliography, completed metrics,
result adapters, and figure assets. Keep prose and reference edits in the
input files. Generation alone does not verify compilation, five-page length,
or figure readability. Compile `manuscript/main.tex`, inspect every PDF page,
confirm tables/figures/references, and record final source/PDF fingerprints.
The current Phase 5 revision has completed five-page compilation, visual
review of all five rendered pages and its full numerical/font/source/PDF gate.
The saved `outputs/phase5/submission_check.json` report records that checked
revision. Source/PDF changes require new final checks and a refreshed release.
The archived Phase 4 October 10 revision had five US Letter pages,
22 references, five tables and three figures; that earlier formatting result
applies only to its archived source/PDF. Neither local check constitutes IEEE
PDF eXpress certification. Prior sources remain archived.

## Anonymous package and future work

`release/anonymous-code.zip` contains allowlisted code, portable configuration,
checkpoint fingerprints, public source indices/splits, aggregate reports,
and recorded versions. Dataset images, private metadata, embeddings, and
weights are excluded.
Generated illustration assets require provenance. Refresh the final package
with `--require-completed-phase3 --require-completed-phase4
--require-completed-phase5` after every
analysis finishes; approved
figure assets are supplied explicitly to the packager.

No public anonymous URL is available. Public pair IDs/seeds use their own
strict field allowlist; benchmark images, caption/prompt text and private
predictions stay excluded. See [release_export.md](release_export.md).
Archive checks are separate from independent data/end-to-end reproduction.
Compilation, visual review and the final numerical/font/fingerprint gate are
complete. Verified public reviewer access remains pending.
Further research includes additional keyed/learned
marks, published removal attacks, unseen-collection/generator validation,
and a licensed benchmark release.
Follow [venue_selection.md](venue_selection.md).
