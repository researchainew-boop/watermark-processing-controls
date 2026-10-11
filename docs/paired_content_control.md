# Prospective caption-paired content control

## Purpose and current scope

This control aligns the described scenes of real and generated images while
holding the final image processing constant. It uses 200 unused COCO val2017
photographs and one generated image per photograph, with identity, JPEG50 and
a 32×32 round trip applied to both origins. All inputs are unmarked: 400 origin
sources and 1,200 copies. Identity is the primary endpoint.

The 200-photograph selection is complete and passed the overlap screen before
generation or detector scoring. Of the candidates considered, 176 were
previously used public IDs, eighteen had prior/within-sample hash overlap and
two were smaller than the minimum side. The saved protocol and selected
sources precede generation and detector scoring. The run is now complete:
generation and derivation completion records, both inference records,
predictions, 144 metric rows and a final summary are present. An independent
numerical check verifies the actual hashes, frozen heads/cuts, denominators and
paired-bootstrap calculations. Earlier measurements are preserved separately.

## Selection and scene pairing

The public source index identifies all 7,592 previous acquisition records,
including excluded original items. Every previously used COCO public photo ID
is excluded. Candidates have minimum side 256 and are considered in SHA256
order using seed 20261010 and the twelve-digit COCO ID. Native or standardized
exact-pixel matches and 8×8 average-hash Hamming distance at most two exclude
overlap with all previous records and earlier selected photographs.

The official `annotations/captions_val2017.json` supplies the lowest integer
caption annotation ID for each selected photograph. The prompt is exactly
`Photograph: ` followed by that caption. Neither caption editing nor generated
candidate resampling is allowed. Caption/archive/member fingerprints, selected
photo fingerprints, generation seeds, the prior exclusion manifest and all
pair IDs are saved before generation. Captions and image pixels remain local
and are outside the current anonymous package's public index.

All pairs are test only. Their two origins and every derivative stay in the
same pair cluster; neither member enters fitting or calibration. The runtime
supplement locks two fitting regimes before generation: joint-collection
Phase5 heads from split seed 20260920, and additional heads excluding Stable
Diffusion and Midjourney from every supervised fitting stage. The latter fit
uses DALL-E 3 as its only synthetic collection. In each regime, fusion fitting
and threshold calibration use disjoint validation sources. The image, core13
source-max and all16 source-max cuts are fixed before new-pair inference;
the all16 policy uses the three weak conditions for calibration, although they
are excluded from supervised fitting. Joint probe weights are unchanged from
the earlier full probe. Public pretrained models can have unknown exposure
to these collections. The SD-excluded regime changes both supervised generator
exposure and synthetic training mix and is a restricted transfer control.

## Fixed generator and common processing

The base is the existing cached Realistic Vision checkpoint, revision
`1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`, with every component fingerprint
checked against the prior regeneration record. The fixed official
`latent-consistency/lcm-lora-sdv1-5` adapter uses weight one and is fused once.
Its pinned revision is `cf2fced511dbe7e26c8d1d397e728fbab875db4b`; the
134,621,556-byte weight file has SHA256
`8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4`.
The generation-assets lock binds this identity before pair selection and
generation.

The text-to-image pipeline uses `LCMScheduler`, eight steps, guidance one,
512×512 output and batch one. A CPU generator seed is derived from each public
photo ID. The photograph is not supplied as an initialization image. FP16
UNet/text encoder, FP32 VAE and existing torch2 SDPA are used. Model CPU offload
is an allowed memory fallback without changing seeds or generation parameters.
The official adapter card supports SD1.5 derivatives, two to eight steps and
guidance between one and two. See the
[developer model card](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
and [Diffusers memory guide](https://huggingface.co/docs/diffusers/v0.35.1/en/optimization/memory).

Both origins receive RGB conversion, OpenCV INTER_AREA resize to 256 square,
the same OpenCV JPEG95 encode/decode and lossless PNG staging before the three
fixed transforms. The original generated 512-pixel output is retained locally
for provenance. No watermark is embedded. Probe, ordinary fusion and gated
scores may be compared using the unchanged expected-payload decoder.

A 50-step guided DDIM draft was superseded before source selection, generation
or observation of any new detector scores. Its source and fingerprint remain
in `protocol_ddim50_superseded.json`; `protocol_amendment.json` records the
compute-driven change. The LCM configuration is an accelerated SD1.5-family
pipeline, not evidence from a new generator family.

## Reporting and feasibility

Report origin AUC with 2,000 paired bootstrap draws, keeping the real/generated
pair and all copies together. Report fixed-validation-threshold TPR and actual
test FPR separately. Intervals are conditional on the frozen heads and the
selected generator. No test threshold tuning or refitting is allowed.

The local GTX1050Ti has approximately 4GB VRAM. Per-image generation runtimes,
output fingerprints and the fixed seed are recorded in the private generated
source manifest. A stopped run resumes only matching saved outputs and the
remaining fixed seeds; it cannot choose replacement candidates. Resumption
does not refit heads, change thresholds or alter generation settings.
All 200 generations finished on 11 October 2026 without candidate resampling,
using resident CUDA. Recorded image runtimes sum to 3,098.50 seconds (51.64
minutes), with mean 15.49 and range 14.04–21.46 seconds. The completion record's
`total_runtime_seconds` describes the last resumed invocation, not the sum over
both invocations; the per-image manifest preserves the full measured duration.

Prepare without GPU inference:

    python scripts/phase5_paired_content.py --stage protocol
    python scripts/phase5_paired_content.py --stage select

The official COCO annotation archive must be acquired locally first. COCO's
[download page](https://cocodataset.org/#download) and
[official annotation API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)
describe the caption annotation source and image-ID association. The runtime
supplement handles generation, extraction and analysis after selection is locked:

    python scripts/phase5_paired_runtime.py --stage all

The preparation driver does not execute those stages. The runtime records a
hash-bound derivative manifest, one inference record per backbone, predictions,
all 144 metric rows and the final completion record. The public release
contains the safe pair index and aggregate metrics, not private prompts,
images, paths, vectors or per-image predictions.

## Completed measurements

The fixed joint ResNet/CLIP identity AUCs are 0.7292 [0.6857, 0.7731] and
0.8200 [0.7849, 0.8554]. Their image-calibrated TPR/FPR are 0.170/0.045 and
0.310/0.020. The all16 source-max cuts reduce identity TPR to 0.020/0.025,
with zero observed positives among 200 photographs. A zero-event bootstrap
interval is degenerate; under independent common-risk sources, the two-sided
95% binomial upper bound for zero/200 is 1.83%, not zero future risk.

CLIP's JPEG50/low-res32 AUCs are 0.7953/0.6962. Its image-policy FPRs are
0.020/0.325; the latter is 65/200, with pair-bootstrap interval [0.260, 0.385].
The SD-excluded ResNet/CLIP identity AUCs are 0.6542 [0.6089, 0.6992] and
0.8385 [0.8031, 0.8729]. These point differences alone do not establish a
benefit from removing supervised SD exposure: the synthetic fitting mix also
changes, and a paired contrast would be needed for that separate claim.

The strong collection-test ranking transfers poorly to this particular new
pipeline. Content pairing and LCM generation change together; the lower AUC
cannot be attributed to content pairing alone. All images and fixed outcomes
remain included; no thresholds were tuned on this test.

## Postgeneration overlap audit and sensitivity

A separate, post hoc audit finds no native or standardized exact-pixel matches
against 7,592 prior records, the 200 selected photographs, or other generated
images. Three generated images have ten total prior average-hash matches at
Hamming distance at most two: six prior training records and four prior test
records. These coarse signals do not prove identical scenes. All 200 locked
pairs remain in the primary measurements.

A supplementary post hoc diagnostic omits the three entire flagged pairs
without changing models or thresholds. On the 197 remaining pairs, joint
ResNet/CLIP identity AUC is 0.7327 [0.6899, 0.7758] / 0.8227 [0.7898, 0.8567].
The weak transfer remains. The aggregate hash-bound audit and twelve
fixed-policy sensitivity rows are saved in `postgeneration_overlap_audit.json`
and `postgeneration_overlap_sensitivity.json`; no private flag lists or scores
are included in the public evidence.

Caption pairing reduces coarse scene differences, but does not create identical
scenes or pixel alignment. It does not remove unknown pretraining overlap,
generator style, caption selection effects or photographs' earlier encoding
histories. No acceptance probability follows from this control.

## Cached factorial control for process matching

A separate retrospective diagnostic can distinguish transformation matching
from simply adding training slots. Existing unmarked-edit caches cover 2,796 AI
sources, including all 962 held-out Midjourney sources. The non-Midjourney
fitting pool must therefore be restricted to the available 1,834 AI sources;
all retained real-source edit features exist.

Use a two-by-two factorial: real training edits A/B crossed with unmarked AI
training edits A/B. Let A contain JPEG70 and Gaussian blur3, and B contain
Gaussian noise sigma6 and 0.8-scale resize. Each origin gets its clean feature
and two different augmented features, preserving source IDs, labels, row
counts, class weights and optimization settings across all four cells.
Evaluate every cell on the same held-out sources with clean and the union of
all four edits. Keep the five existing source assignments and fit only training
sources. The primary contrast is the mean of A/A and B/B minus the mean of A/B
and B/A, with paired source-bootstrap uncertainty.

The balanced contrast cancels augmentation-set main effects more directly than
the completed clean/JPEG repetition controls. Equal condition counts still do
not imply equal distortion or information. A positive interaction would support
matching for these four processes and this restricted pool; it would not prove
that matching is the sole mechanism of the larger edit-suite gain. The protocol
must be frozen before any factorial fits, and these already examined sources
cannot become a new independent test.
