# Prospective frozen replication on unused source items

## Protocol and status

This replication uses a protocol and validation thresholds frozen before
selecting new source items. It uses 400 sources, 80 each from COCO,
Open Images, DiffusionDB Stable Diffusion, DALL-E 3, and Midjourney. It
replicates within existing collections; it is not an unseen-generator or
new-collection benchmark. Completion is established by saved selection,
derivation, inference, and analysis artifacts, rather than by this plan.

Selection and derivation are complete: exactly 400 distinct sources and
2,560 derived images, with all output hashes checked. The explicit native
ID inventory contains 5,192 exclusions: 2,048 COCO, 1,096 DALL-E 3,
1,048 Open Images, and 1,000 Midjourney items. Independent recomputation
finds zero selected-ID or native-pixel intersections, minimum average-hash
distance three to any of the 7,192 prior records, and zero within-sample
pairs at distance two or less. Saved files are `selection_complete.json`,
`native_exclusion_audit.json`, `selection_overlap_audit.json`, and
`derivation_complete.json`. Both frozen backbones have completed inference
and the final `summary.json` reports detection results. No new fitting or
test-selected threshold changes occurred.

The four existing seed-20260920 probe/fusion heads are used unchanged.
The driver directly checks each head against the hashes in
`outputs/phase4/domain_generalization/frozen_seed_thresholds.json` before
locking the replication protocol. The frozen calibration uses the existing
seed's eligible validation images and both real collections, retaining
whole logit ties at an empirical real-copy FPR no greater than 5%.
The fusion heads were previously fitted to those validation rows: their
thresholds are retrospective calibration of a fixed model, not calibration
on a separate fusion holdout. Neither models nor thresholds use these new
source items.

## Source selection and overlap exclusions

Candidates are ranked by SHA256 of fixed seed, collection, and native
archive entry or source ID. Native images must have a short side of at
least 256 pixels. Download/decoding failures, small images, previously used
IDs, and the fixed duplicate checks can exclude a candidate. Detector
scores never enter selection.

The archive ID exclusions are reconstructed explicitly because earlier
AI source exports were saved as JPEG 95, which changes native pixel hashes.
The original DALL-E ZIP ordering and the ordered `mj-000`, `mj-001`, and
`mj-002` parquet rows are replayed using the original label, embedded-image,
and minimum-size eligibility rules until the 1,000 previously exported
images of each collection are identified. The driver also excludes all
original COCO basenames, every earlier fresh-sample archive ID, and the
1,000 Open Images training-pool IDs plus the earlier fresh photographs.
The new Stable Diffusion archive is part 3; the original source exports
came from parts 1 and 2.

The explicit ID inventory, hashes of its exclusion inputs, and replay
counts are saved before selecting candidates. Supplemental COCO and new
Stable Diffusion archive fingerprints are recorded after selection and
before detector inference in `archive_inventory.json`. All candidates undergo
exact native-pixel and 8×8 average-hash Hamming-distance-at-most-two
screening against the original 6,000 sources, earlier 192 fresh sources,
1,000 additional Open Images sources, and preceding accepted sources in
this replication. This detects specified forms of duplicate overlap; it
does not prove semantic independence or absence from backbone pretraining.

## Common processing and image counts

All sources use PIL RGB conversion, metadata-specified positive
counterclockwise Open Images rotation, OpenCV area resize to 256×256,
and the same JPEG 95 encode/decode. PNG files retain those decoded pixels.
Open Images rotation is constrained to 0, 90, 180, or 270 degrees.

Each real source contributes four unmarked images: identity, JPEG 50,
side crop, and 32×32 down/up sampling. Each AI source contributes both an
unmarked and a classical marked cover independently under those four
conditions. DWT-DCT uses strength 12 and the original public 16-bit payload,
applied after the common JPEG 95 round trip. Thus 160 real sources yield
640 images and 240 AI sources yield 1,920 images: 2,560 images total.

Separate marked and unmarked comparisons reuse the same real controls.
Each comparison therefore includes 1,600 images overall, or 400 sources
and images for one condition. The two comparison totals must not be added
as independent observations. Gate verification remains fixed at 13/16 bits.

## Frozen scoring and inference safeguards

Existing backbone implementations and preprocessing are reused. Before new
feature extraction, clean-source anchors are compared with saved original
embeddings using cosine similarity and relative norm tolerance. Every new
image hash and manifest ID order is checked. The planned evaluation has
one feature pass per backbone and no selection based on test scores. Both
comparability checks passed: ResNet's anchor relative L2 differences are
zero; CLIP's largest is 1.92×10^-6, with minimum cosine above 0.999999999998.

Probe probability feeds the existing two-feature fusion head together
with bit agreement. Unsaturated float64 logits are used for discrimination
and frozen decisions. The gate returns the fusion logit when agreement
clears 13/16 and the unchanged probe logit otherwise. All four seed heads
store float64 coefficients, so this arithmetic matches frozen calibration.

Implementation-only corrections after the initial script snapshot are
recorded in `implementation_repairs.json` before detector inference. They
normalize file-hash paths, initialize analysis result containers, validate
manifest hashes, and compute the same bootstrap source multiplicities with
weighted tied-rank AUC for speed. They change neither selected sources,
processing rules, model weights, thresholds, nor the analysis estimands.

## Analysis and uncertainty

Each condition and marking view reports AUC and fixed-validation-threshold
TPR/FPR, with collection-specific decision counts. The 2,000 bootstrap
draws use seed 20261011, stratify by all five collections, and resample
source units with all their derivatives retained. Common draws support
paired gate-minus-probe, fusion-minus-probe, and gate-minus-fusion AUC
intervals and fixed-threshold rate differences. Bootstrap arrays and their
manifest/calibration hashes are saved for audit.

Single-condition collection decisions have one image per source and use
exact two-sided binomial intervals. Overall rates include repeated images
and therefore use source-cluster bootstrap intervals. These intervals are
conditional on the existing frozen heads. They do not account for new
training runs, broader collection selection, or semantic/pretraining overlap.
Zero-event bootstrap intervals can be degenerate; exact single-condition
source intervals must accompany any interpretation of an observed zero.

## Measured results

The numbers below come from `summary.json`. All decisions use the thresholds
saved before new source selection. Each marking view contains the same 160
photographic sources and 240 AI sources. Reported intervals are conditional
source-bootstrap 95% intervals, unless identified as exact.

### Identity, unmarked AI

- ResNet probe AUC is 0.965313, TPR 0.808333, and FPR 0.031250.
  Photographic false positives are 2/80 COCO and 3/80 Open Images.
- CLIP probe AUC is 0.994453, TPR 0.983333, and FPR 0.081250.
  False positives are 5/80 COCO and 8/80 Open Images. Thus even this clean
  subset does not establish that the nominal validation 5% rate transfers.

### All four conditions, unmarked AI

- ResNet probe/fusion/gate AUCs are 0.956730 / 0.911994 / 0.956110.
  Probe TPR is 0.809375 [0.765599, 0.853125]; gate TPR is 0.808333.
  Probe and gate FPR are 0.051563 [0.029688, 0.076563].
  Gate-minus-probe AUC is -0.000620 [-0.001240, -0.000127], while
  unconditional fusion-minus-probe is -0.044736 [-0.056714, -0.033793].
- CLIP probe/fusion/gate AUCs are 0.988464 / 0.953639 / 0.988241.
  Probe and gate TPR are 0.986458, with FPR 0.142188
  [0.106250, 0.179688]. Gate-minus-probe AUC is -0.000223
  [-0.000443, -0.000076], and fusion-minus-probe is -0.034824
  [-0.048018, -0.022739].

The gate strongly reduces the damage caused by unconditional fusion on
unmarked AI, but the saved paired intervals do not support an exact
no-loss claim. Accidental payload matches can activate it on unmarked
images, and separate validation thresholds can yield slightly different
decisions even when the underlying fallback score is unchanged.

### All four conditions, marked AI

- ResNet probe/fusion/gate AUCs are 0.959082 / 0.949504 / 0.973441.
  Gate-minus-probe gain is 0.014359 [0.009007, 0.020568].
  Probe and gate TPRs are 0.819792 / 0.890625 at the same observed
  FPR 0.051563. The paired TPR gain is 0.070833 [0.051042, 0.091667].
- CLIP probe/fusion/gate AUCs are 0.988193 / 0.975884 / 0.990745.
  The gate gain is 0.002552 [-0.000817, 0.006500], so its interval
  includes zero. Probe and gate TPRs are 0.986458 / 0.994792 at
  FPR 0.142188.

Condition-wise marked ResNet gate gains are 0.023776 for identity and
0.026641 for JPEG 50, but -0.000339 for crop and -0.000729 for low
resolution. The corresponding gate AUCs are 0.992266, 0.996693,
0.967552, and 0.923802. Thus improvement depends on preserved payload
evidence rather than marking alone. The pooled four-condition gain must
not be described as a positive gain under every operation.

### Calibration under low resolution

The strongest remaining fixed-threshold problem is the 32×32 round trip.
ResNet probe/gate FPR becomes 0.100000; CLIP probe/gate FPR becomes
0.337500, split as 26/80 COCO and 28/80 Open Images. CLIP's low-resolution
unmarked AUC still equals 0.972031. High ranking accuracy therefore does
not ensure calibration transfer to this operation. No thresholds were
adjusted after observing these errors. This is a measured limitation, not
a claim that the pipeline maintains a 5% test FPR.

## Provenance and reconstruction

The driver is `scripts/phase4_fresh_replication.py`. The stages `select`,
`derive`, `extract`, and `analyze` write separate markers under
`outputs/phase4/fresh_replication/`. Selection may need network access to
the official Open Images test S3 files. Derivation needs neither detector
inference nor a GPU. Extraction reuses pinned cached backbones; analysis
performs no refits or threshold search.

Chronology is saved in UTC on 2026-10-10: validation thresholds were locked
at 04:29:45, replication protocol at 04:38:55, source selection completed
at 04:43:06, derivation at 04:44:47, ResNet test inference began at
04:47:59 and CLIP at 04:48:26. Final analysis completed at 04:50:49.
The inference JSON files record both starts, image/feature hashes, and
anchor checks; `summary.json` records unchanged model hashes and the
frozen calibration hash. Source IDs are unique and source-level files stay
outside the anonymous aggregate release.

`bootstrap_draws.npz` preserves common source multiplicities and all score/
decision draws. `summary.json` contains 60 system measurements and 60 paired
comparisons. Regenerating descriptive analysis from saved features is
different from performing another model pass or retuning: the new sample
was selected without detector outcomes and evaluated with fixed existing
heads. This replication cannot exclude pretraining overlap, semantic
duplicates missed by the stated screen, or collection-specific artifacts.
