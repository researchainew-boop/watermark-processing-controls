# Phase 6: paired fusion with explicit unmarked-performance constraints

This plan is fixed before any detector or watermark scores are computed on the
new calibration or test sources. Development uses previously examined data;
its results are not an independent confirmation. Novelty is an assessment to
be made after comparison with primary literature and successful evaluation.

## Primary hypothesis

The single confirmatory system is the frozen original ResNet probe with
TrustMark evidence. At an independently calibrated nominal 5% photographic
source false-positive target over identity, JPEG50, side crop, and lowres32,
the proposed method improves source-balanced marked-image TPR over both the
probe and the strongest origin-fusion baseline selected using development
data. Unmarked AI TPR loss must be at most one percentage point on the fixed,
equal-weight BigGAN/ADM/VQDM test mixture. Both threshold recalibration and
maximization over the three registered expected payloads are included.

Success requires a paired 95% source-bootstrap interval above zero for each
of the two gain comparisons, a one-sided 95% lower bound for unmarked TPR
change above -0.01, and realized photographic source FPR at most 0.05 for
the proposed method. Actual source/image FPR is disclosed for all methods.
This stricter success criterion is explicitly sealed before fresh inference;
the implementation already requires it.
The two gain comparisons are an intersection requirement, not selection of
the favorable comparison. The primary claim is unsuccessful if either gain
comparison fails. An empirical or bootstrap noninferiority result is not
automatically a finite-sample population certificate. In particular, ordinary
zero-event bootstrap intervals are supplemented by harmful-switch bounds.

At 200 AI sources per family, a family-specific 1% harmful-switch rate can
remain unresolved even with no observed events. Report those intervals;
never describe a pooled result as a guarantee for every generator or future
generator. Calibration requires independent/exchangeable sources within the
declared collection and processing family. No guarantee under arbitrary
distribution shifts is asserted.

## Fixed inputs and separation

- Development: 68 SD and 68 DALL-E-labeled sources plus 400 COCO and 400
  Open Images photographs from existing Phase 5 diagnostic sources.
- Fit/tune/constraint roles: collection-stratified deterministic hash order,
  50/25/25 percent, with every derivative inheriting the source role.
- Fitting and tuning use only identity/JPEG50. Constraint diagnostics use all
  four views at unchanged tune thresholds and never select another candidate.
- Independent final calibration: 400 new COCO and 400 new Open Images sources.
- Final test: 200 new BigGAN, 200 ADM, 200 VQDM, 200 COCO and 200 Open Images.
  These AI families are outside supervised detector/fusion development.
  Pretrained ImageNet/CLIP exposure cannot be ruled out.
- All final sources undergo the same area resize to128, linear restoration
  to256, JPEG95, then lossless staging before watermark embedding. A native
  to256 JPEG95 version is retained solely as a diagnostic contrast.
- GenImage and photographic collections are not caption paired. The existing
  caption-paired results remain a separate, previously observed control.

## Prediction and comparisons

Prediction receives only the passive score and the watermark decoder outputs
from the received image and its fixed JPEG90 view. No clean image, sibling
edit, source ID, collection, attack name, embedded ground-truth payload,
origin label, or role is a prediction input. Query the same three registered
payloads on every marked, unmarked, and real received image; maximize scores
and calibrate the corresponding source maximum.

Compare the frozen probe, static logistic fusion, hard13/16 gate, simple
confidence weighting, refitted established reliability mixture, capped
positive evidence, full-feature unconstrained logistic fusion, and watermark
scores. Ablations remove null penalty, paired objective, received-view
stability, or complete-decoding features. All share sources and calibration.
UniversalFakeDetect additionally uses its published native CenterCrop224
preprocessing and the common identity image, unchanged score polarity.

## Secondary transfer, attacks, and prevalence

CLIP, DWT/DCT, generator-specific results, ablations and marked fractions from
0% to100% are secondary. Marked fractions use declared counterfactual weighted
mixtures of matched marked/unmarked states, not newly collected prevalence.
Three payloads and three source-coupled noise seeds remain paired in analysis.

Published WAVES ResNet18 embedding PGD, learned-compression reconstruction,
and SD1.5 DDIM reconstruction run on a fixed20-source-per-test-collection
subset. A50-source-per-photographic-calibration-collection subset supplies
attack-specific calibration diagnostics. Raw attacks and35dB residual-budget
adaptations are reported separately. The adaptations do not reproduce the
complete native attack. Shared pixel PSNR is not shared semantic/perceptual
quality. Learned compression and SD are two distinct reconstruction pipelines;
they are not two independent diffusion-model families.

Reconstructed photographic inputs are labeled by input-source provenance and
reported separately: diffusion reconstruction introduces generated pixels.
Neither complete message matches nor AI scores authenticate an issuer.

## Result handling and release

Freeze source IDs, feature code, method code, model parameters, registered
queries, analysis choices, and primary endpoints before final inference.
Calibrate on independent negatives, freeze cuts, then run final test once.
Implementation repairs must be documented with hashes; they may not change
scientific choices in response to test scores. Keep every unsuccessful method,
incomplete certificate, negative transfer, and attack failure in the evidence.
Release code/protocol/aggregate evidence without redistributing restricted
source images, private paths, feature vectors, weights, or credentials.
