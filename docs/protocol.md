# Conference Evaluation Protocol

This is an implemented evaluation and retrospective duplicate correction
for the conference submission, not a registered protocol. The original results were viewed
before filtering and refitting. Describe the corrected run as retrospective
and exploratory.

## Questions and operational labels

The primary comparisons ask how transforms affect fixed-payload recovery,
whether validation-fitted fusion helps the origin probe, and how removing
matched real edits from training changes the probe. A secondary attack-label
analysis is exploratory.

- R-C: clean real photograph without the expected mark.
- R-T: real photograph after a transform also applied to watermarked AI.
- R-S: real photograph embedded with a different payload, as a study spoof.
- S-U: unmarked AI image.
- S-W: AI image embedded with the expected payload.
- S-A: S-W after a tested transform, whether or not the payload survives.

S-A is a constructed experimental label, not verified removal or evidence
of humanification. The secondary head cannot establish editing intent.

## Watermark and threat scope

The DWT–DCT baseline orders mid-band tile coefficients, using strength 12
and fixed expected bits 1011010010110100. Agreement with those bits is
measured, with classical verification at agreement of at least 13 of 16
bits. This corrected-core experiment uses one payload. The separate
October 9 benchmark compares three balanced payloads with DWT–DCT and
official TrustMark Q. TrustMark BCH5 carries the same 16 payload bits plus
45 fixed zero bits; verification requires an exact 61-bit expected-message
match, valid ECC, and the expected schema. Raw bit agreement and ECC validity
alone are not TrustMark verification. Neither experiment demonstrates
secret-key authentication or resistance to an attacker who knows the bits.

The listed JPEG, blur, noise, resize, crop, color, denoise, inpaint,
low-resolution, grid-shift/JPEG/noise, and spectral operations have matched
real controls. The spectral wipe targets the embedding band and is band-aware.
The legacy regenerate operation is a 7-pixel roll, JPEG quality 19, and
Gaussian noise with standard deviation 16. It is not diffusion reconstruction.
The core does not test diffusion removal. The October 9 extension completed
288 outputs from 48 sources: actual DDIM reconstruction at strengths
0.25/0.50 and a separate posterior-mean VAE control. This is one checkpoint
and fixed settings, not an exact Zhao or UnMarker reproduction; see
extension_run.md. Reconstructed photographic inputs contain generated pixels,
so their labels denote input-source provenance rather than intact-photo truth.
Adaptive-query attacks and camera recapture remain untested.

## Original selection and splits

The seed-20260920 source split is made before derivation. Midjourney is forced
into test. Original counts are:

- Training: 1,400 COCO, 1,390 DiffusionDB, 710 DALL·E 3; total 3,500.
- Validation: 300 COCO, 301 DiffusionDB, 149 DALL·E 3; total 750.
- Test: 300 COCO, 309 DiffusionDB, 141 DALL·E 3, 1,000 Midjourney; total 1,750.

Derivatives share their source ID's split, but different IDs can represent
the same image. The original source split did not prevent content overlap.

## Retrospective content screen

1. Link equal decoded color pixels at native or standardized 256×256 size.
2. Link all-source 8×8 average hashes at Hamming distance at most two.
3. Form connected components over those links across collections and splits.
4. Keep one representative per component by original training, validation,
   test, then source-ID priority.
5. Exclude all derivatives of other members; retain original split assignments.

The audit records 403 candidate links, including five exact-pixel pairs,
189 original cross-split links, and 107 cross-generator links. Those counts
overlap. Perceptual links are candidates, not proven visual duplicates;
collisions and transitive links can exclude distinct images. Train-first
retention can change collection and split composition.

The screen retains 5,823 sources and excludes 177:

- Training: 1,345 COCO, 1,370 DiffusionDB, 700 DALL·E 3; total 3,415.
- Validation: 291 COCO, 292 DiffusionDB, 145 DALL·E 3; total 728.
- Test: 284 COCO, 300 DiffusionDB, 134 DALL·E 3, 962 Midjourney; total 1,680.

The corrected test contains 26,880 images. No exact/average-hash candidate
link remains across retained splits under this rule. Other overlap,
ordered-selection bias, and source/content/encoding confounds remain possible.

## Fits and evaluation

Keep original frozen ResNet and CLIP embedding arrays unchanged. Subset by
retained IDs and refit class-balanced logistic heads with scikit-learn.
Training rows fit heads; eligible validation rows fit fusion and thresholds.
Midjourney stays test only. The 32×32 round trip, grid shift/JPEG/noise, and
spectral wipe are excluded from training and validation fusion fitting.

Distinguish validation-fixed decisions from descriptive TPR at a point on
the test ROC. Test-ROC TPR at 5% FPR is not a deployment threshold fixed in
advance. Bootstrap sources, preserving all derivatives and their labels in
each draw; compare scores using paired source-clustered draws. This also
applies to secondary heads whose labels vary within a source.

An additional abstention baseline requires reporting coverage with errors.
It is not validated real-world authentication. Missing payload evidence or
a low origin score cannot prove authenticity.

