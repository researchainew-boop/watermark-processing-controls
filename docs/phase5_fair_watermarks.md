# Matched-distortion watermark comparison with separate calibration

## Purpose and interpretation

This experiment compares expected-message evidence at a shared nominal
validation false-match target. It uses the same three existing balanced
16-bit information payloads for the classical DWT–DCT mark and official
TrustMark 0.9.2 Q with BCH5. Raw decoder margins supply a new empirical
score; they are not calibrated probabilities or issuer authentication.
Original method-specific verification remains a separate secondary result.

The protocol is saved before source selection and decoding. Selection has
completed with 1,004 distinct source identities: 400 calibration photographs,
400 disjoint test photographs, and the existing 204 AI sources (68 each from
DiffusionDB, DALL-E 3, and Midjourney). Each photographic arm contains 200
COCO and 200 Open Images sources. The test retains 200 of the earlier 204
COCO watermark controls, chosen by fixed source-ID hash order. Calibration
COCO items come from the original validation split; Open Images items come
from the already screened 1,000-source pool with a fixed hash-order split.
All selected-source native-pixel and average-hash duplicate checks pass.

These are reused collections and collected source items. Source-disjoint
watermark calibration does not turn them into an untouched origin-detection
benchmark or exclude semantic overlap, uncertain generator labels, or
backbone pretraining overlap. The existing 204 AI items are preserved
independently of their previous recovery results.

## Processing and distortion matching

PIL converts stored sources to RGB. Open Images metadata rotation is positive
counterclockwise, restricted to 0, 90, 180, or 270 degrees. OpenCV area resize
produces 256×256 pixels; every origin then receives identical JPEG-quality-95
encoding and decoding, retained as PNG. Earlier synthetic exports already
received JPEG95 during collection. This common round trip does not erase
their previous encoding history or fully isolate collection differences.

The base encoder uses DWT strength 12 or official TrustMark Q strength 1.
For each marked cover, the complete uint8 RGB residual relative to its
unmarked cover is multiplied by one scalar. A pre-scoring amendment uses
deterministic stochastic rounding and clips pixels to [0,255]. A fixed gain bracket followed by 48 binary
search steps chooses the evaluated pixels closest to PSNR 40 dB; preparation
stops if error exceeds 0.1 dB. This search uses only image distortion, never
decoder output, expected-message scores, or recovery success.

The original nearest-integer draft is preserved: it stopped preparation
before decoding when discrete rounding produced a closest PSNR of
39.87325 dB. Root approved stochastic quantization before any new scoring.
A source/payload-derived seed, excluding method, generates one fixed PCG64
float64 uniform array of shape 256×256×3. Both methods reuse this array,
unchanged across all gain-search evaluations. Output is
floor(uint8-cover + float64-gain × float64-residual + uniform), clipped to
[0,255] and converted to uint8; channel ordering is BGR. No alternative
random seeds or quantization configurations are searched for recovery.

Residual rescaling and stochastic quantization are outside both original
encoding configurations.
The comparison therefore evaluates the stated matched-distortion adaptation,
not the untouched default package. The 16 information bits are shared;
embedding redundancy, ECC, architecture, and robustness training differ.
Embedding SSIM and both methods' original encoder PSNR are also recorded.

There are 2,228 covers: 800 photographic unmarked covers plus 204 AI sources
with one unmarked cover and six marked covers (two methods × three payloads).
Identity, JPEG50, removal of twelve pixels from each edge followed by bilinear
resizing to 256×256, and 32×32 down/up sampling produce
8,912 unique images. Calibration has 1,600 photographic images; test has
7,312 images. Repeated conditions and payloads are dependent source copies.

## Fixed score, query rules, and threshold order

For expected information bits b and 16 signed coefficient margins s, the
classical score is the mean of (2b−1)s/12. TrustMark supplies the mean of
(2b−1) times its first 16 raw decoder logits before hard decisions or ECC.
The official Q processing function, PIL bilinear 245×245 resize, and tensor
range [−1,1] are reused; its decoder also applies its own internal resizing.
All 100 learned logits and 16 DWT margins are archived. Decoder output is
not assumed uniform or independent.

The default query uses `1011010010110100`. The any-of-three query uses the
maximum score for this payload, `0100101101001011`, and `1100101000110101`.
Each method/query rule is calibrated separately on the same 1,600 real
calibration copies. The lowest threshold whose empirical FPR is at most
5% is selected, with acceptance at score ≥ threshold and whole ties kept.
This targets a pooled calibration mixture, not 5% for every individual
condition or collection. A locked threshold file precedes all decoding of
the newly prepared test images; archived previously decoded covers are
used only for implementation/API comparability anchors.

The default-query positive set contains only default-payload marked AI
images decoded by their matching method. The any-of-three positive set
contains all three marked payload covers and averages derivatives within
source. Unmarked photographs and unmarked AI supply separate false-mark
controls. Both scores are computed on each image, but cross-method marked
inputs do not enter the primary false-match population.

Native secondary decisions are DWT ≥13/16 agreement or TrustMark's valid
BCH5/schema1/full expected 61-bit message. These different acceptance rules
are retained as native recovery outcomes and are not reinterpreted as the
same verification test.

## Statistical reporting

Reports distinguish per-image rate, mean per source, source-ever acceptance,
and acceptance of all copies from one source. Collection-stratified source
bootstrap uses 2,000 draws, retaining all conditions/payloads together. Shared
draws support paired TrustMark-minus-DWT differences. Confidence intervals
condition on the fixed encoders and learned calibration thresholds; they
do not include threshold-estimation uncertainty or broader dataset sampling.
Two-sided Clopper–Pearson binomial intervals accompany single source
events, including zeros. Their exact binomial coverage assumes independent
sources sharing the same event probability. Within one sampled collection,
these provide a collection-specific interval under that sampling model.
The pooled COCO/Open Images sample has fixed collection quotas, so different
collection risks violate the common-probability assumption: pooled intervals
saved under the `source_ever_exact_ci95` or `source_all_exact_ci95` field names
are model-based bounds, not exact coverage guarantees for this stratified
mixture. Pooled reporting should use the collection-stratified source
bootstrap; the manuscript uses binomial zero-event bounds only for the
individual collections. For example, zero events among 200 sources from one
collection give an approximately 1.83% upper two-sided 95% bound. Repeated
copies do not enlarge that source count.

The primary comparison shares a nominal calibration target, not guaranteed
equal test FPR. Observed test rates and collection-specific errors must be
reported before interpreting recovery differences. PSNR/SSIM compare edited
outputs both to the original unmarked cover and to the marked attack input.

## Execution and saved evidence

Driver: `scripts/phase5_fair_watermarks.py`. Owned outputs are
`outputs/phase5/fair_watermarks/` and generated images under
`data/phase5/fair_watermarks/`.

Stages are `select`, `prepare-cpu`, `derive-cpu`, `prepare-learned`, `derive`,
`decode-calibration`, `calibrate`, `decode-test`, and `analyze`.
The CPU preparation embeds DWT and measures distortion without decoding.
TrustMark encoding and both decode stages require the scheduled model run.
Source/input/protocol/manifest hashes and model fingerprints guard stage
boundaries. Test inference records the already locked threshold-file hash.
Learned encoding records its start time and the already frozen amendment hash;
completed encoding cannot be repeated through this driver. Elapsed timings
are recorded without changing the sample or decoding conditions.

The CPU stages have completed: 1,004 unmarked covers and 612 DWT covers
produce 6,464 images across the four conditions. Their source, base, cover,
amendment, and derivative manifest hashes agree with the completion locks.
The DWT matched embedding PSNR spans 39.999869–40.000141 dB. This confirms
the distortion preparation, not watermark recovery; no new decoder output
was produced by those stages.

The remaining stages run in this order:

1. `prepare-learned` creates 612 TrustMark covers using the locked sources,
   payloads, distortion rule, and shared stochastic-rounding uniforms.
2. `derive` preserves and verifies the 6,464 existing images, then adds
   2,448 learned-watermark images for a total of 8,912.
3. `decode-calibration` decodes each of the 1,600 calibration images once
   with each raw decoder. Three expected-payload scores are computed on
   CPU from those saved outputs.
4. `calibrate` locks the four method/query thresholds.
5. `decode-test` reads the threshold lock before decoding each of the
   7,312 test images once with each raw decoder.
6. `analyze` computes the prespecified source-level summaries and paired
   confidence intervals from the saved scores.

Raw learned decoding uses batches of 16 images. Query payloads do not cause
additional decoder passes. Archived API anchors are recorded separately
from the calibration and test populations.

The initial driver snapshot is preserved. A preselection implementation
correction recognizes the existing split's `val` label as validation; it
does not change the intended population, selection order, or analysis.
`protocol_amendment.json` and `rounding_seeds.csv` preserve the quantization
amendment, timing, authorization, formulas, and every fixed seed.
## Completed measurements (11 October 2026)

All stages completed on the fixed 1,004 sources. There are 2,228 covers,
8,912 images, 9,600 calibration query-score rows, and 43,872 test query-score
rows. Each raw decoder processed each unique calibration or test image once;
the three expected-payload queries were computed from those saved outputs.
The summary contains 440 rate records, 30 paired method differences, and
eight image-quality aggregates.

Every method/query threshold accepts 80 of the 1,600 calibration images,
exactly the prespecified pooled 5% calibration rate. Thresholds were locked
at 05:33:42 UTC, after calibration inference finished at 05:33:29 UTC and
before test inference began at 05:34:02 UTC. The test inference record's
threshold hash agrees with the final threshold file. Encoding, calibration,
and test decoding used the same official checkpoint fingerprints, and all
recorded archived-bit/public-API anchors passed.

For the primary default-payload signed-margin rule, marked-image acceptance
is 200/204 versus 204/204 for DWT and TrustMark on identity images,
194/204 versus 203/204 after JPEG50, 12/204 versus 204/204 after cropping,
and 0/204 versus 136/204 after the 32×32 round trip. Across all four
conditions, rates are 0.497549 and 0.915441. The paired TrustMark-minus-DWT
difference is 0.417892, with collection-stratified source-bootstrap 95%
interval [0.397059, 0.438725], conditional on the fixed calibration cuts.

Actual default-query photographic false-match rates are 75/1,600
(0.046875) for DWT and 86/1,600 (0.053750) for TrustMark. Their source-
bootstrap 95% intervals are [0.033750, 0.060625] and
[0.042500, 0.065000]. By collection, DWT rates are 0.038750 on COCO and
0.055000 on Open Images; TrustMark rates are 0.048750 and 0.058750.
Observed test false-match rates therefore differ even though the nominal
calibration target is shared. Unmarked-AI false-match rates across the same
conditions are 0.050245 and 0.041667.

The any-of-three calibrated query gives marked-image source-mean acceptance
0.474673 for DWT and 0.905637 for TrustMark, while photographic per-image
false-match rates are 0.053750 and 0.035000. Its positive population includes
all three embedded payloads, whereas the default query includes only the
default payload; these are separate prespecified comparisons.

Signed-margin acceptance is not full-message recovery. On 32×32 images,
TrustMark's full 61-bit message with valid BCH5/schema1 is recovered in only
5/204 cases (0.024510), despite the 136/204 accepted partial 16-bit-margin
scores. The classical native ≥13/16 rule accepts 2/204 (0.009804). Native
TrustMark recovery is 202/204 on identity, 185/204 after JPEG50, and 203/204
after cropping. Its native any-of-three verification produces no false
matches across the four copies of each of 200 COCO and 200 Open Images
test photographs. The zero source-ever count in each individual collection
has a two-sided 95% Clopper–Pearson upper bound of 0.018275 under that
collection's independent common-risk source model. This does not establish
zero population risk or exact pooled coverage across different collections.

Embedding PSNR is 40 dB for both methods within the fixed tolerance: DWT
range [39.999869, 40.000141], TrustMark range [39.999934, 40.000090]. Mean
SSIM is 0.982442 and 0.978286, respectively, so the matched PSNR does not
make all perceptual distortion characteristics identical. Results apply to
the explicitly rescaled, stochastically quantized encoders and these four
processing conditions, not their unmodified defaults or regeneration.

Measured encoder elapsed time was 306.88 seconds; raw calibration and test
decoding took 24.20 and 108.74 seconds. Encoding used official per-image
calls and raw decoding used batch 16 on the local CUDA GPU. No memory
fallback, source exclusion, decoding replay, or result-driven parameter
change occurred. One CPU derivation launch stopped at import time because
a stripped dependency path omitted scikit-image; enabling the existing
scientific-image libraries repaired the environment before any processing
began. The fixed driver and protocol were unchanged.

Authoritative aggregate evidence is `summary.json`, `rates.csv`, and
`paired_method_differences.csv`; `thresholds.json` and the calibration/test
completion locks document stage order and hashes. Individual raw arrays,
query scores, sources, and image manifests remain local for independent
audit. The summary SHA256 is
`0ff6453348b0cddf073f70757e15633e6963de41f320c71207e7579b889af338`.
