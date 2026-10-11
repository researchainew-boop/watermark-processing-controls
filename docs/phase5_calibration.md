# Source-disjoint fusion and conservative photograph calibration

This is a **new retrospective diagnostic**, using the earlier cached feature
pool and the already evaluated 400-source replication. It does not turn any
previously examined sample into an untouched test, and it does not establish a
conference acceptance probability.

## What changes

For each of the five locked source splits, the original full probe is preserved.
Within each validation collection, a fixed SHA256 order divides sources into
two halves. The first half fits a new fusion head and the second half calibrates
its decision thresholds. Each half contains 439 sources: 219 photographs
(144 COCO and 75 Open Images) and 220 AI sources. Each fusion uses the original
13 fitting conditions, giving 5,707 fitting copies. Midjourney remains excluded
from training, fusion fitting, and calibration. Every derivative follows its
source, and the two validation halves are disjoint within each split.

The new fusion receives the unchanged probe probability and recovered payload
agreement. The gate still uses the fusion only at agreement of at least 13 of
16 bits; otherwise it uses the probe. The original heads and earlier results
are never overwritten. The new source role table, protocol, cache hashes and
earlier model hashes are saved before the ten CPU fusion fits.

## Three prespecified decision policies

1. **Independent per-image core13 calibration:** select the smallest observed
   negative calibration logit whose empirical copy false-positive rate is at
   most 5%, accepting scores greater than or equal to that cut. Ties stay
   together. This preserves the familiar per-image target for comparison;
   it does not provide finite-source control of a population error rate.
2. **Source maximum, core13:** for each real calibration source, take the maximum
   score over the 13 fitting-eligible benchmark views. Compute a separate
   finite-source conformal cut in COCO and Open Images. Use the maximum of those
   two cuts globally, accepting a score only when it strictly exceeds the cut.
3. **Source maximum, all16:** apply the same construction over all 16 benchmark
   views. The three weak attacks remain excluded from model fitting but are
   deliberately included in this new retrospective negative calibration.
   They are held out from fitting only in this policy.

The core13 family includes two identity views. For photographs, the second is
a wrong-payload watermark spoof, so it is a controlled benchmark bundle, not
thirteen naturally sampled photographs. The remaining eleven views are the
fitting-eligible edits. All16 adds the 32-pixel round trip, grid shift with JPEG
and noise, and the mid-band spectral wipe. The grid shift is not diffusion
regeneration.

## Exact finite-source rule and its limits

For a collection with `n` calibration sources, let `M_i` be the maximum score in
source `i`'s complete processing bundle. At alpha = 0.05, set
`k = ceil((n + 1) * (1 - alpha))` and use the `k`th smallest `M_i`. If `k > n`,
the cut is positive infinity. Accept only scores **strictly above** the cut;
this preserves entire ties and makes their treatment conservative.

Here COCO has `n = 144`, `k = 138` and finite-source marginal bound
`7 / 145 = 0.04828`. Open Images has `n = 75`, `k = 73` and bound
`3 / 76 = 0.03947`. Taking the maximum of both collection cuts needs no
collection or attack label when a new image is scored. It remains conservative
for either prespecified real collection under the conditions below.

For a fixed fitted scoring function, calibration and future real **source
bundles must be exchangeable within the collection and processing family**.
Under that condition, the rule bounds the marginal probability of at least
one false positive in a future bundle by 5%. This is a marginal probability over
calibration and future-source sampling. It is not a high-confidence bound on
the conditional population FPR of one saved threshold. It does not guarantee
performance for a different collection, changed encoding, uncalibrated edits,
arbitrary removal attacks, or adversarially selected photographs. Choosing a
policy after observing its test results would also need separate justification.

The finite-source quantile and marginal interpretation follow the split
conformal construction in Angelopoulos and Bates, *A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification*
(December 2022 version), Sections 1, 4.4 and Appendix D:
[primary paper](https://arxiv.org/pdf/2107.07511). Applying that construction to
the maximum over a source's prespecified views, then taking the maximum of the
two collection cuts, is the conservative adaptation used in this diagnostic.

## Evaluation and uncertainty

Thresholds are saved before Phase5 test scoring. Neither original test scores
nor the new400 scores select a head, calibration family, hyperparameter or
threshold. The original tests retain full source bundles and unmarked AI edits.
The 400-source sample retains its four matched processing conditions and
uniform JPEG95 preparation, which differs from the calibration acquisition
protocol. Its results are a measured transfer follow-up, with no exchangeability
guarantee asserted.

The report includes copy TPR, copy FPR and the proportion of real sources with
at least one positive view. It also separates the two real collections, seen
AI generators, held-out Midjourney, and marked/unmarked AI. Five-split means
and sample standard deviations are reported. The source bootstrap uses the
same source weights across overlapping tests and all policies; intervals for
policy differences are paired. These intervals condition on fitted heads and
saved thresholds and do not include training or calibration uncertainty.

Some original COCO validation sources lacked CLIP vectors for the three weak
views. Their existing PNGs and ResNet vectors are reused. A manifest of exactly
792 missing CLIP rows from 264 real calibration sources was locked after the
protocol and before the separately scheduled extraction. Sources are not
dropped on feature coverage. Image pixels, row order and manifest hash protect
that extraction.

## Artifacts and execution

The script is `scripts/phase5_calibration.py`, and all new artifacts stay in
`outputs/phase5/calibration/`. Stages are `prepare`, `fit`, `calibrate` and
`evaluate`. Missing CLIP extraction is separately scheduled by the root agent.
`protocol.json` and `source_roles.csv` establish the design; `heads_locked.json`
records the ten fitted fusion heads; `thresholds_locked.json` records the 90
decision cuts; aggregate metrics and paired differences follow only after those
locks. The source role table and missing-image manifest contain local source
metadata and are not aggregate public evidence.

## Additional SD-family exclusion control

A separate prespecified protocol under `sd_family_excluded/` uses the first
locked split. It fits new ResNet and CLIP probes with every Stable Diffusion
and Midjourney source excluded from supervised training. The source-disjoint
fusion-fit half also excludes those families, and calibration excludes their
AI sources. The real calibration half and all three threshold policies remain
the same. DALL·E 3 is the only synthetic training collection: each probe uses
2,729 training sources /35,477 copies, and each fusion uses 292 source-disjoint
validation sources /3,796 copies. Heads and thresholds are locked before the
parent agent scores the planned new caption-paired SD images.

This changes the synthetic training mix as well as supervised generator
exposure, so it cannot isolate a causal effect of generator holdout. CLIP and
ImageNet pretraining may already include related imagery. The control excludes
SD-family supervised feature-head fitting; it does not make the pretrained
representation ignorant of that family. Stages `loo-fit` and `loo-calibrate`
produce four additional heads and eighteen threshold policies without changing
the main five-split calibration diagnostic.

## Measured findings

All ten independent fusion fits, ninety main threshold policies and four
SD-exclusion fits with eighteen additional policies completed. The earlier
heads and locked inputs remained unchanged. The extra CLIP extraction passed
all four comparability anchors. Test scoring began after the corresponding
threshold locks.

Across the five original full-probe tests, the independent per-image policy
produces core13 copy FPRs of 4.95% for ResNet and 5.15% for CLIP. Adding the three
uncalibrated weak views raises them to 5.61% and 9.69%. On the already evaluated
JPEG95 new400 follow-up, they are 4.06% and 15.53%. Separating fusion fitting
from calibration therefore removes the validation reuse, while the measured
processing/encoding transfer problem remains.

The all16 source-max policy reduces mean original photograph source-any FPR to
3.52% for each probe. Its fixed-threshold source-bootstrap mean intervals are
2.65–4.46% for ResNet and 2.68–4.44% for CLIP. Mean copy TPR falls from 73.29% to
40.18% for ResNet and from 96.42% to 66.63% for CLIP. The paired TPR losses are
33.11 percentage points (95% interval 31.90–34.25) and 29.79 points
(28.47–31.13). These are substantial sensitivity costs, not a free repair.

On the new400 follow-up, all16 source-max probe source-any FPR means are 0.25%
for ResNet and 1.375% for CLIP, with copy TPRs of 43.77% and 74.18%. These are
observed transfer results. No unchanged-encoding exchangeability guarantee is
claimed for that sample. CLIP TPR split standard deviations are large: 0.132 on
the original all16 tests and 0.106 on the follow-up. Bootstrap intervals for
the five-split mean condition on these five saved models and thresholds;
reporting the split standard deviation avoids hiding that variability.

Some realized splits exceed a 5% source-any error rate even with the all16
policy: original ResNet seed 20260921 is 5.71%; CLIP seeds 20260921 and 20260923 are
6.62% and 5.71%. This does not convert the marginal rank guarantee into a
high-confidence statement about each saved threshold's population error rate.
The core13 conformal policy is also explicitly sensitive to uncalibrated
processing: CLIP's original all16 source-any mean reaches 11.14%, compared with
1.92% on the calibrated core13 views.

The additional SD-excluded CLIP probe with the all16 source-max cut yields
original source-any FPR 4.57% and TPR 73.04%; on new400 they are 0.625% and
81.51%. Its held-out SD copy TPR on new400 is 62.81%. ResNet's corresponding
overall TPRs are 22.65% and 24.32%, illustrating the sensitivity limitation of
this restricted training mix. These results are descriptive and are not used
to choose a winning head before scoring the new caption-paired benchmark.

Complete per-system and collection/generator/mark results are in
`per_seed_metrics.csv`, `mean_std_ci.csv` and `paired_policy_differences.csv`.
The same files in `sd_family_excluded/` report its single-split diagnostic;
there is no across-split standard deviation for that one split.

## Zero-event uncertainty

An ordinary source bootstrap has a degenerate zero-to-zero interval when no
source events are observed. That interval does not establish zero population
risk. `source_event_intervals.csv` adds collection-specific two-sided 95%
Clopper–Pearson intervals for each saved threshold. They require independent
identically distributed source bundles within that collection. With zero
events among 80 photographs, the upper endpoint is about 4.51%. These additional
intervals are retrospective descriptive intervals; they do not select a
threshold, pool overlapping splits as independent trials, or replace the
finite-source marginal conformal statement.

## Read-only evidence checks

`scripts/check_phase5_submission.py --component calibration` verifies the
source-disjoint roles, preserved head/cache hashes and timing of the fitted-head
and threshold locks. It independently recomputes each negative calibration cut
from the cached feature scores and every saved test-point rate, including the
SD-family exclusion control. It also checks the five-split means, sample
standard deviations, paired policy contrasts and exact source-event interval
endpoints. These checks perform no fitting, inference or threshold changes.

The full Phase5 completion gate also requires the finished caption-pair and
fair-watermark experiments. Preparatory protocols, partial images or empty
placeholders cannot satisfy it. The caption-pair checker uses the caption/photo
pair as its bootstrap unit; the watermark checker rebuilds signed information-bit
margin scores and the source-cluster intervals from the saved raw decoder
archives. Official ECC validity flags remain archived decoder outputs and are
not decoded again by this read-only checker. Neither an integrity check nor a
zero-event bootstrap interval establishes a zero population error rate.

Exact binomial coverage requires independent source events with a common
event probability. The calibration intervals above are therefore reported
within collection under their stated iid source assumption. The fair-watermark
archive also contains binomial-formula endpoints for pooled fixed-quota
collections. The checker verifies those saved formulas numerically; that does
not establish exact coverage for a mixture whose collection risks differ.
Pooled watermark means use the collection-stratified source bootstrap, and
zero-event risk bounds are reported separately within each collection.
The checker preserves leading-zero payload strings and round-trips saved
float64 CSV scores, so its comparisons retain the frozen whole-tie decision
rule at threshold boundaries.
Caption-pair bootstrap intervals likewise degenerate at zero observed events;
they cannot be interpreted as evidence of zero future error. See the
[SciPy binomial test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
for the single-probability binomial model and its exact interval method.

## Postgeneration overlap diagnostic

The caption-pair numerical audit additionally verifies
`outputs/phase5/paired_content/postgeneration_overlap_audit.json`, produced by
`scripts/audit_phase5_generated_overlap.py`. It checks all 200 generated native
and standardized pixel fingerprints against the 7,592 prior acquisition records,
the 200 selected photographs, and one another. A matching pre-JPEG 256-pixel
preparation supplies the same coarse 8-by-8 average-hash rule.

There are no exact decoded-pixel matches in any comparison. Three generated
images have ten total average-hash matches within Hamming distance two against
prior records. Neither the selected photographs nor the other generated images
produce such matches. These coarse matches are reported as indicators, not as
proof of duplicate scenes. The screen was performed **after generation and test
analysis**. All locked test items remain in the reported results; no filtering,
resampling, inference, model fitting or score changes follow the findings.
This post hoc diagnostic does not establish semantic independence or eliminate
pretraining overlap. The prospective overlap clearance applies to the selected
photographs, not to all subsequently generated pixels.

The supplementary `postgeneration_overlap_sensitivity.json` keeps the primary
200-pair results unchanged and evaluates identity only on the remaining 197
complete pairs using saved scores and exactly the same heads and cuts. Its
2,000 pair-bootstrap draws are conditional on this post hoc retained subset.
Joint ResNet/CLIP AUC is 0.7327/0.8227, versus 0.7292/0.8200 in the full test;
SD-excluded ResNet/CLIP is 0.6622/0.8448, versus 0.6542/0.8385. The weaker
transfer therefore persists in this descriptive sensitivity. It is not a new
untouched test, a model improvement, or a causal effect of removing the flags.
The ten matched prior records have original-audit roles of six training and
four test records; their collections are seven COCO, two Stable Diffusion and
one Midjourney. These role counts describe coarse-hash proximity, not confirmed
training/test duplication. The reusable CPU script is
`scripts/analyze_phase5_overlap_sensitivity.py`; the full checker recomputes
its twelve fixed-policy identity rows and input/subset fingerprints.

## Equal-target watermark interpretation

The completed comparison contains 1,004 sources and 8,912 processed images.
Both methods use the same information payloads and approximately 40 dB
embedding PSNR, with their own thresholds fixed at a nominal 5% calibration
copy FPR. Their observed default-query photograph FPRs over the four test
conditions are 4.6875% for DWT-DCT and 5.375% for TrustMark. They are not an
equal-test-FPR operating-point comparison; uncertainty conditions on the saved
calibration thresholds.

The calibrated margin uses evidence for the expected 16 information bits.
It is distinct from the secondary native rule: agreement on at least 13 bits
for DWT-DCT, versus official ECC validity, schema 1 and an exact 61-bit message
for TrustMark. After the 32-pixel round trip, TrustMark's margin accepts 136 of
204 default-payload images, while its complete native message verifies in only
5 of 204. Calling the former full payload recovery would overstate the result.
The respective source-stratified bootstrap intervals are 60.29–73.53% and
0.49–4.90%; each conditions on the fixed calibration thresholds and sources.
