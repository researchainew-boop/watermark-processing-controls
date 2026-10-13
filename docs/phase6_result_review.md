# Phase 6 outcome interpretation

This review uses the completed frozen primary evaluation, sealed additional
comparators and official-preprocessing detector outputs. It does not select a
replacement model, change cuts or refit scores. The independent numerical review
is retained in `outputs/phase6/verification/result_review.json`; its status
records an earlier intermediate review. Secondary results are now complete;
their final interpretation is added below, with a separate final manuscript gate.

## Verified primary scope

TrustMark/ResNet is the single registered primary. It reaches marked recall
0.744028 versus the probe's 0.020972 and tuning-selected capped comparator's
0.0775. The original conjunction of marked benefits, pooled unmarked
noninferiority and realized pooled photographic source FPR passes.
Unmarked recall remains 0.024583. Zero any-view harmful source events among
600 AI sources give the conditional pooled upper harm bound 0.00498044.
Simultaneous family upper bounds are 0.0202636, so a 0.01 margin for each family
is unresolved. Photographic source FPR is 0.0375, with source-bootstrap interval
0.02--0.0575; Open Images alone has 15/200=0.075 while COCO has 0/200.
The realized endpoint is pooled and is not a population/per-collection guarantee.

## Comparators that limit novelty

Complete-message union gives marked recall 0.718194 with identical observed
unmarked and photographic decisions. Primary-minus-union is 0.025833,
paired 95% interval 0.009583--0.041667. Ordinary full-feature logistic fusion
has higher marked recall 0.769028 and lower observed photographic source FPR
0.0325, with a small unmarked recall decrease to0.024167. Its failed tuning
eligibility does not justify concealing its final performance.

The pair term adds 0.004306 marked recall and full-message/ECC features add
0.006389. Removing the null penalty or received-view stability features slightly
improves marked recall, with the same unmarked decisions. Thus the proposed
objective's ingredients do not all have demonstrated incremental benefit.
The method is not established as superior to ordinary learned fusion.

The small pooled union benefit also depends on the edit mixture: primary
marked recall is lower for identity/crop and higher for JPEG50/lowres32.
Low-resolution primary recall remains only0.201111. Other decoder/backbone
systems are secondary and include much larger full-logistic advantages.

## Stronger passive baseline and processing

UFD common-input identity AUC is 0.89145 and native AUC0.919242. Its fixed
unmarked identity recalls are 0.555 and0.631667. Those operating points use
pooled photographic calibration and must not be compared as the same task/cut
as four-view marked detection. Native BigGAN invokes official crop padding;
the preprocessing contrast changes geometry, resolution and JPEG together.

A saved-feature, source-aligned identity diagnostic gives study ResNet/CLIP AUC
0.628317/0.646213 on the same common unmarked inputs. This secondary diagnostic
requires no new inference/cut selection and is separately fingerprinted in
`common_identity_probe_auc.json`. It confirms that the weak passive probe is a
practical bottleneck beyond differences in operating cuts.

Fresh GenImage/photos are not caption paired, source labels are collection
provenance, and upstream content/acquisition/pretraining confounds remain.
Previously examined caption-paired controls remain historical evidence.
Development and the fresh common128 setting also differ in resizing.

## Release and claim boundary

The release now includes only the two small study-owned passive linear heads
in addition to fusion parameters, with exact byte and strict field validation.
Images, per-source predictions/features and pretrained networks are excluded.
This improves replay prerequisites but is not an independent end-to-end replay.

These measured results support a narrower conditional fusion study. They do
not substantiate a claimed 8/10 novelty score or any numerical conference
acceptance probability. Stronger-passive development, a contribution beyond
ordinary logistic fusion and a newly locked independent confirmation would be
needed for the broader research ambition.

## Completed secondary attack failure

All nine fixed stress conditions are complete on100 sources:60 AI and40
photographic-input sources. At unchanged core cuts, primary marked recall is
0.746296 (95% source-bootstrap interval0.705556--0.782114). PGD raw/capped
marked recall is0.111111/0.15 despite native complete recovery0.855556/0.961111.
The independent nine-condition family cut reduces pooled marked recall to
0.000617284 (0--0.001851852), unmarked recall to0.001851852 and photographic-input
source positivity to0. Native verification and calibrated origin classification
are therefore distinct outcomes. Core cuts are outside their calibrated family.

The pixel target is reachable for100% of marked PGD inputs,5% of codec inputs
and81.6667% of diffusion inputs. Weaker raw attacks are not strengthened; these
are distortion caps rather than fully quality-matched attacks. SD1.5-family
regeneration uses one checkpoint and strength0.05 (one effective DDIM step).
Wrong-query margin acceptance is0.543827; native full wrong-message recovery is0.
Photographic margin calibration is not a complete-message verification guarantee.

A separate post hoc algebraic diagnostic verifies that `z+16 <= t` prevents
strict origin acceptance irrespective of decoder evidence. At the core cut,
88.8889% of raw-PGD marked views and85% of capped-PGD views satisfy this condition.
At the broader-family cut, the corresponding fractions are99.4444% and100%.
No scores, model parameters or cuts were changed. The diagnostic's inputs and
aggregate counts are fingerprinted in `cap_failure_diagnostic.json`.
