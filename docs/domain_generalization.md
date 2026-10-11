# Real-collection transfer and validation calibration

This diagnostic uses existing frozen features and the locked source assignment
for split seed 20260920. It measures how the real-image training collection
affects detection and how a validation-selected false-positive target transfers.
It is a retrospective diagnostic on an already examined source pool.

## Equal-size collection comparison

The three regimes use identical synthetic training sources and augmentation
conditions. Each uses 700 real training sources: COCO only, Open Images only, or
350 from each collection. Source selection is deterministic by a SHA-256 rank
specified before fitting. Midjourney sources remain test only.

The original validation source split is divided into independent source halves
for fusion fitting and threshold calibration. Each half contains 75 real
sources and the same synthetic source cohort across the three regimes. The
joint regime uses 37/38 photographs per collection in the fusion-fit half and
38/37 in the calibration half. Neither probe nor fusion fitting sees calibration
sources. Every source contributes the same 13 eligible conditions within its
role; the three held-out attacks are excluded from fitting and calibration.

All regimes are evaluated on one common test set: 150 COCO photographs, 150
Open Images photographs, and 1,405 synthetic sources, including 962 Midjourney
sources. This gives 46,950 copies: 16 per photograph and 30 per synthetic source.
The synthetic copies include the 14 matched unmarked edits. Sampling and source
roles are saved in `outputs/phase4/domain_generalization/source_assignments.csv`;
the protocol is saved before any diagnostic fit.

## Thresholds and uncertainty

For each system, the threshold is the lowest observed unsaturated logit with
empirical false-positive rate at most 5% on calibration photographs. Scores are
accepted with `logit >= threshold`; tied scores are kept together. The selected
threshold is saved before test scoring. This target concerns the average
false-positive rate across the eligible image copies, not a guarantee of a
source-level family-wise error rate or of a population false-positive rate.

The test reports actual false-positive and true-positive rates at those fixed
thresholds, alongside ROC-AUC. Rates are split by real collection and by seen
generators versus held-out Midjourney. Confidence intervals use 1,000 percentile
source-cluster bootstrap replicates, stratified by real collection or AI
generator; all copies of a sampled source move together. Joint-versus-single
collection AUC differences use paired bootstrap draws of the identical test
copies.

## Frozen-head thresholds for the new replication

`frozen_seed_thresholds.json` separately saves 5% false-positive target
thresholds for the already fitted phase-3 full probe, fusion, and gated systems,
for both ResNet and CLIP. The scope is all 438 real and 440 synthetic validation
sources, with 13 eligible conditions per source. It is locked before selection
of the new replication sources. The validation real-copy FPR is 0.049877 for
all six systems. Original head hashes and the validation scope are saved.

That existing validation set is independent of probe training, but the existing
fusion heads were fitted on those same validation rows. Therefore its fusion
and gate thresholds are retrospective calibration, not an independent fusion
calibration set. The separate source halves in the new domain diagnostic above
avoid that reuse.

## Measured findings

All six new probe/fusion pairs completed, yielding 18 systems on the same test
copies. All thresholds attained 4.923% false positives on their independent
calibration photographs. Original frozen heads remain unchanged.

ResNet probe AUC is 0.89144 with COCO-only training, 0.90231 with Open
Images-only training, and 0.91551 with equal-size joint training. The paired
joint gain is 0.02407 over COCO-only (95% CI 0.01479 to 0.03447) and 0.01319
over Open Images-only (0.00450 to 0.02248). CLIP probe AUC is 0.97685, 0.98132,
and 0.98180, respectively. Joint CLIP improves over COCO-only by 0.00495
(0.00172 to 0.00836); its difference from Open Images-only is 0.00048
(-0.00266 to 0.00364).

High AUC does not ensure transfer of the calibration target. On the 13
conditions matched to calibration, COCO-trained CLIP has test FPR 6.000%
against COCO and 30.923% against Open Images (Open Images CI 24.614% to
37.795%). Open Images-trained CLIP has FPR 6.769% against COCO and 3.692%
against Open Images. Joint CLIP has FPR 3.949% against COCO and 9.949% against
Open Images (Open Images CI 6.256% to 14.668%).

The corresponding ResNet matched-condition FPRs against COCO/Open Images are
7.231%/15.795% with COCO-only training, 8.410%/6.000% with Open Images-only
training, and 8.564%/10.205% with joint training. Thus equal-size joint training
improves ResNet ranking but does not eliminate collection-specific calibration
failure. The single Open Images regime gives better photographic calibration
in this diagnostic; this observed result does not justify selecting it after
looking at this test and calling that an independent validation.

Across all 16 real-image test conditions, including the three attacks absent
from calibration, pooled FPRs are 11.771%, 7.563%, and 10.042% for ResNet, and
22.979%, 9.000%, and 10.958% for CLIP. Full results separate the 13 matched
conditions from the three held-out conditions, preserving the distinction
between collection transfer and attack-distribution shift.

At the unchanged joint thresholds on the expanded test, ResNet probe TPR is
76.365% overall, 79.398% for seen generators, and 74.969% for held-out
Midjourney. CLIP probe TPR is 95.798% overall, 96.283% for seen generators,
and 95.575% for Midjourney. Their corresponding pooled photographic FPRs are
10.042% and 10.958%, so these TPRs are not performance at a realized 5% test
FPR.

`summary.json` contains source-cluster intervals, collection-specific rates,
marked/unmarked TPRs, and paired AUC differences. `metrics.csv` gives the 18
system-level results. `attack_scope_false_positives.csv` gives 108 fixed-
threshold photographic rates separated by condition scope and collection.

## Reproduction

Run `scripts/phase4_domain_generalization.py --stage frozen-calibration` to
compute the frozen-head thresholds, then `--stage domains` for the collection
comparison and `--stage attack-scope` for the matched-condition versus held-out
condition false-positive breakdown. This is CPU-only cached-feature analysis. It writes new diagnostic
heads, predictions, intervals, and metrics under
`outputs/phase4/domain_generalization/`; it does not change original heads,
features, split assignments, or fresh-sample predictions. The protocol prevents
source overlap among fitting, calibration, and test roles.

## Limits

These controls can show whether joint collection training reduces a measured
transfer failure. They cannot eliminate content, collection, compression,
pretraining, or semantic-duplicate confounds. The diagnostic uses one existing
split seed and the same examined source pool; its bootstrap intervals describe
source-sampling uncertainty conditional on the fitted model, not uncertainty
from retraining or an independent replication.
