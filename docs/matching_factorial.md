# Balanced processing-matching diagnostic

This is a retrospective experiment on already examined sources and cached
frozen features. It estimates whether giving real and AI training images the
same processing families helps detection when the number of source images,
training slots and distinct processing families is held fixed.

## Design and source scope

Set A is JPEG70 and Gaussian blur3. Set B is Gaussian noise sigma6 and
resize0.8. The four cells AA, AB, BA and BB select the real edit set first and
the unmarked AI edit set second. Each training source contributes its clean
image and the two chosen edits. No marked AI or sham-marked real features
enter this diagnostic. All four cells share source identities, labels, row
counts, class weights, frozen backbones, solver and validation/test conditions.

The available fitting pool contains 1,834 non-Midjourney AI sources with all
four unmarked edit features, 1,920 COCO sources and 1,000 Open Images sources.
All 962 Midjourney sources remain test only. The resulting 5,716-source pool
is smaller than the original repeated-split pool; these are different fits.
Each of the five original source assignments retains every existing test
source: 1,843 sources and 9,215 copies, with clean plus all four unmarked
edits, independent of the cell used for training. Training ranges from 3,184
to 3,206 sources across seeds, with three rows per source. The five tests
overlap and reuse the same Midjourney holdout.

The primary endpoint is the five-seed mean ROC-AUC contrast
`0.5 * (AA + BB - AB - BA)`. Balancing the A/B marginal frequencies cancels
augmentation-set main effects. Fixed-validation-threshold TPR and observed
test FPR are secondary. The separate test-ROC TPR at 5% FPR is descriptive,
not a deployable threshold. Validation thresholds retain whole ties and
are fixed before test scoring.

## Completed measurements

All 40 fits (two backbones, five seeds, four cells) converged. Maximum solver
iterations were 262. Original heads and input hashes were preserved.
ResNet overall AUC, mean ± sample SD, is AA 0.926470 ± 0.002383,
AB 0.920689 ± 0.002740, BA 0.921529 ± 0.003403 and
BB 0.926858 ± 0.002306. CLIP is AA 0.991054 ± 0.001306,
AB 0.986938 ± 0.002012, BA 0.986640 ± 0.000948 and
BB 0.992489 ± 0.001194.

The matched-minus-mismatched AUC contrast is 0.005555 for ResNet
(95% paired source interval 0.005008–0.006164) and 0.004983 for CLIP
(0.004410–0.005576). The corresponding fixed-threshold TPR differences are
0.018221 (0.015636–0.020641) and 0.023815 (0.021110–0.026550).
Test FPR differences are approximately zero for ResNet
(−0.001993–0.001995) and 0.001187 for CLIP (−0.002158–0.004492).
See `outputs/phase5/matching_factorial/complete.json`,
`mean_std_metrics.csv` and `paired_source_bootstrap.csv` for full precision.

Uncertainty uses 2,000 collection-stratified draws over the union of 4,238
test sources, preserving every derivative and split membership. Identical
source multiplicities apply to all cells and both backbones. Intervals
condition on these fitted heads, thresholds and source assignments; they
exclude training uncertainty. Split SD is sensitivity to the saved splits,
not independent experimental replication.

## Reproduction

Supply the original and Phase 3 unmarked feature caches, source manifests and
five-seed assignments identified by `protocol.json`. Use the recorded CPU
environment. The protocol-only stage performs no model fitting:

    python scripts/phase5_matching_factorial.py --prepare-only
    python scripts/phase5_matching_factorial.py --bootstrap-samples 2000
    python scripts/check_phase5_submission.py --component matching

The first two commands describe independent reconstruction; do not rerun
them over the archived completed output directory. The completion record
binds the protocol, input/head hashes, convergence and output fingerprints.
Source predictions, fitted heads and feature caches are excluded from the
anonymous snapshot; aggregate metrics and fingerprints are included.

## Interpretation limits

The result supports processing matching for these four transforms, frozen
features and this restricted fitting pool. Condition and family counts are
equal, but distortion and information content are not equalized. It does not
show that matching uniquely explains the broader real-edit-suite gain, remove
collection/pretraining confounds, or provide evidence from an untouched test.
