# Training-slot-controlled real-edit ablation

This retrospective follow-up addresses the training-row-count and balanced-class-weight confounds in the earlier `no_matched_reals` ablation. It does not change the original conference or Phase 3 fits. The protocol was saved in `outputs/phase4/size_controlled/protocol.json` before any new model was fitted.

## Design

Both frozen ResNet and CLIP feature probes are fitted on each of the five existing source assignments (20260920–20260924). All Midjourney sources remain held out. The source pool had already been examined; these are diagnostic repeats, not independent external validation.

The three primary training treatments use the same source assignments, eligible conditions, unchanged validation cohort and unchanged expanded test cohort:

- **Full:** thirteen original eligible rows per source, including eleven transformed photographs per real source.
- **Repeated clean real:** replace each transformed photograph's feature vector with the clean feature vector from that same real source. Source identities, source counts, row labels, row ordering, AI rows, wrong-payload photographic rows and total training slots remain identical to full training.
- **Reduced-row removal:** remove the eleven transformed-photograph rows per real source, reproducing the earlier ablation.

A fourth treatment was declared in a separately saved amendment after the three-arm run finished, and before fitting any fourth-arm model:

- **JPEG-only repeated:** replace the eleven real-edit slots with alternating cached JPEG-quality-70 and JPEG-quality-50 features from that same real source (six and five slots, respectively, in original manifest order). Keep the clean and wrong-payload real rows, all AI rows, labels and slots unchanged. This compares the diverse original suite with a narrow two-quality real augmentation baseline; it does not equalize augmentation diversity.

Each equal-slot treatment has 53,326 rows from 4,102 sources: 26,754 AI rows and 26,572 real rows. Exactly 22,484 transformed-real slots are replaced in the repeated-clean treatment. The reduced-row treatment has 30,842 rows, including 4,088 real rows; it has different balanced class weights. Every treatment uses 11,414 unchanged validation rows and 49,158 unchanged test rows from 1,843 sources. Test copies include both marked and unmarked AI edits.

The first thirty heads and ten amended JPEG-only heads are newly fitted with the same CPU runtime, four-thread limit, frozen features, `LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)`, default L2 regularization, LBFGS solver and default tolerance. The full and reduced-row coefficients are compared with their preserved earlier fits. Iteration counts, elapsed fit times, warnings and convergence are saved for every new head. The first three-arm protocol, scientific source and result aggregates remain archived, and the fourth-arm amendment binds their hashes.

## Operating points and uncertainty

For each head, the operating threshold is the lowest attainable logit threshold with empirical **validation FPR at most 5%**. This maximizes attainable validation TPR under that constraint; score ties are not randomized. The threshold is then fixed for test inference. Both test TPR and actual test FPR at this threshold are reported. Test-ROC TPR at 5% FPR is retained in a separately named column as a descriptive statistic; it is not used for threshold selection.

The source-cluster paired bootstrap uses 2,000 resamples, stratified by source collection. A unique source is sampled from the union of all five test cohorts, and all of its derivative rows and split memberships receive the same multiplicity. The same resample is used for every treatment, backbone and split. This preserves dependency from overlapping test cohorts and the reused Midjourney holdout. Confidence intervals are percentile intervals for per-split differences and the arithmetic mean of the five metric differences. They condition on the fitted heads, fixed validation thresholds and fixed split memberships; training and threshold-selection uncertainty are excluded. Sample standard deviations across the five splits are also reported, and are not confidence intervals.

Row multiplicities implement the ordinary cluster bootstrap exactly, including tied-score ROC handling. The script verifies its unresampled weighted AUC against the standard ROC-AUC calculation as part of measurement verification.

## Interpretation limits

Repeated clean features control row count, class weight, source identity and optimization budget. Repetition does not recreate the diversity or information content of eleven distinct augmentations. Therefore a full-versus-repeated-clean improvement supports the usefulness of diverse real edits beyond a simple slot-count explanation. It does **not** establish that matching these exact AI transformations is uniquely beneficial, or exclude benefits from other equally diverse real augmentation. A future experiment should compare an equal-size, equally diverse alternative-real-augmentation arm on an independently collected benchmark.

This analysis concerns frozen-feature probes only. It does not fit fusion, watermark, secondary attack or blur/low-resolution heads, and it does not change the manuscript automatically.

## Reproduction and artifacts

On a new reproduction checkout with the required existing caches, run `python scripts/phase4_size_controlled.py --prepare-only` to freeze the primary protocol, followed by `python scripts/phase4_size_controlled.py` using the same recorded NumPy/scikit-learn versions. Then run `python scripts/phase4_size_controlled.py --jpeg-only` to freeze and fit the JPEG-only amendment. In the completed research checkout, the primary aggregates are protected from a three-arm rerun; `--jpeg-only` reuses unchanged saved heads and shared source resamples. The script refuses changed inputs, source assignments, earlier heads or protocol-defining settings. All output files are under `outputs/phase4/size_controlled/`.

- `protocol.json`: treatment definitions, source counts, fit specification and frozen input/model hashes.
- `jpeg_protocol_amendment.json`: separately frozen fourth-arm definition and links to the preserved primary protocol and code.
- `primary_archive/` and `initial_script_snapshot.py`: first three-arm protocol, source and aggregate history.
- `convergence.json`: all forty fits and coefficient agreement with earlier full/removal heads.
- `per_seed_metrics.csv` and `mean_std_metrics.csv`: overall, COCO and Open Images operating-point metrics.
- `paired_source_bootstrap.csv`: paired treatment differences, including the dependent five-seed mean.
- Seed directories: separately saved new heads, validation thresholds and source-level test predictions.
- `bootstrap_source_union.csv` and `bootstrap_source_multiplicities.npz`: exact shared source resamples.
- `complete.json`: completion, runtime, measurements and output hashes.

## Completed primary results

Across five splits, ResNet AUC is 0.928834 ± 0.000960 for full training, 0.882160 ± 0.011464 for equal-slot repeated-clean training, and 0.885864 ± 0.010431 for reduced-row removal. The full-minus-repeated-clean AUC difference is 0.046675, with a paired source-cluster 95% confidence interval [0.043516, 0.050108]. Its validation-threshold test TPR is 0.739004 ± 0.008365 versus 0.607265 ± 0.025656; the actual test FPR is 0.056564 ± 0.004222 versus 0.074772 ± 0.011137. The paired TPR gain is 0.131739 [0.123347, 0.139983].

CLIP AUC is 0.986326 ± 0.002307 for full training, 0.971577 ± 0.005011 for repeated-clean training, and 0.972657 ± 0.004610 for reduced-row removal. Its full-minus-repeated-clean AUC difference is 0.014748 [0.013303, 0.016265]. Validation-threshold test TPR is 0.965163 ± 0.003816 versus 0.935042 ± 0.006310, with actual test FPR 0.098231 ± 0.009898 versus 0.117180 ± 0.009950. Thus a 5% validation false-positive target does not guarantee the same test false-positive rate, especially for CLIP on this expanded attack cohort.

All thirty primary heads converged without warnings, with at most 175 of 3,000 iterations. Independently refitted full and removal heads reproduce the earlier coefficients exactly. The first run took 297.69 seconds, including input verification, fits, 2,000 shared source-cluster resamples and output generation. Earlier fitted heads and frozen inputs remained unchanged.

These results support a benefit from real-image augmentation beyond extra training slots and balanced-weight changes. They do not show that exact matching is uniquely beneficial compared with other equally diverse real augmentation.

## Completed JPEG-only amendment

The fourth arm was frozen at 2026-10-10 04:37:19 UTC before any JPEG-only fit. ResNet JPEG-only AUC is 0.898495 ± 0.004031, validation-threshold test TPR is 0.688584 ± 0.015327, and actual test FPR is 0.085987 ± 0.006390. Its descriptive test-ROC TPR at 5% FPR is 0.568218 ± 0.028200. Full training improves AUC by 0.030340 [0.027564, 0.033118] and validation-threshold test TPR by 0.050420 [0.044256, 0.056681], with an actual test-FPR difference of −0.029424 [−0.033821, −0.024819].

CLIP JPEG-only AUC is 0.974128 ± 0.002711, validation-threshold test TPR is 0.940508 ± 0.006776, and actual test FPR is 0.118550 ± 0.007197. Its descriptive test-ROC TPR at 5% FPR is 0.856019 ± 0.018461. Full training improves AUC by 0.012198 [0.011128, 0.013390] and validation-threshold test TPR by 0.024655 [0.021185, 0.028351], with an actual test-FPR difference of −0.020320 [−0.024566, −0.015834].

The broad real-edit suite outperforms these same-source, equal-slot JPEG-only repetitions on the fixed test cohort. The comparison still differs in transformation diversity and information content; it does not establish an isolated causal effect of exact matching.

All forty heads converged without warnings, at most 175 iterations. Pure fitting consumed 173.07 seconds. The primary and amended runs together consumed 423.22 seconds, including input verification, bootstrap analysis and file generation (297.69 seconds primary, 125.53 seconds amendment). The shared bootstrap source union contains 4,238 sources: 1,101 Stable Diffusion, 1,071 COCO, 962 Midjourney, 566 Open Images and 538 DALL-E 3. The original fourteen conference heads, sixty five-seed heads and every recorded frozen input remained unchanged. Exact full-precision values and per-source-family diagnostics are in the CSV and completion artifacts.
