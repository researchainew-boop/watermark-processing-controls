# Completed Phase 3 detection controls

The original heads and the 192-source frozen sample remain unchanged. This report describes retrospective controls and repeated source splits; it does not turn the source pool into a fresh test set.

## Matched unmarked edits

The same fourteen transformations were applied directly to all 1,396 unmarked AI images in the original test. The corresponding marked copies use the same 284 real sources and attack settings. Adding the 19,544 copies gives 46,424 rows from 1,680 sources.

- RESNET probe: marked-edit AUC 0.9415, unmarked-edit AUC 0.9408; test-ROC TPR at 5% FPR 0.7501 and 0.7470. The mean marked-minus-unmarked probability response is 0.002194, with source-clustered 95% interval [0.000821, 0.003551].
- CLIP probe: marked-edit AUC 0.9935, unmarked-edit AUC 0.9934; test-ROC TPR at 5% FPR 0.9723 and 0.9723. The mean marked-minus-unmarked probability response is 0.000306, with source-clustered 95% interval [-0.000320, 0.000937].

These comparisons reduce the concern that the probe merely reads the embedded mark. They do not remove collection, scene, codec, or backbone pretraining confounds.

## Five split seeds and a second real collection

The source pool contains the original 5,823 screened images plus 1,000 separately selected Open Images photographs. Each split contains 4,102 training, 878 validation, and 1,843 test sources. Open Images contributes 700/150/150 sources, and all 962 Midjourney sources remain in test. Training and validation each use thirteen original conditions. Tests include all sixteen original conditions and fourteen additional unmarked edits per AI source: 49,158 rows, with thirty per AI source and sixteen per real source.

Unmarked edited copies never enter training or calibration. The probe fits training only, fusion fits validation only, and the payload gate remains 13/16. The three originally held-out settings stay excluded from training and calibration.

- CLIP / full / fusion: AUC 0.9761 ± 0.0021; test-ROC TPR at 5% FPR 0.9336 ± 0.0161.
- CLIP / full / gated: AUC 0.9876 ± 0.0017; test-ROC TPR at 5% FPR 0.9468 ± 0.0095.
- CLIP / full / probe: AUC 0.9863 ± 0.0023; test-ROC TPR at 5% FPR 0.9336 ± 0.0106.
- CLIP / no_attacks / fusion: AUC 0.9670 ± 0.0031; test-ROC TPR at 5% FPR 0.8985 ± 0.0247.
- CLIP / no_attacks / gated: AUC 0.9825 ± 0.0029; test-ROC TPR at 5% FPR 0.9060 ± 0.0207.
- CLIP / no_attacks / probe: AUC 0.9770 ± 0.0036; test-ROC TPR at 5% FPR 0.8740 ± 0.0224.
- CLIP / no_matched_reals / fusion: AUC 0.9484 ± 0.0059; test-ROC TPR at 5% FPR 0.6280 ± 0.0478.
- CLIP / no_matched_reals / gated: AUC 0.9533 ± 0.0075; test-ROC TPR at 5% FPR 0.5402 ± 0.0161.
- CLIP / no_matched_reals / probe: AUC 0.9727 ± 0.0046; test-ROC TPR at 5% FPR 0.8532 ± 0.0271.
- RESNET / full / fusion: AUC 0.9310 ± 0.0028; test-ROC TPR at 5% FPR 0.7185 ± 0.0434.
- RESNET / full / gated: AUC 0.9472 ± 0.0006; test-ROC TPR at 5% FPR 0.7781 ± 0.0078.
- RESNET / full / probe: AUC 0.9288 ± 0.0010; test-ROC TPR at 5% FPR 0.7208 ± 0.0082.
- RESNET / no_attacks / fusion: AUC 0.9123 ± 0.0070; test-ROC TPR at 5% FPR 0.5796 ± 0.0472.
- RESNET / no_attacks / gated: AUC 0.9219 ± 0.0083; test-ROC TPR at 5% FPR 0.6302 ± 0.1398.
- RESNET / no_attacks / probe: AUC 0.9129 ± 0.0062; test-ROC TPR at 5% FPR 0.6559 ± 0.0293.
- RESNET / no_matched_reals / fusion: AUC 0.8854 ± 0.0069; test-ROC TPR at 5% FPR 0.5017 ± 0.0355.
- RESNET / no_matched_reals / gated: AUC 0.8770 ± 0.0121; test-ROC TPR at 5% FPR 0.3278 ± 0.0267.
- RESNET / no_matched_reals / probe: AUC 0.8859 ± 0.0104; test-ROC TPR at 5% FPR 0.5174 ± 0.0427.

## Results by real source

- CLIP probe versus coco: AUC 0.9903 ± 0.0018; test-ROC TPR at 5% FPR 0.9520 ± 0.0078.
- CLIP probe versus openimages: AUC 0.9786 ± 0.0036; test-ROC TPR at 5% FPR 0.8985 ± 0.0141.
- RESNET probe versus coco: AUC 0.9373 ± 0.0024; test-ROC TPR at 5% FPR 0.7561 ± 0.0074.
- RESNET probe versus openimages: AUC 0.9125 ± 0.0062; test-ROC TPR at 5% FPR 0.6602 ± 0.0244.

The ± quantities are sample standard deviations over five splits, not confidence intervals. Test sources overlap across splits and the same Midjourney holdout is reused. Collection-specific performance and the fixed validation-threshold error rates are retained in the underlying reports.

## Artifacts

- `outputs/phase3/controls/unmarked_frozen_results.json`: unchanged-head comparisons and paired source intervals.
- `outputs/phase3/controls/split_seeds/mean_std_metrics.csv`: all systems and ablations, separately for COCO and Open Images.
- `outputs/phase3/controls/split_seeds/source_counts.csv`: source counts by generator and split for every seed.
- `outputs/phase3/controls/split_seeds/complete.json`: checksums, software versions, source assignment, and original-head preservation.
- `outputs/phase3/controls/summary.json`: compact machine-readable report.
