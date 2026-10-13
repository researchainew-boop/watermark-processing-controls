# Phase 6 official UniversalFakeDetect baseline

`scripts/phase6_native_baseline.py` preserves the published UniversalFakeDetect
CLIP ViT-L/14 backbone, unnormalized visual features, released linear head,
score polarity, and official CenterCrop224 preprocessing. It imports the pinned
asset identities already verified by `scripts/native_baseline.py`; no detector
parameters are refitted.

## Prospective protocol

The input is the root's `data_inventory/primary_source_manifest.csv`:800
independent calibration photographs (400COCO,400OpenImages), then1000test
sources (200COCO,200OpenImages,200BigGAN,200ADM,200VQDM). Each source receives
two identity views:

- Original native pixels, with source-inventory metadata rotation and lossless
  storage, no resize or JPEG re-encoding.
- The exact root-controlled common128->256 JPEG95 image. The baseline itself
  adds no resize.

Both use CenterCrop224, ToTensor, and the official CLIP channel normalization.
Images smaller than224 invoke the official transform's zero padding. Counts
and padding fractions are reported by source family. Native/common performance
differences also change geometry, resolution and encoding; do not claim that
one factor explains a failure. The common view addresses the unequal native
resolution/padding diagnostic.

## Independent decisions

Native and common each have a separately calibrated photographic threshold.
For800negative sources and nominal5%FPR, choose the761st ordered logit
(`ceil((800+1)*.95)`) and accept only strictly larger logits. Synthetic
calibration scores are never used. Also report the released logit0 threshold
as a descriptive diagnostic. No test tuning or polarity inversion is allowed.

The source/method root lock must exist before calibration inference. Its
`input_sha256` must include:

- `scripts/phase6_native_baseline.py`
- `outputs/phase6/native_baseline/plan.json`
- `.tools/extension-assets/ViT-L-14.pt`
- `.tools/extension-assets/univfd_fc_weights.pth`
- `.tools/extension-assets/official_clip_model.py`

The root `source_manifest_sha256` must match the finalized inventory. Before
test inference, both the baseline's `thresholds_locked.json` and the main
fusion's threshold lock must exist. The baseline threshold lock fingerprints
the independent calibration predictions and the root protocol.

## One inference pass and resumability

Inference stores128-view blocks with exact image IDs, source-manifest and root
protocol fingerprints, finite FP32 logits, and SHA256 receipts. A completed
block is reused. Changed or incomplete block identity fails explicitly. Final
predictions have a completion receipt; requesting a completed inference stage
reuses those predictions without another model pass.

The final report contains paired native/common AUC, TPR and actual FPR at the
locked cuts, with2000source bootstrap draws stratified by collection. Report
overall, each fresh generator against the400real negatives, and each real
collection against all600AI sources. Bootstrap samples are paired across
native/common views, so the AUC difference also gets an interval. Nominal5%
calibration depends on exchangeability and does not guarantee5%FPR under shift.

## Execution order

    python scripts/phase6_native_baseline.py --stage plan
    # Root freezes source/method protocol and includes the listed baseline inputs.
    python scripts/phase6_native_baseline.py --stage prepare
    python scripts/phase6_native_baseline.py --stage calibration --device cuda
    python scripts/phase6_native_baseline.py --stage calibrate
    # Main fusion thresholds must also be locked before test.
    python scripts/phase6_native_baseline.py --stage test --device cuda
    python scripts/phase6_native_baseline.py --stage analyze

Batch size4 follows the previously completed native baseline. That run took
38seconds for192sources on the GTX1050Ti. Extrapolating gives about12minutes
for3600views, plus model loading, hash checks and bootstrap analysis. This is an
estimate, not a promised runtime. Run only when the GPU is available.

No final-source inference has been performed by the plan stage. Official
detector exposure to public benchmarks or pretraining cannot be excluded;
the fresh families are held out from this study's supervised development.
