# Native-resolution published-detector check

This is a retrospective preprocessing comparison on the same 192 sources used
in the separately frozen fresh experiment. It is not another untouched test
sample. The study's ResNet/CLIP heads and validation thresholds are unchanged;
their fresh inference is not repeated.

## Inputs and fixed method

`scripts/native_baseline.py --stage prepare` reconstructs 96 DALL-E archive
images, 48 unused COCO photographs, and 48 Open Images photographs from their
original archive or S3 bytes. Both original-file and original decoded-pixel
hashes must match the locked fresh manifest. Open Images rotation matches its
metadata and the earlier fresh preparation. RGB conversion and lossless PNG
storage preserve native dimensions; no resize or JPEG round trip is applied.

The unchanged released UniversalFakeDetect ViT-L/14 backbone and linear head
use the pinned official model implementation, CenterCrop(224), ToTensor, and
CLIP mean/std normalization. There is no Resize and no feature normalization.
The published logit-zero operating point and original score polarity are kept.
The protocol records the expected backbone, head, and model-code checksums
before inference.

The comparison joins these native scores to the archived detector scores for
the exact same source IDs after INTER_AREA 256x256 resizing and common JPEG95.
Therefore the contrast jointly changes resizing and JPEG re-encoding. It does
not isolate resizing as the cause of any observed failure. The native result
is retained regardless of performance; removing this baseline from the short
main paper does not remove its negative result from the research archive.

## Reproduction

Use the recorded GPU experiment environment, original fresh selection artifacts,
original input archives/S3 files, and the official cached checkpoint assets.

    python scripts/native_baseline.py --stage prepare
    python scripts/native_baseline.py --stage evaluate --device cuda --batch-size 4
    python scripts/native_baseline.py --stage analyze

Preparation freezes the manifest and protocol under
`outputs/phase3/native_baseline/`; inference refuses to overwrite a completed
prediction file. Analysis reports source-stratified paired bootstrap intervals
(2,000 draws, seed 20261009), overall and separately against each real-image
collection. Real and AI source IDs are resampled within their collections.

Native images are under `data/phase3/native_baseline/` and are excluded from the
anonymous code package. The package contains this method, code, and aggregate
results after the run completes; it does not grant redistribution rights for
the images or downloaded checkpoints.

## Completed measurements

The native inference pass completed on all 192 sources. Overall ROC-AUC is
0.6593 (paired source-bootstrap 95% CI 0.5834–0.7328), versus 0.4423
(0.3639–0.5284) for the archived 256/JPEG95 inputs. The paired AUC difference is
+0.2170 (0.1473–0.2915). Against COCO alone, native AUC is 0.7530
(0.6660–0.8353); against Open Images alone, it is 0.5655 (0.4683–0.6593).

At the unchanged published logit-zero threshold, native false-positive rate is
3/96 = 0.03125 and true-positive rate is 20/96 = 0.20833. All three false
positives are Open Images photographs (3/48); none are COCO photographs. The
archived common-processing run accepted no sources at this threshold.

Native AUC did not remain near 0.44, so the requested conditional removal of
the baseline is not triggered. The result is modest and varies by real-image
collection. Its failure cause remains unconfirmed; the preprocessing contrast
alone does not establish a complete explanation. The complete protocol,
timings, paired predictions, and interval estimates remain in
`outputs/phase3/native_baseline/` without changes to score polarity or fitting.
