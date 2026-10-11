# Fixed Content and Encoding Controls

This extension defines and prepares additional controls before their new
features or detector scores are computed. It reuses the screened public
sources whose original experiment results were already observed. It is a
retrospective study with a newly fixed extension policy, not a newly collected
prospective confirmation dataset and not a registered study.

## Questions answered by these controls

The extension measures origin discrimination after the same new encoding
operation on both classes, in an approximately matched-content sample.
Comparing conditions on identical source IDs tests sensitivity to the new
encoding. It can expose dependence on processing or collection cues.
It cannot prove that prior compression, generator provenance, style,
or content confounds have been eliminated.

## Frozen source selection

The selection starts from outputs/conference_clean/manifest_sources.csv,
with its existing source IDs and split assignments. The screen retained
5,823 sources. Only original validation and test sources are prepared as
extension images. Training sources are used solely to set a matching
diagnostic. No source is moved to another split.

configs/extension_encoding.yaml fixes seed 20261007 and these counts:

- Validation: 60 DiffusionDB plus 60 DALL·E 3 images, paired with 120
  distinct COCO photographs; 240 total sources.
- Test: 60 DiffusionDB, 60 DALL·E 3, and 60 Midjourney images, paired with
  180 distinct COCO photographs; 360 total sources.
- Total: 300 generated images and 300 real photographs, or 600 sources
  in 300 pairs.

Synthetic sampling orders eligible source IDs by the SHA-256 of the fixed
seed and ID, then takes the specified prefix for each split/generator.
This avoids sampling based on classifier results or manually preferred
images. The Midjourney files remain test only.

## Approximate content matching

Clean, unmarked CLIP image embeddings are read from the existing immutable
feature caches, normalized, and compared by cosine similarity. Within each
split, a one-to-one assignment maximizes total similarity between the fixed
synthetic sample and candidate COCO photographs. Real partners cannot repeat.

The matching rule is fixed before the new controlled-image results. A
diagnostic caliper is the fifth percentile of nearest-real cosine similarity
for a hash-selected training-only sample of 300 real photographs and 100
images from each seen synthetic collection. Below-caliper pairs remain in
the extension; there is no test-dependent exclusion or adjustment.

The matching does not use origin probabilities, fitted detector coefficients,
watermark scores, test ROC values, captions, or prompts. Test embeddings are
used to apply the locked matching rule, not to choose its parameters.
Accessing original test content nevertheless makes this a conditional
selected-subset analysis rather than an untouched independent test.

CLIP similarity can reflect style or origin as well as scene content.
Using CLIP features for both matching and one evaluated detector conditions
that comparison on its representation. Cosine pairing does not guarantee
the same objects, subjects, composition, or semantic content. Report
similarity distributions and below-caliper counts. A blinded review of
pair content would be useful, with any review-driven exclusions reserved
for a separately defined future sample.

Local discovery found no COCO caption/annotation file at task initialization.
The runner records available local caption files in summary.json. It does
not invent captions or generate matching prompts from unavailable annotations.
New caption-conditioned synthetic generation would constitute a separate
experiment requiring generation settings, seeds, and original output records.

## Shared image pipeline

Each selected stored source is decoded using the same OpenCV color decoder
and resized with area interpolation to 256×256. Four conditions are then
prepared, all stored as PNG for common feature-extraction inputs:

1. Common PNG: the standardized pixels, with no additional JPEG round trip.
2. JPEG 95: encode the standardized pixels at quality 95 and decode.
3. JPEG 75: encode those same standardized pixels at quality 75 and decode.
4. JPEG 50: encode those same standardized pixels at quality 50 and decode.

Every condition derives directly from the same standardized source pixels,
not from a previously encoded extension condition. Encoding settings, encoder
version, dimensions, and storage format are identical across classes.
No watermark is added in this encoding control.

The synthetic stored sources were already re-saved at JPEG quality 95 during
the original collection, whereas COCO retained supplied JPEG encodings.
The extension does not recover generator-native originals or erase that
history. Identical subsequent encoding cannot guarantee identical compression
traces. Metadata and hashes distinguish stored-source lineage from native
generator output claims.

## Run and saved artifacts

Run from the project directory with the recorded CPU image/numerical
dependencies, original source files, and CLIP caches:

    python scripts/prepare_encoding_controls.py

The script freezes selection before preparing images and refuses to replace
an existing frozen selection. A new run requires a separately named output
and data directory in a new configuration. It does not import Torch,
train a model, recompute CLIP embeddings, or alter original files.

Images are written under data/extension_controls/encoding/. Artifacts
under outputs/extension_controls/encoding/ include:

- selected_sources.csv and content_pairs.csv: fixed IDs, partner assignment,
  collection, original split, and matching diagnostics.
- matching_calibration_sources.csv: training-only diagnostic sample.
- selection.json: policy frozen before new image features/results.
- native_inventory.csv: dimensions, format, JPEG quantization/subsampling,
  file size, and ICC/EXIF presence for all retained validation/test sources.
- source_lineage.csv: selected source-file hashes, decoded-pixel hashes,
  standardized-pixel hashes, and comparison to the previous base pixels.
- manifest.csv: all controlled images, source and pair IDs, settings, output
  file and pixel hashes, and the unchanged split.
- summary.json and provenance.json: executed counts, diagnostics, dependency
  versions, input hashes, and explicit limits on provenance and independence.

The default allocation prepares 2,400 images from 600 sources. Treat
manifest.csv as the extraction input, not a count of independent images.

The preparation executed this allocation successfully: 300 pairs and 2,400
images, with 240 validation sources and 360 test sources. The metadata
inventory covers all 2,408 retained validation/test sources. All 600 common
PNG conditions exactly match the decoded pixels of their previous standard
base images. The training-only diagnostic caliper is 0.54556; 49 pairs fall
below it and remain included. No local COCO captions were found. Model
training or fresh feature extraction was not part of this preparation.

## Fixed evaluation policy and executed extraction

Reuse the existing fitted origin probes for the primary comparison. Extract
fresh features for each controlled image using the same recorded backbones
and preprocessing. Rank using linear decision scores in float64. Compare
conditions on the same source IDs and report the original validation-fixed
thresholds as the primary fixed-threshold decision analysis.

The secondary calibration uses only the extension validation set. Its fixed
rule selects maximum TPR at FPR at most 5% across the four equally weighted
conditions, choosing the highest threshold on ties, and reports actual test
FPR/TPR. JPEG quality, matching rules, and thresholds are not selected from
test performance.

Evaluation has completed with the unchanged full and no-matched-real-control
probe/fusion coefficients. Each backbone reused 600 common-PNG vectors after
exact base-pixel checks, original-cache hash checks, and seven fresh inference
anchors; 1,800 JPEG vectors per backbone were extracted on CUDA with batch
size eight. No models were refitted. The frozen embeddings, predictions,
encoding_metrics.json (including calibration thresholds), and
evaluation_provenance.json are saved in the encoding output directory.

Report each test condition separately, and each generator against its paired
real partners. Overall aggregates mix these prespecified strata and do not
replace them. Use paired bootstrap draws of matched pairs, stratified by
paired synthetic collection, carrying both class members and every encoding
condition together. Confidence intervals are conditional on the saved
models, selected sample, and pairing; they do not include training variation.

For interpretation, distinguish an encoding effect on fixed sources from
universal generalization. A lower AUC after pairing or shared re-encoding
does not by itself isolate a particular causal artifact. A high AUC does
not rule out remaining source, style, prompt, or prior-encoding differences.
Separate extensions execute single-checkpoint DDIM reconstruction and a
small fresh-source evaluation; see extension_run.md. Neither the common JPEG
operation nor these additional samples erase prior encoding history or
establish broad independent generalization.
