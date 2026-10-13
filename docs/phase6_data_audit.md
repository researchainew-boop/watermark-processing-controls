# Phase 6 data and native-baseline audit

This audit validates frozen inputs and their release metadata. It does not
reselect sources, inspect final detector outcomes, or change a model or scoring
rule. The machine-readable report is
`outputs/phase6/data_inventory/public_index_audit.json`; its status is complete.

## Source counts, roles, and selection

The finalized inventory contains 1,800 distinct source IDs:

- Independent calibration: 400 COCO and 400 Open Images sources.
- Final real test: 200 additional COCO and 200 additional Open Images sources.
- Final AI test: 200 BigGAN, 200 ADM, and 200 VQDM sources.

All labels, roles, collection names, and shared calibration metadata match the
immutable calibration CSV. Calibration and final test IDs are disjoint. Each
source ID matches its declared seed-20261011 collection/public-ID hash. Within
each real collection, every selected calibration source precedes every selected
test source in the prescribed hash order. This validates role partitioning of
the selected eligible sources; it does not establish absence of collection bias.

The full source CSV has SHA256
`137150d57cfc625bfd99a9d47d6a2f6f2528f6bae4390f44745992206c4c55a1`.
The calibration CSV retains SHA256
`469873ead86f30def2cc416b6394f859f2a32379708d8b7b0b4bea6c6bd527a6`.
The final sampling record differs from its root-frozen original only by the
explanatory `score_separation` field; all preceding fields have identical values.

## Overlap and stored pixels

The independent metadata check repeated the public-ID, exact native/direct256
pixel-hash, and 64-bit average-hash comparisons against 11,505 preceding audit
records and earlier selected new originals. It found zero matches under the
declared rules. The minimum surviving average-hash Hamming distance is three.
Audit records can represent the same preceding source at several pixel stages;
11,505 is not an independent source count. These coarse hashes do not prove
absence of every semantic or transformed near-duplicate.

All 5,400 stored native, primary, and diagnostic views were checked independently
against file and decoded-pixel hashes. Native dimensions and the pre-JPEG coarse
hashes also match. Recreating both processing paths with OpenCV 4.10.0 gave exact
pixel equality for every source:

- Primary: native -> INTER_AREA 128 square -> INTER_LINEAR 256 square -> JPEG95.
- Diagnostic: native -> INTER_AREA 256 square -> JPEG95.

All 1,200 original real-source payloads were read again from the COCO archive or
cached official Open Images S3 file, checked against input SHA256, decoded, and
compared with the stored oriented native pixels. Open Images rotations match
the recorded counterclockwise metadata. No replacement images were acquired.

For all 600 AI sources, public IDs and row indices match the cached mapping at
the pinned `nebula/GenImage-arrow` revision
`3f4b9f921a673be09a93b335ed728cea0c6ecf33`. The five acquired Arrow shards,
totalling 2,457,486,296 bytes, match the pinned Hub-tree sizes and SHA256 hashes.
This independent audit did not decode the Arrow row labels again; the frozen
preparation driver performs the explicit image-path, AI-label, and generator-code
checks before selection. That distinction is retained in the report.

## Public index and image licenses

`public_source_index.csv` has 1,800 unique sources and SHA256
`a53ca8d491a3f90ed6698d845c01d7d0e11213a27f12cdf43ab4a711479e5055`.
Its fields match the internal source metadata, with COCO's license descriptor
replaced by the individual official annotation license URL. It exports public
IDs, roles, source provenance, dimensions, hashes, and license descriptors.
The field and value scan found no private local paths, credential-bearing URLs,
account credentials, original author/profile fields, captions, or image/pixel
payloads. Dataset-relative GenImage public IDs are public acquisition identifiers.

Every selected COCO license ID, URL, name, and original Flickr/COCO URL was
checked against `instances_val2017.json`. Of the 600 COCO sources, 198 have
NoDerivatives variants: 128 calibration and 70 final test. COCO image rights are
distinct from its annotation license. See the [official COCO image terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm).

All 600 selected Open Images entries have CC BY 2.0 license metadata. This is
recorded source metadata, rather than an independent assertion about current
copyright ownership. The [official Open Images description](https://storage.googleapis.com/openimages/web/download_v7.html)
explains its individual author, license, URL, and rotation fields. These full
acquisition records remain internal.

The cached [upstream GenImage license](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
states CC BY-NC-SA 4.0 with additional noncommercial dataset terms. Its archived
SHA256 is `31dc535c86af126e640314411211c98ddab2a4f3b2c99bf23a80e3d72e818169`.
The upstream license was acquired from `main`; its archived hash identifies the
actual snapshot, rather than claiming an upstream commit pin. The acquisition
mirror itself is commit-pinned. Its README maps validation and test aliases to
the same public test partition, so these aliases are not independent datasets.
The three primary terms/metadata pages were checked on 12 October 2026.

Release the public index and acquisition instructions without dataset pixels,
edited images, downloaded shards, or private acquisition-path CSVs. Metadata
publication does not grant unrestricted rights to redistribute the referenced
images. The audit exports no image bytes and changes no license record.

## Supplemental data receipts

The previously described `final_test_source_manifest.csv` and
`final_data_lock.json` were created as supplemental data-only artifacts from the
unchanged finalized inventory. The test CSV contains exactly the existing 1,000
test-role records, preserving their fields and order; no new selection is made.
It is an internal acquisition manifest containing local paths and must not be
published as the public index.

These auxiliary artifacts were written after independent photographic
calibration. They do not purport to be the original prospective lock.
`test_sources_locked.json` remains the separate source binding fixed before
test scoring; `final_data_lock.json` records its hash, the existing root protocol,
and the audited data artifacts. It explicitly states that no model score was
read or used when constructing the supplementary receipt.

## Native detector integration repair and calibration

The frozen native driver expected the full source hash inside the root protocol
and expected main thresholds in the superseded `fusion` directory. The actual
source binding is the separate `test_sources_locked.json`, and the root selects
`fusion_v2`. `scripts/phase6_native_bound.py` replaces only the driver's
`validate_lock` integration guard. It checks the separate source receipt,
unchanged root dependencies, exact calibration metadata, dynamic learner fit,
and both calibration/threshold completion receipts before test inference.

The original native script and root protocol remain unchanged. Original model
loading, CenterCrop224 preprocessing, score polarity, FP32 logits, threshold
formula, bootstrap, and analysis function objects are preserved. The standalone
`native_baseline/guard_locked.json` was sealed before this baseline's first score
and binds wrapper, original scripts, plan, source manifest/receipt, model assets,
and root fit. Its SHA256 is
`ba898e611a6abc7496c67712c9949fcf47aa19288f2b1c8fe8f2fab2c9b8fba3`.
Seven synthetic guard tests passed without consuming real data, models, or GPU.

Native calibration completed once on 1,600 views from the 800 independent real
sources. The unchanged strict-exceedance rank is 761. Locked logit cuts are
-0.26214913 for native and -0.7844037 for common images; both have pooled
calibration FPR 39/800 = 0.04875. Native collection rates are 0.0025 for COCO and
0.095 for Open Images; common rates are 0.035 and 0.0625. Pooled calibration
therefore does not imply identical collection-specific error rates or control
under arbitrary distribution shift. Test inference subsequently completed with
its unchanged driver and frozen cuts; the final results are summarized below.

All 200 native BigGAN test sources invoke the official CenterCrop224 padding:
their 128-square pixels leave about 67.35% of the crop padded. None of the native
ADM or VQDM sources needs padding. This is a prespecified diagnostic, and the
paired common views have no padding. Native/common differences jointly involve
geometry, resolution, and encoding; the comparison cannot isolate one cause.

## Completed native detector results

The 1,000-source identity-only test completed after primary core inference.
Native recall is 0.631667 (95% source-bootstrap interval 0.596667--0.665),
AUC 0.919242 (0.904194--0.934118), and photographic FPR 0.0375.
Common-input recall is 0.555 (0.516667--0.593375), AUC 0.89145
(0.871431--0.909505), and FPR 0.04. Both use the baseline's prespecified pooled
800-negative calibration, not the paired rule's per-collection maximum cut.

Native BigGAN/ADM/VQDM recalls are 0.825/0.36/0.71; common-input recalls are
0.745/0.365/0.555. Native COCO/Open Images test FPRs are 0.005/0.07;
common-input FPRs are 0.035/0.045. Photographs are reused across generator
breakdowns, so subgroup rows do not multiply their independent count.
Native and common polarity/cuts were not tuned on test scores.

These outcomes expose weak unmarked detection by the study probe but do not
establish a same-task superiority comparison with marked four-view fusion.
The native test uses identity inputs; its padding and calibration distinctions
must accompany interpretation. Completion/model/score fingerprints and
chronology are preserved in the native-baseline receipts.
