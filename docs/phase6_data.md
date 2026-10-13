# Phase 6 source preparation

This document describes source selection, not scientific results. The preparation
driver imports no detector, watermark decoder, or learned model and computes no
test scores. Its outputs are under `outputs/phase6/data_inventory/`; source images
and the isolated Arrow reader are under `data/phase6/`.

## Fixed source quotas and roles

The sampling record precedes method calibration and final inference. The seed is
20261011. Within each collection, candidates are ordered by
`SHA256(phase6|seed|collection|public ID)`. Only availability, decode failure,
dimensions, metadata rotation, and predeclared duplicate rules affect selection.
No detector score, watermark score, subjective visual difficulty, or scientific
outcome affects selection or replacement.

- Calibration: 400 fresh COCO photographs and 400 fresh Open Images photographs.
- Final test: 200 additional COCO photographs and 200 additional Open Images
  photographs, disjoint from calibration.
- Final AI test: 200 GenImage BigGAN, 200 GenImage ADM, and 200 GenImage VQDM
  images, disjoint from every preceding study source.

For each real collection, the first 400 eligible ordered IDs are assigned to
calibration and the next 200 to final test. Each original and every derived edit
must keep the same source role. The three new AI collections may not enter method
training, hyperparameter selection, calibration, or development inference. Final
test inference requires the separately frozen method/evaluation protocol first.
Preparing a source without model scores is compatible with this separation.

The archived public GenImage validation partition is the benchmark test
partition. Its mirror's `validation` alias is the same data, not an independent
validation set. We use only AI-labeled rows from that partition.

## Families and provenance

BigGAN is a GAN family; ADM is class-guided pixel-space diffusion; VQDM is
diffusion over discrete quantized latent codes. These are different generative
mechanisms, not three Stable Diffusion checkpoint names. They were absent from
the supervised study probe/fusion development, which used Stable Diffusion and
DALL-E3. The preceding Midjourney and Realistic Vision/LCM images have already
been examined and do not qualify as fresh final evidence.

The public acquisition mirror is
[nebula/GenImage-arrow](https://huggingface.co/datasets/nebula/GenImage-arrow),
pinned to commit `3f4b9f921a673be09a93b335ed728cea0c6ecf33`. It retains original
image paths and binary source labels. Mapping row IDs are checked against the
actual Arrow row's image path and AI label before selection. Downloaded shard
sizes and SHA256 hashes are checked against the commit-pinned Hub tree. The
authoritative dataset and generator descriptions are the
[GenImage paper](https://arxiv.org/abs/2306.08571) and
[official repository](https://github.com/GenImage-Dataset/GenImage).

Unseen means absent from study supervised fitting and development. It does not
exclude exposure in ImageNet/CLIP pretraining or in an external published
detector's original training. This qualification must accompany any claim of
generalization.

## Exclusion audit

The preselection audit binds the SHA256 of nine preceding source tables:
original conference sources, the first fresh sample, added Open Images,
Phase 4 fresh replication, Phase 5 preceding-source exclusions, paired real
sources, generated paired sources, fair-watermark source inputs, and independent
calibration additions. The original audit contains 10,457 records. A prescore
supplement includes 1,000 available superseded processed originals and 48 demo
originals, for 11,505 audit records before any new model inference. The paired
real-source metadata namespace is also made explicit as COCO. Multiple records
can describe the same original at different pixel stages; this count is not an
independent sample count.

Candidates are excluded by known collection/public ID, exact decoded native or
standardized pixel hashes, or 64-bit 8-by-8 grayscale average-hash Hamming distance
at most 2 against every preceding and earlier selected source. For the Phase 5
generated pairs, native and standardized exact pixel hashes and native coarse
hashes are freshly computed without a learned model. Coarse-hash matching is a
conservative screening rule, not a proof that the images are duplicates. All
screened rejections are recorded. No candidates are rejected after seeing final
scientific results.

## Common processing and remaining confounds

Decode to OpenCV BGR pixels and apply Open Images' metadata counterclockwise
rotation. A prescore resolution amendment preserves every selected ID and role
and defines the primary view as `INTER_AREA` downsampling to 128 by 128, then
`INTER_LINEAR` upsampling to 256 by 256, followed by common JPEG95 encoding and
decoding. Store decoded pixels as PNG. This common bottleneck prevents only the
BigGAN subset from being upsampled in the primary comparison. The secondary
diagnostic view resizes native pixels directly to 256 square with `INTER_AREA`
before the same JPEG95 round trip. These steps occur before watermark embedding.
Preserve decoded native pixels, native dimensions, and original input hashes.
The fixed eligibility floor is native minimum side 128. Duplicate selection
screening retains its predeclared native/direct256 pixel stages; changing the
primary display/embedding view does not resample selected originals.

Common re-encoding controls the final file codec. It cannot equalize original
resolution, semantic content, collection procedures, or upstream compression.
The GenImage class-conditioned images are not generated from the same COCO
captions as the photographs. Consequently this transfer test complements the
earlier caption-paired control; it does not replace a truly content-paired test.
Native dimensions are explicitly retained in the manifest. Attack and watermark
comparisons must use paired views of each selected source, and interpret any
cross-collection origin score with these confounds in view.

The negative label is the established COCO/Open Images source-collection label,
not an individual forensic certificate of camera capture. Open Images includes
Flickr image content with varied visual styles; no postselection detector-based
or subjective photographic-quality filtering is used. Describe these as real
source-collection images rather than claiming every item is independently
verified camera photography.

## Licenses and release restrictions

The [upstream GenImage license](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
is authoritative: CC BY-NC-SA 4.0 plus additional dataset terms restrict use to
noncommercial purposes. A mirror's metadata is not permission to override those
terms. The upstream license and mirror readme are archived with this sample.
Open Images provides individual image license/author/original URL fields; these
are retained per source, following the
[official metadata description](https://storage.googleapis.com/openimages/web/download_v7.html).
COCO images have individual original Flickr licenses, following
[COCO's image terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm),
distinct from the annotation license. The cached official
`instances_val2017.json` metadata is joined to selected public IDs in
`coco_image_licenses.csv`, with individual license URL and original Flickr URL.
This annotation does not alter source eligibility or roles. Image licenses must
be consulted before any redistribution; in particular, derivative publication
is not automatically allowed for an image whose license prohibits derivatives.

The public research-code release should include source IDs, protocol, hashes,
and acquisition instructions, not these image bytes, private captions, or large
public dataset shards. Public accessibility alone is not evidence of unrestricted
reuse. Downstream users must follow the original dataset and per-image terms.

## Precision limits

At a true false-positive rate of 5%, 400 independent final real sources give an
approximate standard error of 1.09 percentage points; each 200-source real
collection gives about 1.54 points. For 200 independent AI sources in a family,
worst-case TPR standard error is about 3.54 points. Confidence intervals and paired
source bootstrap comparisons must cluster all edits and payloads from the same
original. Derivative images do not increase the independent sample size.

Zero events among 200 independent sources has a two-sided exact 95% binomial
upper confidence bound of approximately 1.83%. The 800-real calibration sample
has empirical conformal quantile resolution 1/801 pooled, or 1/401 within each
real collection. Calibration claims remain conditional on the specified
exchangeability assumptions; these counts do not establish robustness to arbitrary
distribution shifts. A one-percentage-point noninferiority margin may be too
narrow to confirm with this sample, even when the observed difference is small.

## Reproduction and artifacts

Run `scripts/phase6_prepare_sources.py --stage all` using the project's bundled
Python 3.12 runtime. The driver uses existing image-processing dependencies and
an isolated PyArrow 19.0.1 reader in `data/phase6/runtime-deps/`.

Important artifacts:

- `sampling_protocol.json`: fixed seed, quotas, exclusions, preprocessing, roles.
- `preceding_source_exclusions.csv`: exact/coarse hashes and public-ID provenance.
- `sources.partial.csv`: acquisition progress; not a finalized test manifest.
- `selection_rejections.csv`: availability/content-hash exclusions only.
- `primary_source_manifest.csv`: all calibration/test sources, identities,
  native/common-JPEG95 paths, hashes, labels, roles, and public provenance.
- `selection_complete.json`: completion and manifest hash; explicit absence of
  detector/watermark inference.
- `source_overlap_audit.json`: complete preceding and within-new public-ID,
  exact-pixel, and coarse-hash exclusion checks.
- `coco_image_licenses.csv`: per-image license and original URL from official
  COCO metadata.

`--stage ai-assets` downloads/validates the public Arrow inputs without reading
or changing source-selection manifests and may run during real acquisition.
`--stage common128` applies the prescore resolution amendment to fixed selected
native copies and updates the final primary source manifest. The manifest's
`base_path`/`path` point to the primary common128/JPEG95 view;
`diagnostic_base_path` points to the direct256/JPEG95 view.

Expected public Arrow transfer is about 2.46 GB, plus approximately a few hundred
MB of photographs and native/base image storage. Preparation uses CPU only; no
generator or GPU inference is needed. Actual byte totals and completion time are
recorded when all fixed quotas have been acquired.

## Completed source lock

The fixed 1,800-source inventory completed on 11 October 2026 at 13:15 UTC:
800 independent real calibration sources and 1,000 final test sources
(400 real and 600 AI, with exactly 200 per new generator). All native, primary,
and diagnostic image bytes were checked against their manifest hashes. Final
duplicate screening found zero public-ID matches, zero exact native/direct256
pixel matches, and zero average-hash matches at Hamming distance at most 2,
against the 11,505 preceding records and earlier selected new originals.
The minimum surviving average-hash distance is 3. This is the specified audit,
not a guarantee against every possible near-duplicate transformation.

The immutable calibration CSV SHA256 is
`469873ead86f30def2cc416b6394f859f2a32379708d8b7b0b4bea6c6bd527a6`.
The full primary manifest SHA256 is
`137150d57cfc625bfd99a9d47d6a2f6f2528f6bae4390f44745992206c4c55a1`.
The final sampling protocol SHA256 is
`49e9b913da254e317b4db78dec7855433f6ae9cafa7475bf1ff61de7b6774008`.
The root method freeze preceded completion and bound sampling-protocol SHA256
`b2a55e6df0ce3ee395f51dbe37b91187feeb54f6fa89c72207a066df16402ff2`.
Its exact bytes are preserved as `sampling_protocol.root_frozen_original.json`.
The sole later protocol change is addition of the explanatory `score_separation`
field, clarifying that independent calibration may be scored after delivery of
its immutable manifest. Seed, quotas, source roles, eligibility, selection order,
exclusions, and primary/diagnostic processing are identical. No source-preparation
or final-test model score informed this addition.

`public_source_index.csv` contains 1,800 distinct public source IDs, roles,
provenance, image dimensions, hashes, and license fields. It contains no local
paths or account credentials. All 600 selected Open Images items have CC BY 2.0
metadata. The 600 COCO items include 198 with NoDerivatives license variants;
publish the acquisition instructions and IDs, not edited image bytes. GenImage
items retain the upstream noncommercial and ShareAlike terms.

Verified Arrow inputs total 2,457,486,296 bytes. The public-assets cache occupies
5,101,098,589 bytes because successful checked byte-range downloads retain their
recovery chunks as well as joined files. Native plus primary and diagnostic PNG
storage occupies 1,453,271,175 bytes. Recovery chunks have not been deleted.
`final_data_lock.json` binds the final inventory and audit artifacts, and
`final_test_source_manifest.csv` contains only the 1,000 fixed final test originals.
Final test scoring still requires the parent method/evaluation lock and its
explicit implementation/dependency amendment; this preparation performed none.
