# Dataset Card

## Purpose and original collection

This selected collection supports a controlled image-forensics evaluation.
It is not representative of all photographs or all generators. Historical
CIFAKE measurements are excluded and archived in outputs/cifar_pilot/.

- COCO 2017 validation: 2,000 filename-prefix photographs, with images whose
  short side is below 256 replaced by subsequent qualifying files.
- DiffusionDB: 2,000 Stable Diffusion images, 1,000 from each of the first
  two archives in archive-entry order.
- DALL·E 3: 1,000 qualifying non-thumbnail images from
  ehristoforu/dalle-3-images, in archive-entry order.
- Midjourney: 1,000 qualifying AI-labeled images from the Sharma release.
  Its real photographs are excluded; all selected Midjourney files are test only.

Ordered prefixes were used rather than collection-wide random sampling.
Synthetic files were re-saved as JPEG quality 95. All sources are then
standardized to 256×256 using the same area resize and stored as PNG before
study transforms. This does not eliminate compression history, content,
or collection artifacts. Prompts are not model inputs.

Raw data intentionally reside in the existing sibling project
data/raw/, as recorded in configs/study.yaml.
They are not bundled with this repository.

## Original counts

- Training: 3,500 — 1,400 COCO, 1,390 DiffusionDB, 710 DALL·E 3.
- Validation: 750 — 300 COCO, 301 DiffusionDB, 149 DALL·E 3.
- Test: 1,750 — 300 COCO, 309 DiffusionDB, 141 DALL·E 3, 1,000 Midjourney.

The original test has 28,000 rows: 16 images per source. Derived rows are
correlated observations, not independent source images.

## Corrected counts and screening

Native/standardized exact color-pixel matches and all-source 8×8 average-hash
pairs at Hamming distance at most two form connected components. One
representative is retained by original training, validation, test, then
source-ID priority, without reassignment.

The audit records 403 candidate links, including five exact pairs, 189
original cross-split links, and 107 cross-generator links. These counts
overlap and are not exclusion totals. A perceptual link is not proof of
visual duplication.

The screen excludes 177 and retains 5,823 sources:

- Training: 3,415 — 1,345 COCO, 1,370 DiffusionDB, 700 DALL·E 3.
- Validation: 728 — 291 COCO, 292 DiffusionDB, 145 DALL·E 3.
- Test: 1,680 — 284 COCO, 300 DiffusionDB, 134 DALL·E 3, 962 Midjourney.

The corrected test has 26,880 images. audit.json, source_audit.csv,
duplicate_pairs.csv, and manifest_sources.csv under outputs/conference_clean/
document the exclusions. No candidate link remains across retained splits
under this rule. Hash collisions can exclude distinct images. This does not
prove absence of every duplicate, prompt overlap, or collection confound.

## Licensing and disclosure

Public download access does not imply uniform redistribution permission.
COCO annotations, COCO photographs, and synthetic releases can have different
terms. Check the exact release and underlying image terms before sharing
data or derivatives. Preserve identifiers, acquisition details, and
selection order with the manifest.

No blanket license is assigned to all selected images. No claim is made
that every image lacks identifiable people, copyrighted material, or
sensitive content. This evaluation does not identify depicted people.
A public image release requires a separate review of exact release terms.

## Limitations

Real and synthetic classes can differ in content, prompts, styles, and
encoding history. Midjourney is one selected held-out collection, not every
unseen generator. Generator checkpoint/version is not recovered for every
image. Transforms are controlled operations rather than naturally edited
social-media images. Actual diffusion reconstruction is added in a separate
small October 9 extension. Filtering followed inspection
of original results, so the correction is retrospective and exploratory.
Source counts and clustered intervals do not remove collection bias.

## October 9 fresh source extension

A frozen sample adds 48 Open Images test photographs, 48 unused COCO
photographs, and 96 unused DALL·E-labeled archive entries. All use the same
area resize and OpenCV JPEG95 round trip. Fixed SHA-256 ordering precedes
availability, minimum-side, and old/new pixel/hash screening. Dataset image
IDs, original URLs, rotation, license metadata, output hashes, and exclusions
are retained in outputs/extension/fresh/. The exact original DALL·E archive
prefix has zero entry-ID overlap with the new sample. Labels still denote
release provenance; pretraining overlap and content authenticity are not
verified. See extension_run.md for inference and limits.

