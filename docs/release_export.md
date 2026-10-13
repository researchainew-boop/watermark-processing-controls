# Anonymous reviewer export

The existing public repository is
[watermark-processing-controls](https://github.com/researchainew-boop/watermark-processing-controls).
Its previous publication receipt describes the previous snapshot. A new Phase 6
release is complete only when a new checked ZIP, matching manuscript receipt,
and verified public commit are recorded. Preparing an export does not claim that
its new contents have already been published.

`release/anonymous-code.zip` and `release/package_check.json` describe the current
local export. `release/publication.json` records the externally verified revision,
ZIP and manuscript fingerprints outside the package to avoid a circular hash.
Every receipt applies to its own bytes. Source, PDF, figure or evidence changes
require refreshed checks and a new package/publication receipt.

## Evidence and exclusions

The final export requires completed Phases 3, 4, 5 and 6. Earlier numerical gates
are retained. Unsuccessful gates, adverse transfer findings, collection/processing
controls, caption-paired results and equal-budget watermark comparisons remain
historical evidence; they are not renamed fresh Phase 6 tests. The new hypothesis
may fail. A final release requires truthful reporting of its actual frozen status.

Included files are study code, portable configuration, listed method/data/
reproduction/attack/related-work/review documents, anonymous manuscript source,
first-party synthetic test sources, recorded dependencies, and explicitly listed
aggregate evidence. Phase 6 includes
fusion coefficients and normalization constants in small aggregate JSON files,
selection/cut receipts, the confirmatory assessment, separately labeled attack
diagnostics, the published native-detector comparison, and the two additional
simple comparators sealed before final inference. Original and amended protocol
receipts remain distinguishable.

JSON source/image/cover/pair lists are reduced to counts and author paths are
sanitized. Aggregate CSVs have explicit file and field allowlists. Grouped attack
PSNR/SSIM is flattened to condition, marking method and eight summary statistics;
unavailable/infinite numeric values become empty cells. Receipt hashes can refer
to excluded inputs without releasing their contents.

Dataset and generated benchmark pixels, captions/prompts, private acquisition
manifests, per-image predictions, decoder outputs, feature vectors,
pretrained network checkpoint weights, caches, credentials and downloaded
third-party code are excluded. Syntax/inventory checks do not claim independent
execution. `phase6_reproducibility.md` describes required inputs and the
exact-reproduction limitation.

## Public reproduction indices

Earlier audited public-source and split indices, and the 200-pair public COCO
image/caption-ID and deterministic-seed index, remain in `reproduction_index/`.
The latter contains annotation IDs, not caption or prompt text.

The Phase 6 index is separately located at
`reproduction_index/phase6/public_source_index.csv`. Its 1,800 pseudonymous records
map 800 calibration and 1,000 final sources to public releases. Its exact 23-field
allowlist is:

```text
source_id collection generator family label role public_id public_repository
public_revision public_row original_release input_file_sha256 raw_pixel_sha256
native_height native_width output_sha256 base_pixel_sha256
diagnostic_output_sha256 diagnostic_base_pixel_sha256 primary_processing
common_bottleneck_pixels jpeg_quality License
```

The export checks unique 24-hex pseudonyms, registered collection/role quotas,
origin-label consistency, release identities, dimensions, processing constants,
all fingerprint fields, and the saved mapping audit and its input hashes. Private
image/profile/acquisition URLs and machine paths are refused. The `License` field
alone may contain the audited public Creative Commons or Flickr Commons license
URLs. This exception supplies license facts and grants no new rights.

## Figures and completed reporting

Approved Phase 6 figures are `phase6_pipeline`, `phase6_roc_tm_resnet`, and
`phase6_marked_share_tm_resnet`, each as PDF and PNG. All six output hashes must
match completed reporting provenance. Checks bind the reporting implementation,
full input/output receipts, four root/source registrations, calibration cuts,
2,000-draw evaluation and separately labeled attack report. ROC AUCs and marked-
share points must equal the saved primary evaluator. Final reporting requires
`--attack-analysis`: a primary-only intermediate report is refused. All 19
completed aggregate CSV outputs are checked.

Older `resnet_roc.pdf/png` may be requested separately and still needs its original
caption-paired saved-score provenance. Explicit project-generated illustrations
retain their original no-dataset-images provenance. These allowances do not
permit benchmark image redistribution.

## Final local checks and build

Complete all numerical stages, build the final PDF, inspect all five rendered
pages and embedded fonts, and review every claim against saved results and
failures. Only after those reviews use both operator flags:

```text
python scripts/check_phase6_submission.py --manuscript --pages 5 --references COUNT --visual-reviewed --claims-reviewed
```

Replace `COUNT` with the actual reviewed reference count. The final gate is
`outputs/phase6/submission_check.json`; Phase 5 paper counts/claims are not the
final manuscript gate. It binds current source/PDF bytes, completed evidence,
actual primary success/failure status and reporting-provenance SHA256.

A final build with the three PDF figures is:

```text
python scripts/package_anonymous_release.py --require-completed-phase3 --require-completed-phase4 --require-completed-phase5 --require-completed-phase6 --paper-asset manuscript/assets/phase6_pipeline.pdf --paper-asset manuscript/assets/phase6_roc_tm_resnet.pdf --paper-asset manuscript/assets/phase6_marked_share_tm_resnet.pdf
```

Request approved PNG counterparts or older figures with more `--paper-asset`
arguments. Approved generated illustrations use `--generated-asset` plus their
`--asset-provenance`. Before replacing the existing Phase 5 ZIP/check, the packager
copies that verified pair and manifest into a dated `release/history/phase5-.../`
directory. The previous `release/anonymous-code/` stays intact; new copies use
`release/anonymous-code-phase6/` with the compatible ZIP member prefix.
Unexpected stale members cause refusal. No recursive deletion occurs.

Portable substitutions affect copied files only. Original code/input hashes
describe the measured run; `release_manifest.json` hashes the released copies.
Export checks cover syntax, inventory, privacy fields, allowed binaries, archive
CRC and every member fingerprint. They do not run models or reproduce the study.

## Publication procedure

The author-side publisher accepts only the checked final ZIP, never the research
checkout or its Git history. It requires all phase gates, public-index/aggregate
allowlists, matching Phase 6 source/PDF and reporting receipt, five pages, embedded
fonts, actual table/figure/reference counts, and visual/claim reviews. Hypothesis
failure is accepted when reported faithfully.

```text
python scripts/publish_reviewer_release.py --owner ACCOUNT --check-only
python scripts/publish_reviewer_release.py --owner ACCOUNT
```

The second command prepares a clean local repository under
`release/publication_staging/<full-zip-sha256>/repository/`. Anonymous attribution,
disabled inherited hooks/signing/configuration, approved working-tree bytes and
matching committed blobs are checked. Receipts stay outside that tree. The public
account/repository identity still needs to suit the intended review venue; local
anonymous attribution cannot hide an identifying public account.

External publication occurs only on explicit invocation:

```text
python scripts/publish_reviewer_release.py --owner ACCOUNT --publish
```

For this procedure's previously recorded repository use `--publish --update`.
Its current remote commit must match the prior receipt; updates are normal
descendant commits without force push. The default repository name is
`watermark-processing-controls`; `--repository NAME` selects another reviewed
name. The authenticated login must match `ACCOUNT`. Only the approved staged
tree is pushed, then its public main commit is checked without authentication.
To update an existing private repository while preserving its visibility, use
`--publish --update --private-update`. This mode never creates a repository or
changes visibility/default branch. It verifies the remote main commit using the
authenticated account and records private visibility separately from public
reviewer access. The author chose this private update mode for the final Phase 6
snapshot; the repository link alone does not give reviewers access.
Interrupted operations retain an attempt receipt without claiming success.
Credentials never enter command arguments or the package.

This export assigns no new code/image license. Third-party dataset/model/code
terms continue to apply. Public code access, exact independent reproduction and
a licensed full benchmark release are separate outcomes.

## Study-owned decision parameters and current illustration

The final Phase 6 snapshot includes two 2,842-byte study-owned linear heads,
encoded in `decision_parameters/passive_heads.json`. A separate schema checker
requires only float32 coefficients of shape (1,512), a scalar float32 intercept,
and classes [0,1]; original bytes must match the sealed protocol. The publisher
independently repeats this check. No pretrained network, training feature or
source metadata is exported in this exception. The replay helper restores the
exact NPZ bytes without refitting, and refuses to replace different local heads.

The new `phase6_demo_strip.pdf/png` reuses the earlier project-generated
illustration and its measured decoded bits without new inference. Its separate
provenance binds the old image/measurement records and every original/derived
illustration image. Include these using `--generated-asset` and
`--asset-provenance manuscript/assets/phase6_demo_provenance.json`.
The illustration's classical strength12/diffusion strength0.50 setting is excluded
from every benchmark and differs from the primary Phase 6 settings.

Calibration-manifest byte restoration and interrupted uncommitted attack files
are disclosed in `phase6_runtime_environment_repair.md`. Original receipts,
scientific drivers, registered cuts and already committed inputs remain unchanged.
