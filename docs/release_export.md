# Anonymous reviewer export

The clean reviewer snapshot is publicly hosted at
[watermark-processing-controls](https://github.com/researchainew-boop/watermark-processing-controls).
Its public `main` commit was verified without authentication. The local
`release/publication.json` records the verified repository/commit URLs and
published ZIP, manifest and source/PDF fingerprints; this receipt stays
outside the clean package to avoid a circular hash dependency.
`release/anonymous-code.zip` is the local export, and
`release/package_check.json` records its fingerprint, member count and
publication status. Hosting supplies reviewer access; independent
reproduction and benchmark-image redistribution remain separate.
Source/PDF changes require refreshed checks, packaging and a published
snapshot update. The immutable manifest describes only its own snapshot.

## Included evidence and public identifiers

The packager copies study source code, portable configuration, explicitly
listed method/reproduction documents, the anonymous manuscript source and
bibliography, recorded dependency versions and selected aggregate evidence.
Every Phase 5 aggregate CSV has an exact column allowlist. Aggregate JSONs
have an exact file allowlist; machine-specific paths are sanitized and
source/image/cover/pair identifier lists are replaced by counts. Input and
output file hashes can refer to excluded research files without including
their contents.

The audited public reproduction index is a separate exception to the
aggregate-only rule. `public_source_index.csv` identifies public archive
members, parquet rows or photographic IDs with pseudonymous study identifiers,
retention/split flags and fingerprints. Original and five-seed split CSVs
share the audited identifier pool. The index is copied only after its saved
mapping checks and file hashes pass the explicit field validation.

`public_pair_index.csv` has exactly 200 records. Its only fields are
`pair_id`, `real_source_id`, `ai_source_id`, `public_coco_image_id`,
`archive_filename`, `archive_entry`, `caption_annotation_id`,
`generation_seed`, `split`, `native_file_sha256`,
`real_standardized_pixel_sha256`, and `real_standardized_file_sha256`.
The packager checks deterministic pseudonyms/seeds, unique test-only pairs,
relative COCO archive entries, prior-image nonoverlap and the audit/protocol/
selection hashes. Caption annotation IDs identify public COCO annotations;
caption and prompt text are absent. Pairing metadata enables reacquisition
subject to dataset terms and model availability.

Dataset images, generated benchmark images, private `source_pairs.csv`,
captions/prompts, acquisition URLs, source predictions, embeddings, model
checkpoints, fitted heads, caches and third-party downloaded code are excluded.
The project-generated figure illustration and aggregate ROC figure can be
supplied explicitly with their provenance; they do not permit copying
benchmark images. The initial fair-watermark driver snapshot is an internal
archive and is excluded.

The approved ROC depicts the completed caption-paired identity test, with
200 pairs and 400 unmarked sources. Its saved-score, completion, head and
output-file fingerprints must pass before packaging. The earlier expanded-
test ROC remains only in the local manuscript archive. Aggregate-only
postgeneration overlap and sensitivity reports may be released: all 200
locked primary pairs remain in the primary analysis, while the 197-pair
coarse-hash sensitivity is explicitly post hoc. Flagged identifiers and
per-image predictions are excluded.

## Final build and checks

Require all requested phases, then supply the approved figure assets:

    python scripts/package_anonymous_release.py --require-completed-phase3 --require-completed-phase4 --require-completed-phase5 --generated-asset manuscript/assets/generated-original.png --generated-asset manuscript/assets/pipeline_examples.pdf --asset-provenance manuscript/assets/generated-asset-provenance.json --paper-asset manuscript/assets/resnet_roc.pdf

The Phase 5 flag checks completed matching, independent calibration,
supervised SD-family-exclusion, paired-content and fair-watermark artifacts
through `check_phase5()`. An unfinished stage refuses a final build. The flag
also requires the audited public pair index. It does not substitute for
compiling and inspecting the final manuscript PDF.

Copied paths are made portable. Research archives remain unchanged. Thus
original input/script hashes describe the measured run; the released-file
manifest describes the portable copies. The package checks copied Python
syntax, its exact inventory, identifying paths, allowed binary assets, ZIP
CRC and every member hash. These are export checks, not new model/data tests.
Unexpected stale export files cause an error instead of recursive deletion.

## Reviewer access and publication

The repository link above was verified after successful publication;
`release/publication.json` records the published revision. Future updates
must pass the same unauthenticated public-commit check. Account ownership,
repository visibility and intended reviewer access must agree. For anonymous
review, inspect the account/repository name, commit metadata, README and
downloaded snapshot for identifying information as part of choosing a suitable
host. A public account can reveal author identity even if code paths are clean.

This export assigns no new license. Authors may publish their permitted
original code while third-party dataset/model/code terms continue to apply;
image redistribution and a full benchmark release require their own rights
review. A successful upload must record the commit/release URL and the ZIP
hash, then update the manuscript only with a working reviewer-access link.

## Local GitHub staging procedure

The author-side `scripts/publish_reviewer_release.py` uses only the finished
`release/anonymous-code.zip`. It never copies the project checkout or its Git
history. This deployment helper and `release/publication.json` are excluded
from the research snapshot; they are local author tools, not prerequisites
for running the released study code.

Before staging, complete the numerical audit and final manuscript review:

    python scripts/check_phase5_submission.py --component all --manuscript --pages 5 --references 24 --visual-reviewed

Use `--visual-reviewed` only after inspecting every final rendered PDF page.
Then rebuild the ZIP with all three completion flags and the approved assets
shown above. Staging requires its verified member hashes and exact inventory,
completed phase records, public-index allowlists, and the matching five-page,
24-reference, six-table, three-figure submission check. Source/PDF changes
after those checks require new checks and a refreshed ZIP.

Replace `ACCOUNT` with the intended GitHub owner. These commands remain local:

    python scripts/publish_reviewer_release.py --owner ACCOUNT --check-only
    python scripts/publish_reviewer_release.py --owner ACCOUNT

The second command extracts the audited ZIP to
`release/publication_staging/<full-zip-sha256>/repository/` and creates a new
local Git commit. Commit author and committer are both
`Anonymous Research Authors <anonymous@example.invalid>`. Hooks, signing,
global attributes/configuration, inherited Git checkout paths and tracing
are disabled for this staging operation. The working-tree bytes and committed
Git blobs must both match every approved ZIP member. `.tools`, private data,
predictions, embeddings and model files cannot be added. Preparation receipts
are stored alongside the staging repository, never inside it.

Only this explicitly invoked command permits external repository creation
and pushing; it must be run after reviewing the prepared result:

    python scripts/publish_reviewer_release.py --owner ACCOUNT --publish

The default repository name is `watermark-processing-controls`; specify
`--repository NAME` to use another reviewed name. The authenticated GitHub
login must match `--owner`. The procedure refuses an existing repository
without its own matching publication/attempt receipt, creates a public empty
repository, pushes only the clean commit, and verifies the public main commit
through an unauthenticated GitHub API request. Its default branch is checked
and set to `main` if necessary so the repository link opens the released code.
No passwords or tokens are
command arguments or publication-receipt fields. The existing local GitHub
CLI authentication supplies credentials to GitHub.

Successful publication writes `release/publication.json`, updates the local
package report, and records the working repository/commit URL, ZIP/manifest
hashes, exact commit attribution and the submission-check/source/PDF hashes.
An interrupted external operation leaves `publication_attempt.json` without
claiming verified publication; retrying the same reviewed command can resume
that recorded attempt. No existing external repository is deleted.

To publish a revised clean ZIP to this procedure's previously recorded
repository, refresh its final checks/package and invoke `--publish --update`.
The remote main commit must still match the prior receipt. Only that public
release history is fetched; the new approved tree becomes a normal descendant
commit, with no force push. Changes by another contributor cause a refusal
for inspection. Publication metadata stays outside the clean snapshot to
avoid a receipt/snapshot hash dependency. Adding the verified URL to the
paper changes the source/PDF hashes and therefore requires rechecking before
a subsequent code-snapshot update.
