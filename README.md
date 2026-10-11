# Anonymous research code snapshot

This package contains the implementation and selected aggregate evidence
for the image-watermark conference manuscript. Hosting status is recorded
separately from this immutable snapshot; the archive alone does not certify
external publication.

## Contents and scope

- `src/aiwstudy/`: watermark, transforms, frozen probes, fusion, and decisions.
- `scripts/`: collection, controlled evaluations, source bootstrap, and paper
  analysis implementations. `native_baseline.py` evaluates the released
  UniversalFakeDetect head on the same fresh sources at native resolution.
  The Phase 3 drivers add matched unmarked AI edits, an additional real-image
  collection, repeated source splits, and an enlarged watermark comparison.
  Phase 4 adds equal-slot clean-repeat/JPEG-only controls, real-collection
  transfer with independent calibration source halves, watermark false-positive
  reporting, and a separately locked replication on unused native source items.
  Phase 5 adds a balanced processing-matching factorial, source-disjoint fusion
  fitting and calibration, a supervised SD-family-exclusion control, caption-
  paired photographs/generation, and a watermark comparison at common nominal
  calibration FPR and embedding PSNR. Use completion records for status; the
  release manifest states which phase checks were required for this snapshot.
- `configs/`: portable relative input paths; edit these for locally acquired data.
- `docs/`: protocol, data scope, reproduction sequence, and primary model sources.
- `evidence/`: selected aggregate results and checkpoint/input fingerprints.
- `requirements/`: separate recorded CPU-refit, figure and GPU environments.
- `release_manifest.json`: released-file checksums and packaging checks.

Dataset images, private acquisition manifests, model checkpoints, fitted heads,
embeddings, downloaded third-party code, and private source metadata are absent.
If its mapping audit passes, `reproduction_index/` contains a separate public
source index limited to public dataset/archive IDs, pseudonymous study IDs,
numeric splits, retention flags, and image fingerprints. It contains no images,
URLs, captions, prompts, or author metadata. The package does not by itself
reproduce the paper's exact dataset or numbers. Licensed acquisition of the
original releases using that public index or archived selections, followed by
the recorded preprocessing and feature extraction, is additionally needed.
No independent end-to-end reproduction of this portable snapshot is claimed.
If the paired mapping audit passes, `reproduction_index/public_pair_index.csv`
adds 200 COCO image/caption annotation IDs and deterministic generation seeds.
It contains no caption or prompt text. These pairs are a different evaluation
scope from the previous source inventory and do not enlarge an old test set.
The anonymous manuscript template and bibliography are included. If a generated
illustration is supplied, its provenance records that no dataset images were
used; the explicit generated-asset inventory is in `release_manifest.json`.
The optional ROC figure contains only saved aggregate score curves, not source
images. Its figure-provenance file identifies the completed caption-paired
identity test: 200 pairs, 400 unmarked sources, and locked joint-head scores.

## Installation and an initial run

Use Python 3.12. Install an appropriate PyTorch/torchvision pair for the machine
and the relevant recorded environment. The CPU refit and GPU experiments used
different NumPy/scikit-learn runtimes and should be kept in separate environments.
Then install this package with `pip install --no-deps -e .`.

Place legally acquired native inputs under `data/raw/real_coco/` and
`data/raw/synthetic_native/`, and review `configs/study.yaml`. A new local run is:

    aiw scan --config configs/study.yaml
    aiw split --config configs/study.yaml
    aiw preprocess --config configs/study.yaml
    aiw derive --config configs/study.yaml
    aiw train --config configs/study.yaml --variant full
    aiw evaluate --config configs/study.yaml --variant full

This is a new run conditional on the acquired inputs, not a promise of exact
reproduction. For saved-cache corrections and separately frozen evaluations,
follow `docs/reproducibility.md` and `docs/extension_run.md` with their required
artifacts supplied. The additional controls and their different test scopes
are recorded in `docs/phase3_controls.md` and `docs/phase3_watermarks.md`.
The additional controlled diagnostics and their calibration limitations are
recorded in `docs/size_controlled_ablation.md`, `docs/domain_generalization.md`,
`docs/watermark_comparison.md`, and `docs/fresh_replication.md`.
The Phase 5 stage ordering and completion gates are in
`docs/reproducibility.md`, with method details in `docs/phase5_calibration.md`,
`docs/phase5_fair_watermarks.md`, and `docs/paired_content_control.md`.
`docs/release_export.md` distinguishes the original archive from portable
copies and explains the public field allowlists and publication status.
Official checkpoints are obtained through the documented
primary sources and checked against archived fingerprints.

## Anonymization and verification

Machine-specific author paths were replaced in copied files, and the copied
configuration now uses `data/raw/`. Original project files were preserved.
Archived original script/config hashes describe the research run before these
portable edits; `release_manifest.json` hashes the actual released snapshot.
Every included Python file was parsed for syntax. The ZIP member inventory and
all file hashes were checked. No model, data, installation, or reproduction run
was executed as a packaging test.

## Publication and rights

Before a public release, the authors must review code rights, choose a license,
and resolve redistribution permissions for source data. No license is assigned
by this snapshot. Third-party source/data/model licenses continue to apply.
Do not add identifying source metadata to an anonymous review repository.
Publication, an anonymous hosted URL, and a complete licensed benchmark release
remain separate author actions.
