# Phase 6 reproduction and evidence map

Phase 6 evaluates a development-fitted bounded correction of a frozen passive
logit using decoder agreement, margin, received-view stability and complete
message/ECC features. The registered primary system is TrustMark/ResNet. Its
fixed comparator is the strongest eligible origin/fusion rule chosen on tuning
sources. Its positive claim requires all declared gain, unmarked-harm and
photographic-source-FPR criteria. A completed run is preserved even if the claim
fails. `phase6_method.md` gives the objective, weighting and statistical procedure.

## Release contents and required local inputs

The reviewer snapshot contains code, protocol and implementation receipts,
small fusion coefficient/normalization JSONs, frozen selection/cut rules,
aggregate outcomes and figure provenance. Its public index contains public
photographic IDs, pinned GenImage Arrow rows, pseudonymous IDs, roles, dimensions,
image/pixel fingerprints, processing and license facts. The audited 1,800 sources
comprise 800 independent calibration negatives and 1,000 final sources. Earlier
indices and the 200 caption-paired control remain separate.

No pixels, prompts, per-image scores, features, private manifests, pretrained
networks or downloaded third-party code are released. The two study-owned linear
passive heads are included in `decision_parameters/passive_heads.json`. Their
original NPZ bytes are encoded with checksums, and the schema contains only 512
coefficients, one intercept and binary class values per head. A narrow independent
reader validates field names, shapes, dtypes, finite values, classes and byte
fingerprints. This adds no training features or source metadata.

Run `python scripts/materialize_passive_heads.py --check-only` to validate those
assets; omit the flag to restore the original head input files. Full extraction
still requires licensed images and legally acquired external networks/decoders.
The public index enables mapping/reacquisition but does not provide all original
development inputs. Training new heads is a new experiment, not an exact replay.
No independent end-to-end reproduction of this portable snapshot has been run.

Use Python 3.12 and separate the recorded CPU/GPU environments in `requirements/`.
CUDA 12.1 wheels and versions describe the study environment, not universal
hardware requirements. Obtain external models/code from primary sources in
`phase6_attacks.md` and `phase6_native_baseline.md`, respecting their terms and
checking pinned revisions/fingerprints. The passive dependency receipt identifies
ResNet IMAGENET1K_V1 and the actual CLIP backbone bytes.

First-party checks are in `scripts/test_phase6_fusion.py` and `tests/`. They
construct synthetic fixtures locally; no test image/data fixtures are exported.
The recorded verification comprises 11 fusion checks, 12 feature/attack/native
checks, 13 secondary-attack guard checks, 7 native-wrapper checks and 4 additional-
baseline arithmetic checks: 47 distinct checks. This verifies those implementations
and guards, not a full model/data reproduction or conference acceptance.

With the recorded scientific dependencies and `requirements/recorded-verification.txt`
installed, run the synthetic checks separately from any inference:

```text
python scripts/test_phase6_fusion.py
python -m pytest -q tests/test_phase6_features.py tests/test_phase6_attacks.py tests/test_phase6_native_baseline.py tests/test_phase6_attack_reporting.py tests/test_phase6_native_bound.py tests/test_phase6_additional_baselines.py
```

The original verification disabled automatic third-party pytest plugin loading
and used a dedicated workspace temporary directory. No fresh test outcome is
needed by these fixtures. Review runtime/path adaptations before executing a
portable copy.

The research archive uses `outputs/phase6/`; selected aggregate evidence is in
`evidence/phase6/` in the release. Hashes of excluded inputs remain identifiers.
Portable code has different hashes after path edits, recorded in
`release_manifest.json`; do not represent it as unchanged execution code.

## Source and prediction separation

Sources follow registered hash ordering and eligibility/duplicate screening,
never detector scores. Each edit/payload retains its source role. Screening uses
known public IDs, exact native/processed pixel hashes and conservative average-
hash distance. See `phase6_data.md` for releases, rotation and remaining confounds.

Development contains 936 sources split within collection into fitting, tuning
and constraint roles. Transformations use fitting rows only. Tuning selects the
model; constraint assessment cannot choose a replacement. Final calibration has
400 fresh COCO and 400 fresh Open Images sources. Final test has 200 sources
from each photographic collection and 200 each from GenImage BigGAN, ADM and
VQDM. These AI families never enter this study's supervised development. Backbone
or external-detector pretraining exposure is not excluded.

Every primary source uses a common 128-pixel bottleneck, restoration to256 and
JPEG95 before embedding. Native-to256 processing is a separate diagnostic.
Original content, compression and acquisition are not made identical. GenImage
and photo collections are not caption paired; retain the earlier paired-content
control and its limitations.

Prediction uses the received image, frozen passive score and decoder outputs
from that image and a fixed JPEG90 view. All three registered queries apply to
every input. Source ID, label, collection, attack, true payload, sibling edit and
clean reference cannot become prediction features. Such metadata serve fitting
weights, calibration, integrity and evaluation only.

## Execution order in the research archive

This sequence explains the measured dependencies. Reacquiring data or changing
portable code requires a new recorded run with its own identities/receipts;
never alter a completed archive to make its old hashes pass.

1. Use `phase6_prepare_sources.py` for score-free source selection. Prepare
   development features and external detector/attack assets/plans. Fit with
   `phase6_fusion.py --output outputs/phase6/fusion_v2 fit --features outputs/phase6/development/features.csv`.
2. Freeze dependencies with `phase6_freeze.py --stage method` and source IDs/
   quotas with `--stage test-sources`. Preserve the original protocol, prescore
   amendments, original v1 fit and corrected v2 fit; do not silently substitute.
3. Prepare/derive calibration core covers with `phase6_runtime.py` and extract
   their features with `phase6_features.py`. Calibrate the learner on
   `outputs/phase6/calibration/features.csv`. Each score gets a strict source-max
   cut across all three queries/four core views, conservative for each represented
   photographic collection.
4. Seal `phase6_confirm.py --stage seal`. Independently calibrate the published
   detector and lock the supplemental comparators with
   `phase6_additional_baselines.py --stage calibrate` before first final inference.
   Their bound selection uses tuning only and their own cuts use the same 800
   negatives; they cannot replace the primary comparator.
5. Prepare the fixed extra-attack calibration subset and all nine conditions,
   then extract features. Lock separate cuts with
   `phase6_attack_reporting.py --stage calibrate` before secondary test inference.
   These never alter primary four-view cuts.
6. Derive final core covers, extract features once with the registered driver and
   use `phase6_confirm.py --stage evaluate`. This binds exact registered sources/
   metadata, feature/manifest/driver/root hashes and 2,000 source draws. Finish
   native/common published-detector analysis, additional comparator evaluation,
   fixed-subset extra attacks and their separately labeled analysis.
7. `phase6_reporting.py --attack-analysis outputs/phase6/attack_analysis` consumes
   completed saved scores only, creating tables/figures without inference, fitting
   or new cuts. `check_phase6_submission.py` checks evidence and saved decision
   arithmetic. Final visual/claim reviews and export steps are in `release_export.md`.

For the original native-detector run, invoke `phase6_native_bound.py` for its
prepare/calibration/calibrate/test/analyze stages. Its separately sealed
`guard_locked.json` repairs the stale root/source integration check without
changing the original driver's preprocessing, scores, cuts or statistics. The
wrapper and repair chronology are documented in `phase6_data_audit.md`.

Completed stages verify receipts before reuse. Changed inputs/code/cuts cause
refusal or require a separate experiment. Original hashes, full output inventories
and start/lock timestamps record the history; timestamps alone are not a
statistical guarantee.

## Statistical and claim limits

The 2,000 draws resample source clusters within collection, keeping edits,
payloads and seeds together. All systems share paired draws. The final mixture
contains 600 AI and 400 photographic sources; additional covers/query rows do not
increase independent sample size. Primary gain and unmarked-loss procedures are
registered. Subgroups, attacks, AUC and marked-share analyses are secondary.

Nominal source FPR assumes independent/exchangeable sources within the calibrated
photographic collection and a fixed score/query/view family. It is marginal,
not a universal shifted-domain or realized-split guarantee. Core-cut extra-attack
diagnostics are outside the family; secondary nine-condition cuts cover only
their stated family. Reconstructed photographs have input-source provenance
labels and contain generated pixels. Shared PSNR is not matched semantic quality;
residual-budget variants are adaptations of published attacks.

Zero harmful switches make bootstrap loss intervals degenerate. The independent-
source event bound is reported separately; pooled support does not certify a 1%
margin in every 200-source family. Nonzero-harm bootstrap claims remain approximate
unless conservative independent group bounds also pass. Full native message
recovery authenticates neither issuer nor photographic/AI origin. Three fixed
payloads are not an open-set authentication deployment.

Retrospective method exploration, earlier failed gates and adverse controls stay
visible. Simple union/equal-bound confidence comparators are secondary additions
after calibration but before final test scores, not original primary comparisons.
Their outcomes must constrain novelty claims. Neither experiment completion nor
a clean reviewer export guarantees conference acceptance.

The complete-message union uses its locked sigmoid passive-score mapping.
Extreme logits can produce numerical probability ties, making its strict cut
conservative. Preserve that arithmetic and disclose ties; a tied comparator's
failure alone is insufficient evidence of a new fusion advantage.
