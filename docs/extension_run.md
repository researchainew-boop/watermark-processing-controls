# October 9 extension of the corrected conference experiment

The core remains the retrospective 5,823-source correction. Extension results
are separate; the 192-source fresh sample is selected before its first
inference. No fresh threshold or model is refitted in response to its results.

## Saved status and reproduction order

The common-encoding evaluation, fresh sample, published-detector comparison,
watermark attacks, and diffusion reconstruction have saved final artifacts.
The diffusion completion marker records 288 outputs; the watermark summary
includes both diffusion strengths and VAE-only among its sixteen conditions,
with 2,784 transformed outputs and 5,568 detector measurement rows.
The source verification interval artifact has 64 completed exact intervals.
The corrected attack-prevalence analysis has completed on 1,152 selected
rows; its 1,776 inference rows remain archived. The final prevalence.json
records selected_counterfactual_rows = 1152. Predictions alone are not a
final prevalence result; use the corrected final analysis artifact.
Partial measurement files are resumable progress, not final evidence.

Use Python 3.12 with the versions recorded in each provenance file. The
extension runtime uses NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.6.1,
torch 2.5.1+cu121, torchvision 0.20.1+cu121, OpenCLIP 2.26.1,
TrustMark 0.9.2, Diffusers 0.35.1, Transformers 4.46.3, and
huggingface-hub 0.34.4. The original correction's runtime is kept separate.
Saved coefficients are evaluated directly; changing the inference runtime
does not refit them.

1. `scripts/hard_gate_analysis.py` implements the literal 13-of-16 fallback,
   source bootstrap, and eleven clean marking shares. Outputs:
   `outputs/extension/hard_gate/`. The old test was already examined.
2. `scripts/wider_margin_analysis.py` selects margin 0.48 from 0.02–0.49,
   using corrected seen validation only. The gate and policy are frozen
   before fresh inference.
3. `scripts/evaluate_encoding_controls.py` scores the previously locked
   600-source/300-pair controls. It reuses 600 common-PNG vectors per
   backbone only after seven anchor checks and extracts 1,800 JPEG vectors.
   Both origins receive JPEG 95/75/50. This equalizes the current encoding
   operation, not earlier encoding history or native scenes.
4. `scripts/download_extension_assets.py` records pinned official detector
   code, head, backbone, and Open Images metadata. The assets' provenance
   records hashes. Standard curl transfers were used where Python's
   transfer client stalled; the resulting official hashes were verified.
5. `scripts/prepare_fresh_sample.py` freezes 48 Open Images, 48 unused COCO,
   and 96 unused DALL·E-labeled archive sources, screens against all 6,000 old
   sources and within the new sample, and gives every source identical
   resizing/JPEG95. Availability and screening exclusions are archived.
   `scripts/audit_fresh_archive.py` reconstructs the exact original 1,000
   qualifying DALL·E archive entries and verifies zero identity overlap.
6. `scripts/evaluate_frozen_extension.py` scores the fresh manifest once.
   It refuses a second pass when predictions exist. Outputs:
   `outputs/extension/fresh/`. The sample is unmarked, with one known AI
   archive; it does not establish unseen-generator generalization.
7. `scripts/evaluate_published_baseline.py` uses official UniversalFakeDetect
   ViT-L/14 code and weights, raw features, CenterCrop224 (no resize), and
   CLIP normalization. The official OpenAI backbone has its expected
   SHA-256. Logit zero is the published fixed decision point. Evaluation
   covers 1,680 clean core images, 1,440 test encoding variants, and 192
   fresh images. Training data differ from the collection-specific probes.
8. `scripts/benchmark_watermarks.py --stage prepare --device cpu` publishes
   192 covers from 24 matched pairs. Three balanced payloads are embedded
   under each watermark, with unmarked controls. Official TrustMark Q
   encoder/decoder/config MD5 and SHA-256 are recorded. BCH5 has 61 data
   bits: the 16-bit string plus 45 zeros. Verification requires the entire
   expected 61-bit message, schema, and ECC validity. Classical verification
   instead accepts >=13/16 agreement. Both use three balanced 16-bit payloads
   for local attacks; diffusion uses only the original default payload and
   unmarked controls. These unequal criteria and payload capacities must be
   retained when comparing recovery rates.
9. `scripts/benchmark_watermarks.py --stage attacks` publishes 2,496 outputs:
   ten fixed conditions and three new sigma-6 noise seeds per cover.
10. `scripts/run_diffusion_extension.py` uses the cached pinned Realistic
    Vision V5.1 SD1.5-family model, actual DDIM denoising at strengths
    0.25/0.50, and a distinct posterior-mean VAE control. There are 96
    default-payload/unmarked covers and 288 outputs. All checkpoint hashes,
    scheduler, prompt, seed, precision, step count, and timings are saved.
    The fp32 VAE's boundary casts avoid mixed-precision errors while the
    UNet/text encoder remain fp16. Resume verifies completed file hashes.
    The completed outputs are archived in diffusion/manifest.csv and
    diffusion/complete.json; diffusion/protocol.json holds the fixed settings
    and cached component hashes.
11. `scripts/benchmark_watermarks.py --stage analyze --device cpu
    --diffusion-manifest outputs/extension/diffusion/manifest.csv` decodes
    both marks and records quality. Existing complete measurements are
    reused, with only new conditions decoded. The final measurements.csv,
    summary.json, and analysis_provenance.json cover all 2,784 outputs.
    Source bootstrap retains
    payload repeats. For one-output-per-source conditions, exact binomial
    intervals are also reported; bootstrap intervals can degenerate at
    all-success/all-failure boundaries.
12. `python scripts/source_verification_intervals.py` reads the final
    watermark measurements without inference or threshold tuning. It saves
    64 two-sided 95% exact Clopper–Pearson intervals in
    watermarks/source_verification_intervals.json: sixteen conditions,
    two decoders, and two input-origin labels. The binary event is that
    all evaluated payloads verify for one source. Local marked conditions
    have three payloads per source; diffusion marked conditions and real
    unmarked controls have one. This target differs from the mean per-output
    verification rate. Each interval uses 24 distinct sources, conditionally
    treated as independent with a common success probability; the selected
    collection-balanced sample is not a population probability sample.
13. `scripts/attack_prevalence.py --stage prepare` and frozen feature
    evaluation created 1,776 archived prediction rows. The corrected
    `python scripts/attack_prevalence.py --stage analyze` selects 1,152 rows
    without repeating inference: 72 per condition across sixteen conditions,
    comprising 24 unmarked real controls and marked/unmarked counterfactuals
    for each of 24 AI sources. Only the original expected payload
    1011010010110100 is used, matching the expected string used to score the
    unmarked controls. The other two payloads remain in the separate recovery
    benchmark. This selection correction follows inference; models and
    thresholds are unchanged. Source-clustered, collection-stratified
    uncertainty accompanies all eleven marking shares. The completed
    prevalence.json records selected_counterfactual_rows = 1152; interpret
    its results using that corrected counterfactual selection.
14. `scripts/presentation_statistics.py` reconciles Table I against the
    source/eligible-derivative manifests and creates all fifteen Table III
    conditions (identity plus fourteen edits).
15. `scripts/build_conference_paper.py` inserts measured values and orders
    all references by first citation. The saved IEEE source is compiled and
    its rendered pages inspected before delivery.
    The expanded anonymous version has compiled to six pages including
    references, within ICREST's verified 4–6-page limit. The additional
    sixth-page charge applies under the author guidelines.
16. `python scripts/check_conference_submission.py --visual-reviewed` checks
    the final PDF and saved measurements after all rendered pages have been
    inspected. It records counts, source isolation, exact fallback, frozen
    head hashes, references, embedded fonts, page size, and artifact hashes
    in outputs/extension/submission_check.json. It performs no model fitting
    or inference. Omit the flag if visual inspection has not been completed.

## Interpretation

- The gate preserves the exact probe probability and unsaturated ranking
  below 13/16; it does not authenticate origin. It uses the classical mark,
  not TrustMark's exact-message/ECC criterion. False payload matches remain.
- Core weak-recovery ResNet AUC changes from 0.882 under original fusion
  to 0.923 under the gate, versus probe 0.922. CLIP has no clear fusion gain.
- Fresh ResNet/CLIP AUC is 0.912/0.991. The frozen policy accepts 9/96 real
  inputs, all from Open Images (9/48), demonstrating a source shift. Its
  validation 5% empirical error constraint is not a deployment guarantee.
- Published UFD reaches 0.730 on clean core and 0.442 on the fresh sample.
  This negative result is retained; its threshold and score direction are
  not changed after evaluation.
- Three extra noise seeds give classical recovery 1.000 throughout and
  TrustMark 0.986/1.000/1.000 on 72 payload outputs from 24 AI sources.
- Watermarks differ in payload capacity, ECC, and distortion. Small source
  counts, one regeneration checkpoint, and fixed settings restrict scope.
- Actual diffusion denoising is now executed. The legacy grid-shift/JPEG/noise
  operation remains a separate local condition and must not be renamed
  diffusion or treated as a published attack reproduction.
- Reconstructed photographic inputs contain generated pixels. Origin
  labels for those rows mean input-source provenance, not authentic-photo
  ground truth. Positive scores are not automatically false accusations.
- Native collection labels and pretraining overlap remain unverified.
  No public-data provenance check authenticates all depicted content.

OpenCV reports about 361.2 dB for identical pixels because of its numerical
PSNR cap. Interpret those identity-to-cover measurements as no distortion;
the manuscript's clean marked PSNR instead reports embedding distortion.

The anonymous paper and archives are local artifacts. No upload, author
approval, acceptance, or post-acceptance PDF eXpress validation is asserted.
