# Enlarged Phase 3 Regeneration Protocol

## Current status and scope

The protocol was frozen on 9 October 2026 at 09:25:12 UTC. Generation and
recovery decoding have completed under `outputs/phase3/regeneration/`.
[The final summary](../outputs/phase3/regeneration/summary.json) records the
measurements and provenance. The completed design follows the frozen protocol.

There are 204 previously examined AI test sources: 68 each from Stable
Diffusion, DALL·E 3, and Midjourney. Each contributes a DWT–DCT and an official
TrustMark Q cover carrying `1011010010110100`. Three operations on 408
covers yield 1,224 completed outputs. Selection and cover hashes are locked in
`selected_covers.csv` and `protocol.json`.

This larger run contains AI-source covers only, without real-image or
unmarked-image controls. The earlier 24-AI/24-real regeneration comparison
with unmarked controls remains separately archived in
`outputs/extension/diffusion/` and `outputs/extension/watermarks/`.

## Model and operations

The checkpoint is **Realistic Vision V5.1 noVAE**, an SD1.5-family model,
revision `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`. Cached component
identities, including the VAE, are recorded in `protocol.json`. All covers
are 256×256.

- Posterior-mean VAE round trip: encode, use the posterior mode, and decode
  without diffusion denoising.
- DDIM strength 0.25: twenty configured steps, five effective denoising steps.
- DDIM strength 0.50: twenty configured steps, ten effective denoising steps.

Prompt is empty, guidance scale is 1, and batch size is one. A CPU random
generator is reset to seed 20261009 for each output. UNet/text encoder use
float16; the VAE uses float32. DDIM is constructed from the cached scheduler
configuration.

The driver requests `--mode auto`: resident CUDA models, with model CPU
offload as a fallback on CUDA out-of-memory. The recorded initial runtime
uses resident models on an NVIDIA GTX 1050 Ti, PyTorch 2.5.1+cu121, Diffusers
0.35.1, default Torch SDPA attention, and VAE slicing. `runtime.json` and
manifest rows disclose actual placement and any fallback. This memory and
attention configuration is recorded separately from the earlier smaller run.

## Verification and distortion

Classical verification accepts at least thirteen of sixteen expected bits.
TrustMark requires valid ECC, the expected schema, and exact recovery of all
61 expected data bits: the payload followed by forty-five zeros. These
acceptance criteria differ.

Analysis uses the decoder corresponding to each cover. The batched TrustMark
path is checked against the official public decode API on saved anchors.
Each method/condition has one output per source. Final summaries include
recovery by collection, source-bootstrap intervals, and descriptive exact
Clopper–Pearson intervals.

Attack PSNR/SSIM use the **marked input cover** as reference. The local
watermark experiment separately measures embedding distortion against the
unmarked source image. Pixel metrics do not establish semantic preservation
or perceptual acceptability; payload removal must be interpreted alongside
its distortion.

## Completed findings

The VAE-only round trip verifies 76/204 DWT–DCT sources (0.3725) and
21/204 TrustMark sources (0.1029). Mean PSNR against the marked input cover
is 26.22/26.47 dB, respectively.

At diffusion strength 0.25, DWT–DCT verifies 10/204 sources (0.0490),
while TrustMark verifies 0/204. Mean PSNR is 20.75/20.84 dB and mean SSIM
is 0.587/0.599, respectively.

At strength 0.50, DWT–DCT verifies 6/204 sources (0.0294), while TrustMark
verifies 0/204. Mean PSNR is 16.07/16.14 dB and mean SSIM is 0.397/0.407,
respectively. This substantial distortion limits claims about useful removal.
For each zero-of-204 TrustMark result, the descriptive exact 95% upper
verification bound is 0.0179; zero observed recovery does not prove that
recovery is impossible. Full intervals and collection results are in
[summary.json](../outputs/phase3/regeneration/summary.json).

The separate enlarged local comparison has 204 AI and 204 real sources,
21,216 outputs and 42,432 decoder rows. Its TrustMark decoder processed
13,840 outputs on CPU and 7,376 on CUDA. The device is recorded per row;
both paths use official FP32 weights, preprocessing, ECC, and public-API
decode anchors, with no retuning on resume. See
[the local summary](../outputs/phase3/watermarks/summary.json) and
[analysis provenance](../outputs/phase3/watermarks/analysis_provenance.json).

## Reproduction and evidence

Complete and preserve the larger watermark cover preparation first. Use its
recorded environment and the pinned cached diffusion checkpoint:

    python scripts/phase3_watermarks.py --stage select
    python scripts/phase3_watermarks.py --stage prepare --device cpu --threads 2
    python scripts/phase3_regeneration.py --stage run --mode auto
    python scripts/phase3_regeneration.py --stage analyze --device cuda --batch-size 16

Generation resumes from manifest rows after checking output hashes. Completion
requires all 1,224 unique cover/condition outputs. Analysis requires that
complete count and checks input/output identities.

- `protocol.json`, `selected_covers.csv`: frozen settings and selection.
- `runtime.json`: versions, device, attention processors, memory placement.
- `manifest.csv`: hashes, source/method IDs, seeds, times, memory placement.
- `complete.json`: completed counts and manifest fingerprint.
- `measurements.csv`, `summary.json`: recovery, PSNR/SSIM, intervals,
  official decode anchors, and analysis provenance after completion.

The generated illustration is excluded from benchmark aggregates.
`example_regeneration.json` and
`manuscript/assets/generated-asset-provenance.json` identify its source,
actual processing, output identities, and exclusion. Its strength-0.50
regeneration precedes construction of the six-image strip.

After all requested results and figure measurements are complete:

    python scripts/phase3_regeneration.py --stage example_prepare
    python scripts/build_phase3_figures.py --stage both
    python scripts/build_conference_paper.py

The revised manuscript has compiled to five pages including references.
Compile and inspect every PDF page after rebuilding; final visual review,
font checks, and source/PDF fingerprinting are separate from compilation.
The previous verified six-page source/PDF remains preserved in
`manuscript/archive/pre_phase123_20261009.*`.

## Interpretation and future work

This retrospective robustness comparison uses already examined sources, one
checkpoint, one regeneration seed, and two strengths. It is not a new
independent test collection or an exact reproduction of a cited published
removal attack. Labels describe input-source provenance. The AI-only design
supports payload-recovery comparison; photographic false-positive rates and
marked/unmarked forensic counterfactuals require separate controlled runs.

Additional primary keyed/learned marks, published removal attacks,
caption-paired real/AI scenes, another prospectively locked untouched test,
and public benchmark release remain Phase 4. A cleaned local anonymous
package is prepared for eventual publication; no public URL is available.
