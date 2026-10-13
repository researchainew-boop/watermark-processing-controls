# Phase 6 published attacks and quality controls

## Scope

The reusable driver is `scripts/phase6_attacks.py`. Development outputs are
isolated under `outputs/phase6/attack_development/`. Existing experiments and
the manuscript are unchanged by this track. The pilot uses already examined
sources and is not a fresh confirmation experiment. Neither watermark recovery
nor origin-detector scores were calculated to choose attack settings.

There are two distinct regeneration mechanisms: learned-compression
reconstruction and SD img2img. Only one diffusion checkpoint family is present.
This does not establish transfer across independent diffusion families.

## Published implementation and disclosed adaptations

The primary reference is [WAVES](https://wavesbench.github.io/), whose
[official code](https://github.com/umd-huang-lab/WAVES) is pinned to commit
`1477635d306d6f8c77a588e6441bd4ba301a1f7a`.

1. **AdvEmbB-RN18** maximizes the mean squared difference from the input's
   frozen ImageNet ResNet18 embedding. It uses random initialization, 200
   projected sign-gradient steps, epsilon equal to strength/255, and step size
   0.05 epsilon. Published strengths are 2, 4, 6, and 8. Inputs are resized to
   224 before ImageNet normalization and average-pooled feature extraction.
   The port uses the official torchvision weights and changes I/O to one image
   per call, permits CPU execution, and fixes randomness per source. The
   recorded torchvision version determines interpolation/antialias behavior;
   the historical paper environment is not claimed to be reproduced exactly.
   This surrogate is the same backbone architecture as one forensic probe.
   Report this relationship when interpreting attacks on that probe.

2. **Regen-VAE / bmshj2018-factorized q3** uses the official pretrained MSE
   checkpoint of the [CompressAI implementation](https://github.com/InterDigitalInc/CompressAI)
   of [Ballé et al.'s learned image compression](https://arxiv.org/abs/1802.01436).
   WAVES performs a 512-pixel reconstruction; our 256-pixel experiment adds an
   explicitly labeled INTER_AREA restoration. The inference port uses the
   checkpoint's analysis/synthesis convolutions, GDN parameters, and entropy
   medians. It omits entropy coding and likelihood calculation, which do not
   change the reconstructed tensor. It does not create or test a compressed
   bitstream. The upstream attack entry function has a `quality`/`strength`
   argument mismatch; the port calls the published q3 model operation directly.

3. **SD15 DDIM regeneration** uses the cached Realistic Vision V5.1 checkpoint,
   revision `1e9f017a7b1eaefb63a1900ea6c5953d2739fd21`. It performs img2img with
   DDIM, 20 configured steps, empty prompt and guidance1. Strengths .05/.1/.2
   give 1/2/4 effective denoising steps. VAE runs FP32; the UNet runs FP16 on
   CUDA. This is an adaptation of diffusion regeneration, **not an exact
   reproduction of WAVES Regen-Diff**, which uses a custom ReSD pipeline,
   a different checkpoint, explicit noise timestep and guidance7.5.

Published source files and their complete SHA256 identities are retained in
`assets/` and `pilot_protocol.json`. The official compression checkpoint SHA256
is `5c6f152b2597de5e0f9143e8173d8c0529310aec523d07e9e9ced96825621d82`.
Source files retain their upstream copyright/license notices. A release that
redistributes the CompressAI-derived port should include those notices.
The assets stage downloads only9 pinned public files with full SHA256 checks;
it preserves the full CompressAI copyright/license notice alongside them.
CompressAI reference revision is
`1c9e3eee83a8f4c8162d38b52f78f5b0c40de175`. Official RN18 weights and the
separately pinned full RV checkpoint are runtime prerequisites, with paths
recorded in `public_assets.json`; weights are not part of the cleaned release.

## Public calling interface

All operations accept HxWx3 BGR uint8 input and return BGR uint8 with the same
dimensions. The caller supplies `seed=source_seed(source_id, replicate)`.
The seed deliberately omits payload and marking arm, so arms receive coupled
randomness. Real and synthetic images use the same attack procedure.

- `WavesResNetPGD(device).apply(image, strength=8, seed=seed)`
- `WavesFactorizedReconstruction(device).apply(image, seed=seed)`
- `StableDiffusionRegeneration(device).apply(image, strength=.05, seed=seed)`
- `residual_budget(input, raw_output, target_psnr=35., seed=seed)` returns
  the adapted pixels and a metadata record, including residual gain, actual
  PSNR, target reachability and rounding seed.

Construct one model per attack group. After that group, call `release(model)`,
delete its instance and collect garbage before loading a different model.
Final-source inference must be controlled by the root's separately frozen
source/model/threshold protocol; the driver itself only selects development
sources.

## Development findings and common quality budget

The fixed quality pilot contains one DALL-E3 source in unmarked, DWT-DCT and
TrustMark arms, plus one COCO and one Open Images photograph: 3 source units,
5 covers, one payload, and 35 raw outputs. Raw means across these covers are:

- PGD strength2: PSNR42.67dB, SSIM.984, about3.46s/image.
- PGD strength4: PSNR37.23dB, SSIM.951, about3.38s/image.
- PGD strength8: PSNR32.10dB, SSIM.871, about3.37s/image.
- Learned reconstruction q3: PSNR32.09dB, SSIM.911, about.095s/image.
- DDIM strength.05: PSNR24.70dB, SSIM.721, about.80s/image.
- DDIM strength.1: PSNR22.43dB, SSIM.639, about.93s/image.
- DDIM strength.2: PSNR19.99dB, SSIM.546, about1.42s/image.

These are feasibility timings on GTX1050Ti, not a robustness result. Do not
claim raw DDIM is quality matched to PGD or reconstruction: its lowest tested
strength already has much larger distortion. Keep the entire raw quality
curve and use the marked input cover as each attack's reference.

The optional **fixed-PSNR residual-budget variant** scales a raw attack
residual back toward its input. It never increases that residual beyond the
raw attack. Source-seeded stochastic rounding makes the pixel distortion
target attainable without using decoder scores. The same source uniforms are
used across marking arms. It is explicitly an attack adaptation, not the
complete original reconstruction/published attack; blending may preserve more
watermark evidence. Matching PSNR does not match semantic or perceptual quality.

On these development covers, PGD8, reconstructionq3 and DDIM.05 each reached
35/38/40dB across 45 adapted outputs. The largest absolute target error was
0.000036dB. Preserve and report the gain as well as SSIM. The initial nearest
rounding pilot is separately retained in `budget_outputs.csv` and
`budget_summary.json`: its maximum error was .339dB, motivating the pixel-only
stochastic-rounding amendment before final inference.

## Verification and reproduction

`pgd_parity.json` records exact tensor/pixel agreement with the unchanged
pinned upstream `WarmupPGDEmbedding` class on an 8-step implementation anchor.
Production runs use all200steps. `reconstruction_parity.json` records zero
tensor error against the official FactorizedPrior/GDN/EntropyBottleneck forward
code, including likelihood computation. Its only reference-environment changes
are an unused entropy-coder stub, inference-only lower-bound handling, and
historical checkpoint key migration. Entropy bitstream coding was not tested.

Use the Python3.12 runtime with the project's existing CUDA/dependency folders.
Put `.tools/lcm-deps` before `.tools/extension-deps` to select peft0.17 required
by Diffusers0.35.1. Commands:

    python scripts/phase6_attacks.py --stage assets
    python scripts/phase6_attacks.py --stage prepare
    python scripts/phase6_attacks.py --stage parity --device cuda
    python scripts/phase6_attacks.py --stage reconstruction_parity --device cpu
    python scripts/phase6_attacks.py --stage pilot --device cuda
    python scripts/phase6_attacks.py --stage budget --device cpu
    python tests/test_phase6_attacks.py

Cheap checks cover source-seed coupling, input dtype/dimensions, exact file
hashing, repeatable target distortion, perturbation bounds and unreachable
budget disclosure. No new test source is consumed by those checks.

## Remaining scope limits

The pilot is too small to estimate recovery or false positives. A final attack
subset must be chosen by deterministic source ordering before its inference,
including unmarked AI and identically processed photographic controls. At this
runtime, 150covers at one PGD strength take roughly9minutes excluding setup;
3 strengths take roughly25minutes. More payloads multiply that cost. Allocate a
smaller prospectively defined source subset if necessary and disclose it.
Neither these attacks nor better presentation establish an8/10 novelty score;
the new fusion contribution must still succeed against fair alternatives.
