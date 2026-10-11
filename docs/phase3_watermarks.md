# Enlarged classical and learned watermark comparison

The enlarged comparison is an exploratory evaluation of DWT–DCT and official
TrustMark Q. It reuses previously examined corrected test sources, so it is
not a new untouched dataset. It preserves the original 24-pair experiment
under `outputs/extension/watermarks/` and writes its new evidence under
`outputs/phase3/watermarks/`.

## Sources and payloads

The fixed selection contains 204 AI sources: 68 each from Stable Diffusion,
DALL-E 3, and Midjourney. Sources are selected by the SHA256 ordering of
`20261012|source_id` within each collection. A maximum-total-CLIP-cosine
one-to-one assignment pairs these with 204 distinct COCO photographs. Matching
uses frozen clean-image features rather than watermark or detector outcomes.
CLIP similarity matching does not establish identical scenes, prompts, or
capture conditions.

Inputs are the archived standardized 256x256 study images. For each AI source,
the comparison creates one unmarked cover, three DWT–DCT covers, and three
TrustMark covers; each paired real source has one unmarked cover. Thus there
are 408 source units and 1,632 covers. The three payloads are:

    1011010010110100
    0100101101001011
    1100101000110101

DWT–DCT embeds 16 bits at strength 12 and verifies agreement of at least
13/16. TrustMark uses the official 0.9.2 Q variant, default embedding strength
1, and BCH5. Its message has 61 data bits: the 16-bit payload followed by
45 fixed zeros. Its encoded codeword has 100 bits. TrustMark verification
requires successful ECC, the expected schema, and exact agreement with the
full 61-bit expected message. These methods are not matched for payload
capacity, embedding distortion, or acceptance rule.

## Transformations and decoding

The local comparison applies identity; JPEG at two qualities; blur at two
strengths; Gaussian noise; side crop; 32x32 round trip; the original grid
shift/JPEG/noise transform; and mid-band spectral removal. Three additional
Gaussian-noise conditions use sigma 6 and seeds 20261009, 20261010, and
20261011. The 13 conditions produce 21,216 transformed covers before any
separate diffusion manifest is appended. The grid-shift transformation is
not a diffusion reconstruction.

Both watermark decoders run on every transformed cover, including unmarked
AI and real controls. The main recovery summaries use the matching decoder
on marked AI and each decoder's expected-message match on unmarked real
photos. The broader measurement archive also retains cross-method and
unmarked-AI results.

The learned decoder uses the official preprocessing, bilinear resize to its
decoder resolution, input range [-1,1], sign threshold, and official BCH
decoding. Batched inference is checked against the public single-image
decode API on recorded anchors. Payload bit accuracy uses the first 16 raw
bits; TrustMark's final acceptance still requires the complete ECC/schema/
message rule. Every transformed image must match its frozen file hash.

Local recovery rates average payload repeats within source for the bootstrap,
then resample source units (2,000 draws). The separate exact intervals target
the event that every evaluated payload of one source verifies. The common
success-probability assumption of the exact binomial interval is descriptive;
it does not make the three collections a random sample of all generators.
Do not treat payload, noise, or attack repeats as independent source images.

## Diffusion scope

The enlarged regeneration driver, `scripts/phase3_regeneration.py`, selects
the default-payload DWT–DCT and TrustMark covers from all 204 AI sources.
It evaluates 408 marked covers under a posterior-mean VAE round trip and
DDIM strengths 0.25/0.50: 1,224 planned outputs. This larger regeneration
run contains neither real-photo controls nor unmarked AI controls. Its
separate protocol, runtime, output completion, and matching-decoder summary
are under `outputs/phase3/regeneration/`; `docs/phase3_regeneration.md`
records the procedure. Only completed output and analysis markers establish
the enlarged regeneration result. The local watermark summary and this
separate default-payload regeneration summary are not interchangeable.

The earlier 24-AI-source regeneration subset additionally includes paired
unmarked controls and is preserved under `outputs/extension/diffusion/`.
Both runs use the pinned SD1.5-family checkpoint, but their source inventory
and runtime placement differ. Do not label the earlier controls as a
204-source control experiment. Each condition's summary records its actual
source count. Neither run is claimed as an exact reproduction of a cited
published watermark-removal attack.

Origin labels of reconstructed photographs refer to the original input
source. Such outputs contain generated pixels, so their payload-match rate
is not an authenticity false-positive rate on untouched photographs.

## Reconstruction and release

With the selected-source manifests, standardized inputs, and official
checkpoint assets available in a working copy:

    python scripts/phase3_watermarks.py --stage select
    python scripts/phase3_watermarks.py --stage prepare --device cpu
    python scripts/phase3_watermarks.py --stage attacks
    python scripts/phase3_watermarks.py --stage analyze --device cuda --batch-size 16

An external regeneration manifest can be supplied using
`--diffusion-manifest PATH`; it must reference frozen cover IDs and contain
unique cover/condition rows. It changes the evaluated condition inventory,
not the watermark thresholds. The paper uses the separate enlarged
regeneration summary from its own driver. Preserve earlier analyses when
changing scope.

Selection and cover manifests are frozen before attack analysis. Completed
results are identified by the final `summary.json` and source-verification
intervals, rather than by the cover inventory or a partial decode file.
The anonymous local ZIP includes method code, checkpoint fingerprints, and
aggregate evidence, but excludes source IDs, images, embeddings, models,
and source-level decode rows. No hosted repository URL or independent
end-to-end reproduction is claimed.
