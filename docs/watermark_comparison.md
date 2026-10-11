# Fixed-rule watermark comparison and false matches

## What this addition establishes

The reporting analysis uses the completed local 204-AI/204-COCO experiment.
It performs no new decoding, fitting, image generation, or threshold search.
Its plan was saved before the new reporting script read outcome counts.
The underlying experiment and some aggregate results were already examined;
this is a retrospective descriptive analysis, not a preregistered validation.

The analysis recomputed all 42,432 saved verification flags, confirmed hashes
between the attack manifest and measurement archive, and checked the full
cover inventory. Every one of 21,216 transformed covers has both decoder
rows. The archive comprises:

- 15,912 matching-decoder rows on marked AI images;
- 15,912 cross-method decoder rows on marked AI images;
- 10,608 decoder rows on unmarked AI and real controls.

Each of the 204 AI sources has one unmarked cover and three covers for each
watermark. Each of the 204 distinct COCO sources has one unmarked cover.
Thirteen fixed transformations are applied to each cover. Three payload
queries on one unmarked cover reuse the same pixels and decoder output;
they are not three independent images.

## Acceptance criteria

DWT–DCT accepts when at least 13 of 16 stored bits agree with the expected
payload. TrustMark Q with BCH5 accepts only if ECC succeeds, the decoded
schema is 1, and the entire decoded 61-bit message equals the 16-bit payload
followed by 45 zeros. The script verifies these rules against every stored
verification flag. It can query all three payloads from stored DWT bits or
the saved ECC-valid TrustMark message without rerunning a model.

These criteria differ in payload capacity and acceptance stringency.
TrustMark's raw first-16-bit agreement is not equivalent to its ECC/message
verification. The results describe each method under its specified rule;
they do not establish a ranking at equal false-positive rate, equal
embedding distortion, or equal capacity. All 204-source inputs are from the
test partition. Selecting a new threshold here would tune on the test set.
No learned-watermark validation split or calibrated soft verification score
exists in this archive, so a fair validation-fixed matched-FPR comparison
cannot be recovered from it.

## Table-ready photographic false matches

For the default expected payload `1011010010110100`, DWT–DCT photographic
acceptance is:

- Identity: 2/204 = 0.009804; exact 95% source interval [0.001190, 0.034965].
- JPEG 50: 1/204 = 0.004902; interval [0.000124, 0.027008].
- Side crop: 2/204 = 0.009804; interval [0.001190, 0.034965].
- 32×32 round trip: 1/204 = 0.004902; interval [0.000124, 0.027008].
- Grid shift/JPEG/noise: 3/204 = 0.014706; interval [0.003043, 0.042373].
- Mid-band spectral removal: 0/204; interval [0, 0.017920].

TrustMark returns zero expected-message matches for photographs and
unmarked AI sources in each of the 13 conditions, whether querying one
payload or accepting any of the three. Every per-condition zero has the
two-sided exact source interval [0, 0.017920]. Zero observed false matches
therefore does not establish zero population false-positive risk.

Accepting any of the three payloads raises DWT–DCT photographic acceptance
for the six conditions above to 7/204, 6/204, 4/204, 4/204, 5/204, and
4/204, respectively. This change measures the explicit three-query rule;
it is not an improvement in per-payload recovery.

Across all 13 conditions, the default-query photographic mean is
29/2,652 = 0.010935, with source-bootstrap interval [0.005279, 0.017345].
The second and third query means are 36/2,652 = 0.013575 and
19/2,652 = 0.007164. The any-of-three rule accepts 84/2,652 = 0.031674,
with source-bootstrap interval [0.021116, 0.043741]. These denominators
contain dependent repeated transformations of 204 sources.

The distinct event that a photograph is accepted at least once under any
of the three queries and any of the 13 conditions occurs for 46/204 sources
= 0.225490, with exact interval [0.170068, 0.289073]. This is not a 22.5%
per-image false-positive rate. For TrustMark, zero such source events has
the exact upper bound 0.017920; the degenerate [0,0] bootstrap mean interval
must not be presented as evidence of zero risk.

## Unmarked AI and cross-method controls

Default-query DWT–DCT acceptance of unmarked AI is 3/204 for identity,
1/204 for JPEG 50, 1/204 for crop, 8/204 for low resolution, 8/204 for
the grid compound, and 1/204 for spectral removal. These are false matches
to an expected watermark, not false AI-origin classifications: the input
origin is actually AI.

Across all conditions, default-query unmarked-AI acceptance is
40/2,652 = 0.015083 [0.009050, 0.021870]; any-of-three acceptance is
99/2,652 = 0.037330 [0.027526, 0.047511]. Under the latter rule, 62/204
sources verify at least once [0.241627, 0.372019]. Cross-method results
remain separate: an image carrying the other mark is not an unmarked
photograph. Per-condition, per-payload cross-method source intervals are
saved in `cross_method_acceptance.csv`.

## Security and regeneration limits

The payloads are public and unkeyed, and the embedding algorithms are
available. An expected-message match is evidence of recoverable bits under
a known algorithm; it does not authenticate an author, generation model,
or an authorized issuer. This experiment contains neither unauthorized
embedding trials nor a signature/key verification system. Such claims must
not be inferred from a low accidental match rate.

For illustration only, independent uniform 16-bit outputs would pass the
DWT threshold for one payload with probability 697/65,536 = 0.010635,
and for any of these three payloads with probability 2,091/65,536 =
0.031906. Actual image-decoder outputs are neither proven uniform nor
independent; these calculations are not security guarantees.

The enlarged regeneration study has only marked AI inputs. It supplies no
204-source photographic or unmarked-AI regeneration controls and is not
included in these false-match summaries. Its recovery rates can be shown
with the local results only if the missing control cells are explicitly
marked as not measured. The local `regenerate_s2` condition is a grid shift
plus JPEG and noise, not diffusion regeneration.

## Concise novelty positioning

GenImage already evaluates cross-generator and degraded-image detection;
do not claim that such evaluations are new. Its description explicitly
includes low resolution, blur, and compression.
[Primary GenImage source](https://arxiv.org/abs/2306.08571)

AdvMark studies watermark interference with passive deepfake detectors and
fine-tunes robust watermark embedding to improve detectability. The
present study leaves watermark embedding fixed and evaluates the effects
of matched real processing, edited unmarked AI controls, and conditional
score fusion.
[Primary AdvMark source](https://www.ijcai.org/proceedings/2024/673)

Suggested paper wording:

> GenImage establishes cross-generator and degraded-image evaluation, and
> AdvMark optimizes watermark embedding for passive detectability. We hold
> embedding fixed and quantify how matched real-image edits and unmarked AI
> controls affect origin probes and agreement-gated fusion, while reporting
> expected-message recovery separately from photographic false matches.

This is an empirical protocol contribution, not a new watermark or
backbone. Any causal claim that matched processing alone explains the
training ablation also requires a training-size-controlled comparison.

## Reproduction and saved evidence

Run `python scripts/phase4_watermark_reporting.py` with NumPy and SciPy in
an environment containing the saved Phase 3 measurement archive. It needs
no GPU, watermark package, network, or source images. It recomputes the
fixed-rule aggregate checks and writes only new reporting artifacts under
`outputs/phase4/watermark_reporting/`.

The directory contains the analysis plan, input hashes and audit,
per-condition control rates, source-clustered overall rates, cross-method
acceptance, and `table_ready.csv`. That last file contains marked recovery
averaged within source over payloads, default-query control rates, and
any-of-three control rates. The original experiment remains unchanged.
