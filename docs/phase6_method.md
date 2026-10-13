# Phase 6: paired harm-constrained active/passive fusion

## Status and scientific claim

This is a proposed decision-layer contribution. It uses existing watermark encoders, frozen passive probes, constrained model selection and source-level calibration. None of those components is claimed to be new in isolation. The proposed distinction is learning bounded positive watermark evidence from **paired marked/unmarked versions of the same AI source and edit**, while explicitly limiting loss on unmarked AI images at a separately calibrated photographic false-positive operating point. Related work and closest overlaps are documented in `phase6_gap.md`.

Development data have already been examined in earlier project stages. They are retrospective development evidence. A confirmatory claim requires a new source-disjoint calibration sample and a source-disjoint final test collected after the scoring rule, hypotheses, views, queries and selection are locked. A failed final criterion must be reported; it does not authorize selecting another model or changing a threshold.

The single confirmatory system is **TrustMark with the frozen ResNet probe**. Its final population is the fixed quota mixture of 600 AI sources: 200 from each of three previously unseen generator collections, with four common views per source: identity, JPEG quality 50, side crop and a 32-pixel round trip. Each source has one unmarked cover and three marked covers, with different registered payloads, under each view. CLIP, DWT-DCT, subgroup results, marked shares, attacks, watermark verification and abstention are secondary analyses. Their intervals are descriptive and do not establish simultaneous coverage across all systems, families or views.

## Received-image evidence

Each received image is queried against exactly the same three registered 16-bit payload strings. The input CSV preserves leading zeroes and includes one row per image, decoder and query. The decoder is separate from the actual `watermark_method`: only an image marked by that decoder, or an unmarked image, is eligible.

Prediction uses the original frozen passive logit and eleven numeric decoder features:

- Bit agreement and the minimum agreement across the received image and its fixed JPEG-90 view.
- Query-signed mean margin, its tenth percentile and mean absolute margin.
- Bit stability and margin consistency between those two received-image views.
- Full registered-message recovery, recovery in both views, ECC validity and agreement with the registered codeword.

The fixed JPEG-90 transformation is made from the **received image**, never from an undamaged reference. Source identity, collection, role, origin label, embedded payload and attack condition never enter prediction. They are used only for pairing, weighting, splitting and reporting. TrustMark full recovery additionally requires valid ECC, the expected schema and the entire expected padded message. DWT-DCT has no ECC-validity claim. A query result does not authenticate an issuer.

Feature means and standard deviations, and passive-logit normalization, use fitting rows only. Very small scales are floored at `1e-8` and standardized predictors are clipped to `[-20,20]`. There is no feature selection on final calibration or test data. The complete model and every transformation are stored in `models.json`.

## Score and fit objective

Let `z_i` be the frozen passive logit, `x_iq` the standardized decoder features and passive logit for query `q`, and `theta` a learned vector including an intercept. The score is

`c_i = B max_q sigmoid(theta^T x_iq)`

`s_i = z_i + c_i`, with `0 <= c_i <= B`.

The same maximum over three queries is applied to photographs, unmarked AI images, marked AI images, all comparators and independent calibration. It cannot select the true payload using metadata.

The fitted objective is

`sum_i w_i CE(y_i, sigmoid(z_i+c_i))`

`+ lambda_null sum_{i:unmarked} u_i c_i^2`

`+ 0.25 sum_{(m,u)} v_mu softplus(-(c_m-c_u))`

`+ 0.001 ||theta||^2/2`.

The CE weights give equal mass to real and AI classes, then equal mass to sources within a class, then equal mass to the marked and unmarked states present within a source, then equal mass to images within each state. Null weights renormalize the CE weights over **all unmarked covers**, including real and AI images. Pair weights give equal mass to AI sources and then to their paired payload/view covers. Each marked AI cover is paired with the unmarked version of the same source and condition; three payloads share the same unmarked counterpart. A missing or ambiguous counterpart is rejected.

This pairing objective seeks extra evidence on marked covers. It is not an origin or attack-label input at inference. Nonnegative score correction alone does **not** ensure unmarked noninferiority, because separately calibrated decision thresholds can rise. Unmarked harm is therefore checked on development data and independently assessed on final paired sources.

Optimization uses analytic gradients and a deterministic L-BFGS-B procedure, with coefficient bounds `[-20,20]`, at most 500 iterations and a fixed query-symmetric initialization. At a max-query tie the implementation uses the first argmax subgradient. Candidate fits that fail convergence or finite-value checks are ineligible and are logged. The synthetic verification includes finite-difference gradient agreement away from a tie.

## Development roles and selection

The development pool contains 936 source-disjoint images: 68 Stable Diffusion, 68 DALL-E, 400 COCO photographs and 400 Open Images photographs. Within each collection, 50% of sources are fitting, 25% tuning and 25% constraint-assessment sources. Fitting and tuning use identity and JPEG-50 views. Constraint assessment adds crop and the 32-pixel round trip.

Candidate bounds are `B in {1,2,4,8,16}`; null penalties are `{1,10}`. The bound grid was expanded to include 8 and 16 **before fresh scores**, since the frozen passive logits can be strongly negative. Pair strength and ridge penalty remain fixed. The same grid is used for all correction variants, except that the no-null ablation has null penalty zero.

For each candidate, tuning selects its operating cut independently from photographic **source maxima over all supplied views and query scores**. The empirical cut permits at most `floor(0.05*N_real)` false-positive sources, using a next-representable threshold above the relevant tied score. A candidate is feasible only if unmarked TPR loses at most one percentage point relative to the probe in every observed generator/condition group. Among feasible candidates, highest source-balanced marked TPR wins; ties prefer smaller `B`, then larger null penalty. If none qualifies, the scoring rule is exactly the original probe (`B=0`).

The selected model is assessed once on constraint-role sources using its **unchanged tuning cut**. A failure is recorded; no different model is selected from this result. This assessment is a diagnostic stress check, not a confirmatory test or population guarantee.

## Comparators and ablations

All comparators use the same fitting/tuning sources, original frozen passive scores, three-query maximum and independent final photographic calibration:

- Original passive probe.
- Logistic static fusion of passive probability and bit agreement.
- Hard 13-of-16 gate, using static fusion for qualifying queries and the probe otherwise.
- Fixed confidence weighting: `z + max_q 2 max(2*agreement-1,0)*stability`.
- The project's earlier smooth reliability-mixture formula, refitted on Phase 6 fitting sources.
- The project's earlier capped positive-evidence formula, refitted with source-level false-positive constraints.
- Unconstrained logistic fusion using all decoder features and the normalized passive logit.
- Watermark-only maximum bit agreement, maximum signed margin and native full-message recovery.

The earlier reliability and capped-positive comparators are **formula-family adaptations**, not unchanged archived prediction files. Their adaptations are recorded in the model artifact. A learned watermark encoder is not retrained. Binary or quantized watermark scores can yield conservative source-calibrated cuts with many ties; this is reported without changing the score or target to recover recall.

Ablations remove the null penalty, pair penalty, stability-related features or full-message/ECC features. Each has the same tuning rule and its own final photographic cut. They are secondary, not competing primary models selected after test results. The strongest origin/fusion baseline is selected by greatest marked TPR among the original probe and six fusion comparators **passing the same tuning unmarked-harm and source-FPR requirements**, with deterministic name-based tie breaking, and frozen for final comparison. Watermark-only scores remain benchmark rows and cannot qualify as origin-fusion comparators. This eligibility correction was made before fresh calibration or test scores; the first fit is retained at `outputs/phase6/fusion`, its exact implementation source at `outputs/phase6/fusion_v1_source.py`, and the corrected fit at `outputs/phase6/fusion_v2`. Model parameters and the candidate grid are unchanged.

## Independent calibration

Final calibration uses new unmarked photographs, disjoint from every development and final-test source. Every calibration source/decoder must include all four common views. Each frozen system gets its own cut; every cover and query is included in the source maximum.

Within each photographic collection, let `M_j` be the maximum score for calibration source `j`. With `n` calibration sources, use order-statistic rank `k=ceil((n+1)*(1-0.05))`, and accept only a score **strictly greater** than that cut. If `k>n`, the cut is infinity and no image is accepted. The deployed cut is the maximum of the collection cuts. Taking that maximum is conservative for each represented photographic collection.

This gives marginal source-level false-positive control under a fixed score/query/view family and exchangeable sources within the calibrated photographic collection. It does not guarantee every realized split, an arbitrary new collection, a broader future attack family, unmarked AI false matches or wrong-payload false matches. Per-source dependence among edited covers and queries is handled by the source maximum.

Decoder signed-margin verification is calibrated separately against photographic source maxima. Full-message recovery, photographic false matches, unmarked-AI false matches and wrong-payload queries are all reported separately. Photographic calibration cannot be used to assert statistical coverage for the latter two nulls. Native recovery is evidence of a registered message, not proof of image origin or sender identity.

## Preregistered endpoints and inference

The positive TPR endpoint is the mean within each source over its matched marked covers and common views, then the mean over sources. The unmarked endpoint uses the same construction on unmarked views. Photographic false positives are reported both as source-balanced average-image FPR and as the fraction of sources with **any** accepted view. The nominal source-level 5% cut is independently calibrated for each comparator.

The confirmatory TrustMark/ResNet claim requires:

1. Marked TPR gain over the probe, with the lower endpoint of a two-sided paired 95% source-bootstrap CI greater than zero.
2. Marked TPR gain over the strongest baseline chosen on development tuning, under the same paired interval rule.
3. Unmarked TPR loss below `0.01` under the declared one-sided paired procedure.
4. Realized photographic source FPR no greater than 5% on the final sample; its uncertainty is also reported.

Source bootstrap uses 2,000 deterministic draws stratified by collection. The same resampled sources are used for every method and endpoint, retaining each source's payloads and edits. This targets the declared fixed collection quota mixture. Two-sided intervals are percentile 2.5%/97.5%; the one-sided loss upper endpoint is the negative of the gain's fifth percentile. These bootstrap intervals are approximate; no distribution-free finite-sample bootstrap guarantee is claimed. AUCs, per-family/condition results, marked-share reweighting, extra attacks and selection risk/coverage are secondary.

### Zero harmful switches

A zero-loss bootstrap is degenerate and cannot alone certify a one-percentage-point population margin. The evaluator therefore also forms a source event: at least one unmarked common view accepted by the probe but rejected by the new system. Such any-view harm upper-bounds average-view TPR loss.

For independent source events with fixed collection quotas `n_g` and collection risks `p_g`,

`P(no harm) = product_g (1-p_g)^n_g <= (1-sum_g n_g*p_g/N)^N`.

The inequality follows from log concavity. With **zero** harmful source events, the fixed-quota mean harm has a conditional one-sided 95% upper bound `1-0.05^(1/N)`, about `0.00498` for 600 sources. This can support the pooled 1% margin, under the stated independence and sampling assumptions. It does not require equal risk across families. The same argument is **not** used as a pooled binomial interval when any harm is observed.

Separate per-collection Clopper-Pearson bounds use `delta=0.05/G`, a union bound, and iid/common-risk sources within each collection. With 200 sources per family, zero harm does not certify a 1% simultaneous family margin; the result remains unresolved. These group bounds and a quota-weighted upper bound are reported separately. When harm is nonzero, a positive paired-bootstrap noninferiority decision is labeled approximate and a finite-sample certificate is reported only if the conservative group bound passes. The exact probe fallback has analytically zero change for every input, but cannot establish a new-method gain.

### Marked shares and abstention

Marked shares `{0,0.25,0.5,0.75,1}` reweight paired marked/unmarked source TPRs at the same frozen cuts, with paired bootstrap intervals. They do not retune thresholds.

A descriptive abstention band is fixed from the twentieth percentile of absolute distance from the final cut on calibration negatives. Test coverage and error are reported at band multipliers `{0,0.5,1,2,4}`, including retained counts by collection, condition and origin. This is a fixed heuristic operating curve. It has no selective-risk, shifted-coverage or class-balanced guarantee, and it must not be advertised as detecting arbitrary domain shift.

## Artifacts and commands

Run from the project root with the research Python dependencies on `PYTHONPATH`:

```text
python scripts/test_phase6_fusion.py
python scripts/phase6_fusion.py --output outputs/phase6/fusion_v2 fit --features outputs/phase6/development/features.csv
python scripts/phase6_fusion.py --output outputs/phase6/fusion_v2 calibrate --features outputs/phase6/calibration/features.csv
python scripts/phase6_fusion.py --output outputs/phase6/fusion_v2 evaluate --features outputs/phase6/test/features.csv --name fresh_test
```

The learner writes `protocol.json`, `models.json`, `selection.json`, `constraint_diagnostic.json`, development source IDs and `fit_complete.json`. Receipts hash the development CSV, original probe heads, implementation and artifacts. Calibration writes policies, source IDs, `thresholds_locked.json` and a calibration completion receipt before final test scoring. Evaluation writes metrics, per-cover predictions, source IDs and a completion receipt. Repeating a stage with the same inputs verifies its existing receipt and does not refit, recalibrate or rescore. Changed inputs, implementation or artifacts are rejected; an old completed run must be preserved and a new output directory explicitly chosen for a new protocol.

Final evaluation refuses source overlap with development or calibration, and extra conditions require an explicit diagnostic flag. Common-view coverage and the fixed 600-source/three-family sample are checked separately: incomplete coverage makes the confirmatory assessment invalid, even if descriptive metrics are computable. The test must not be used to repair a failed candidate. No acceptance probability or novelty score is guaranteed by this protocol.
