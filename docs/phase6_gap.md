# Phase 6: literature gap and prospective method protocol

Status: research design and literature review, 2026-10-11. This file does not report new experimental results or establish a novelty score. The method below is a candidate that must survive the stated comparisons. The current manuscript, earlier protocols, and completed test outputs remain separate evidence.

The finalized objective, bound-grid amendment, single TrustMark/ResNet primary comparison, zero-event fixed-quota harm bound and artifact-lock behavior are specified in `phase6_method.md`. The candidate formulations below are literature-gap planning, not a replacement for that implementation contract.

## What the project already implements

The manuscript combines a frozen passive probability with payload agreement using logistic fusion and a 13-of-16 agreement gate. Its independent source-max calibration controls a marginal error probability under the stated collection and processing-family exchangeability assumptions. Its caption-paired test is now observed and cannot become an untouched test for a method developed from its failures.

Two existing local candidates are essential baselines:

- `scripts/experimental_reliability.py`: two calibrated experts, a smooth gate using agreement, passive-logit magnitude and their interaction, source-grouped ridge selection, constant-mixture and logit-plus-payload comparisons. Its original evaluation is retrospective.
- `scripts/positive_evidence_fusion.py`: a nonnegative supported-payload correction to a calibrated passive logit, with an empirical FPR cap on clean, edited, opposite-payload and pooled real groups. It explicitly makes no population-risk or formula-novelty claim.

Replacing either candidate's name, adjusting its coefficient grid, or reporting more images is insufficient to establish a new method.

## Closest literature and exact overlap

Each summary is a short paraphrase. The distinctions below are our assessment of the inspected material, not an exhaustive claim that no related method exists.

### ImageDetectBench

Guo et al. compare passive and watermark-based detectors on four datasets, common and adversarial perturbations, and runtime. Their real-image captions generate corresponding synthetic images. Section 2.2 treats watermark detection separately; Section 5.2 distinguishes theoretical random-message false matches from measured photographic FPR. The inspected v2 manuscript does not present the proposed paired unmarked-regret fusion objective. This already covers a broad active/passive benchmark, caption pairing, shared attacks and cross-dataset transfer. Our prospective distinction must concern the fusion decision rule, independently assessed harm and operating-point protocol, not those ingredients.

Primary full text: [arXiv v2](https://arxiv.org/html/2411.13553v2). Publication metadata: [CVPR 2026 SAFE workshop](https://openaccess.thecvf.com/content/CVPR2026W/SAFE/html/Guo_AI-generated_Image_Detection_Passive_or_Watermark_CVPRW_2026_paper.html). The full-text review used the arXiv version; differences in the later proceedings version have not been exhaustively compared.

### AdvMark and conditional watermarking

AdvMark's Section 4 trains the watermark encoder/decoder to preserve visual quality and recovery while improving a fixed passive detector's classification. It already identifies watermark-induced harm and beneficial active/passive interaction. Our candidate leaves the embedder, decoder and passive detector frozen and changes the decision layer, using paired unmarked constraints and separate verification nulls. This is a concrete architectural distinction, not evidence that the broad harmless-watermark concept is new. The later ConMark paper conditions embedding and decoding for different forensic functions; it is another relevant antecedent. Its publisher summary was inspected, but full-text access returned 403, so its exact optimization has not been checked.

Primary sources: [AdvMark proceedings and PDF](https://www.ijcai.org/proceedings/2024/0673), [ConMark publisher](https://doi.org/10.1016/j.ins.2025.123030).

### WAVES

WAVES evaluates watermark detection and identification, attacks and image quality. Section 3 prioritizes TPR at a low FPR and explains why equal theoretical p-value thresholds do not necessarily give fair cross-method comparisons. Its performance-versus-quality analyses cover regeneration, distortion combinations and adaptive attacks. Accordingly, matched operating points, distortion measurement, multiple users and regeneration are established evaluation practices. The proposed extension must add a tested active/passive fusion objective and unmarked-harm analysis, rather than presenting these practices as novel.

Primary sources: [ICML 2024 proceedings](https://proceedings.mlr.press/v235/an24a.html), [full text](https://arxiv.org/html/2401.08573v2).

### Confidence-aware adaptive fusion

DeepFakeBuster combines heterogeneous passive detectors. Its fusion section derives validation-based global weights and input-specific weights from calibrated confidence, then averages detector predictions. This overlaps directly with reliability-aware weighting. The proposed distinction is a decoder-specific active channel, paired marked/unmarked constraints, source-level false-positive assessment, and separate expected-message verification. Generic adaptive weighting is not a contribution claim.

Primary source: [Scientific Reports](https://www.nature.com/articles/s41598-026-61247-6), DOI 10.1038/s41598-026-61247-6.

### Selective prediction and statistical risk control

SelectiveNet jointly optimizes prediction and rejection; the earlier selective-classification paper studies risk-versus-coverage control for pretrained networks. Abstention and risk-coverage curves are established. Neyman-Pearson classification already separates score fitting from a high-confidence type-I-error operating point. Learn then Test treats calibration as multiple hypothesis testing, including simultaneous risk constraints. Conformal Risk Control bounds expected monotone losses under its assumptions. These results are statistical building blocks, not original theory of this project. Text-watermark conformal calibration also predates this extension, so applying conformal cuts to a watermark score alone is insufficient novelty.

Primary sources: [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), [selective classification](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html), [Neyman-Pearson classification](https://doi.org/10.1126/sciadv.aao1659), [Learn then Test](https://arxiv.org/abs/2110.01052), [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html), [conformal text watermarking](https://hdsr.mitpress.mit.edu/pub/ifxlo12v/release/2).

### Local and multiple-message recovery

Watermark Anything already localizes marked regions and extracts different messages from them. Regional reliability, multiple-message extraction and crop recovery cannot be claimed as new merely by adding them. The present candidate uses existing decoders; any regional or multi-view feature requires explicit inference-time implementation and comparison.

Primary source: [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c5ee1911dffe4886367eee7fca314235-Abstract-Conference.html).

## Candidate distinction and honest claim

Working description: **paired unmarked-harm-constrained fusion of frozen passive and watermark evidence**. This is a description, not a claim of priority or a finalized method name.

The candidate combines four project-specific requirements:

1. Learn the contribution of a decoder from matched marked/unmarked derivatives of the same source, with common edits.
2. Optimize benefit on marked images while explicitly limiting harmful switches or excess loss on unmarked images at the final independently calibrated decision point.
3. Assess origin false positives and expected-message false matches separately, including unmarked AI and wrong-payload nulls for verification.
4. Keep a source as the statistical unit when a decision may search multiple views or registered payloads; report abstention and the cost of calibration.

The defensible contribution, if successful, is the implemented constrained objective, its paired evaluation, and demonstrated benefit over the local and literature-derived baselines. No first, universal, deployment-safe, issuer-authenticated or arbitrary-distribution-shift guarantee is supported by this design. A literature search can identify overlap; it cannot prove absence of all prior work.

## Concrete optimization

Let `z(x)` be the frozen passive logit, `m(x,q)` a soft expected-payload margin, and `v(x,q)` decoder reliability features. Use fitting sources only to fix feature scaling. A small candidate correction is

`s_theta(x,q) = z(x) + B * sigmoid(theta^T v(x,q)) * max(0, m(x,q) - u)`,

where `B >= 0` is bounded and `u` is a fitting-selected support cut. Keep `B=0` as an admissible fallback. This nonnegative formula itself is established in the local positive-evidence candidate; the substantive extension is its paired optimization and independently assessed constraints.

Recommended fitting objective: source-balanced logistic loss across marked and unmarked rows, plus regularization, with an explicit penalty or constraint on the source-group mean positive part of

`loss(y, s_theta(x_unmarked)) - loss(y, z(x_unmarked))`.

The marked benefit is measured against the same source's unmarked derivative and against the passive score of the received marked derivative; report both contrasts. The former can reveal mark-induced passive changes, whereas the latter isolates the added decision-layer benefit. Do not treat an unmarked clean-image score as the inference-time baseline for its marked or edited counterpart.

Use a small preregistered family of bounds, support cuts and regularization settings. Select it with source-grouped development folds and an independent selection portion. Hard decision constraints should be checked after fitting; a smooth loss penalty alone does not enforce a desired FPR. Record optimizer convergence, infeasibility and fallback to `B=0`.

### Valid reliability features

Candidate features include signed soft-margin mean, margin dispersion, hard agreement and disagreement between a fixed set of received-image views. Sigmoid entropy of raw decoder logits is only a confidence feature unless its calibration is validated; raw margins are not probabilities.

Every view must be generated from the actual received image, such as identity and two predeclared mild re-encodings. Do not calculate stability by looking up cached sibling edits, the clean original, another payload cover, ground-truth recovered bits, source identity or generator identity. Those are unavailable to the deployed rule and may leak reference information. If new received-image views are not decoded, describe the implementation as soft-margin fusion and report that the multi-view component is absent.

The fixed query set, view set and aggregation must be part of the protocol. Any maximum-over-query or maximum-over-view operation must occur inside each source's calibration statistic. Adding search candidates after calibration invalidates the stated family.

## Independent calibration and harm assessment

Use disjoint source sets for fitting, selection, risk calibration and final testing. Frozen backbones do not make reused head-training sources independent calibration data. Archive each split and all policy hashes before new inference.

Let `d0` be the passive decision using its fixed independent calibration, and `d_theta` the fusion decision using the equally specified calibration policy. Relevant source events are:

- Origin false positive: any declared photographic view receives an AI-positive decision.
- New photographic false positive: any view has `d0=0` and `d_theta=1`.
- Unmarked-AI harmful switch: any unmarked AI view has `d0=1` and `d_theta=0`.
- Verification false match: an unmarked or wrong-payload source passes any declared query/view. Wrong-payload positives remain AI-origin positives but expected-message negatives.

A nonnegative score correction at a shared threshold makes an AI harmful switch impossible by construction. That fact alone does not show preservation at a recalibrated equal-FPR operating point or preservation of AUC; independent fusion calibration can raise its threshold. Measure harmful switches and rank changes under the actual deployed policy.

For a fixed finite candidate family, independent source Bernoulli events can be assessed with one-sided binomial bounds and a prespecified simultaneous error allocation. Candidates or thresholds chosen using those data need simultaneous testing, a valid selection procedure, or another untouched risk-assessment split. Ordinary image-row binomial bounds ignore dependence among derivatives. Split-conformal marginal control is distinct from a high-probability bound on a fitted policy's population risk; label which guarantee is actually used.

For zero errors on `n` independent same-risk sources, the one-sided upper bound is `1 - delta^(1/n)`. Even one unadjusted 95% claim at a 1% tolerance requires at least 299 zero-error sources. Multiplicity or per-collection claims require more. If the sample is too small or the bound exceeds the target, record certification as failed or unavailable; do not substitute an empirical zero rate for a certificate. No such bound covers an arbitrary unseen generator or processing shift.

## Prospective success protocol

These are proposed targets to fix before any final test, not measured outcomes:

1. Primary task: AI-origin detection at a fixed 5% source-level nominal FPR using the same independently specified calibration rule for every method. Report realized FPR and uncertainty, not just the nominal target.
2. Primary benefit: marked-image source-balanced TPR improvement over passive and the strongest preregistered fusion baseline. Use paired source bootstrap intervals and an explicit multiplicity policy. A positive point estimate with an interval crossing zero is inconclusive.
3. Primary harm tolerance: no more than a predeclared one-percentage-point loss of unmarked-AI TPR, plus a separately declared photographic excess-FPR tolerance. If statistical support is insufficient, report noninferiority as unresolved. Do not choose tolerances after seeing the final result.
4. Scope: both existing watermarks; unmarked, correctly marked and wrong-payload inputs; several payloads/noise seeds; clean images, seen edits, and fixed unseen edit compositions. Keep low-resolution margin acceptance separate from full message/ECC recovery.
5. Transfer: target at least three held-out generator families and a second real collection. A checkpoint or accelerated sampler of a known family is a pipeline transfer test, not evidence of a new generator family. Report actual covered families if acquisition limits the target.
6. Baselines: frozen passive probe, original logistic fusion, 13/16 gate, simple soft/constant gate, local reliability mixture, local capped positive evidence, decoder-only margins and native message verification. Match source splits, inputs, calibration, fitting budget and operating point.
7. Ablations: remove paired harm terms, remove the cap, replace soft reliability with agreement, remove stability if implemented, omit wrong-payload nulls, and compare image-only with source-max calibration. Failed or zero-gain ablations remain reportable.
8. Selective outputs: fix abstention policy from development/calibration only. Report risk-versus-coverage, retained counts by origin/generator/edit, and full-set performance. Abstaining on difficult items is not the same as improving full-set detection.
9. Final lock: archive code, weights, feature scales, queries, views, risk budgets, success criteria, split/sample manifests and hashes. Generate or collect new sources after the method lock where feasible. Execute final inference once and publish failures as well as benefits. Earlier observed samples remain development or diagnostic evidence.

An 8/10 or 9/10 novelty rating is not an output of this protocol. If the candidate gives negligible gain, is equivalent to a simpler baseline, or requires unsupported risk claims, retain an empirical evaluation framing and do not claim the upgrade succeeded.

## Bibliography and inspection limitations

`docs/phase6_related_work.bib` records primary-source metadata checked during this review. AdvMark and WAVES full text, ImageDetectBench arXiv v2, DeepFakeBuster's publisher fusion section, and the cited selective/risk-control primary records were inspected. ConMark's publisher summary was available through search, while direct full-text access was blocked. The Learn then Test citation uses its verified arXiv version rather than unverified journal page metadata. The reference file does not automatically change the open manuscript.
