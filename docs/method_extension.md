# Exploratory reliability-gated fusion extension

This extension is an actual fitted method and comparison experiment. It was
designed after the existing test results were examined, so its design is
retrospective. Source-grouped validation controls subsequent parameter
selection; it does not turn the existing test into a fresh untouched benchmark.
No claim that mixture-of-experts gating itself is novel is justified here.

## Method

For a frozen origin probe, let `l` be its linear decision score, `p` its
probability, and `a` the agreement with the expected watermark bits. Fit two
class-balanced logistic experts on the corrected validation images containing
only seen transformation settings:

- Passive calibration: `P = sigmoid(alpha_0 + alpha_1*l)`.
- Static fusion: `F = sigmoid(beta_0 + beta_1*p + beta_2*a)`.

The final score is a smooth convex mixture `M = (1-q)*P + q*F`. A learned gate
uses `q = sigmoid(theta_0 + theta_1*r + theta_2*u + theta_3*r*u)`, with
`r = 2*(a-0.5)` and `u = min(abs(l)/s, 3)`. The scale `s` is the 90th percentile
of absolute logits in the gate's fitting partition, bounded below by one.
Larger `u` represents greater passive certainty. The gate learns conditional
usefulness of the two scores; `q` is a mixture weight and is not a calibrated
probability that the watermark survived or that an image is authentic.

The gate minimizes class-balanced binary log loss plus an L2 penalty on its
three non-intercept coefficients. The experts are fitted first and held fixed
while optimizing the gate. This is a low-dimensional calibration extension,
without new backbone training or a new watermark security mechanism.

## Selection and isolation

Five folds group all variants of a source together, and distribute sources
within each seen collection across folds. Each fold independently fits expert
parameters, the certainty scale, and gate parameters using the other four
folds. The regularization candidates are 0.001, 0.01, 0.1 and 1.0. Selection
uses held-source class-balanced log loss; the held-source AUC is descriptive.
The selected method is refitted on all eligible validation sources, and each
system's probability threshold maximizes balanced accuracy on that validation
set. Final experts, gate parameters, selected hyperparameters, source folds
and thresholds are saved before the script loads test predictions.

Midjourney and the `lowres:2`, `regenerate:2`, and `spectral:1` settings are
excluded from all fitting and selection. Attack names and generator labels
never enter the gate; generator labels are used solely to balance validation
folds. The compound transformation named `regenerate` in the original cache
is a grid shift, JPEG and noise, not diffusion regeneration. A separate
October 9 benchmark now executes actual DDIM reconstruction and a VAE-only
control; its artifacts and verification criteria are in extension_run.md.

Exact validation logits are recovered from the archived coefficients and
unchanged frozen embedding caches using float64 linear arithmetic. Probability
clipping cannot be inverted. Ranking of each fitted expert uses its linear
decision score. Mixture ranking is evaluated with log-sum-exp arithmetic on
expert and gate logits, avoiding artificial sigmoid-saturation ties. Brier
scores and fixed decisions continue to use actual probabilities.

## Comparisons that can falsify the proposed benefit

The experiment retains the original frozen probe and original static fusion.
It also measures passive calibration, a refitted static fusion expert, a
stronger logistic baseline using passive logit and payload agreement, and a
constant mixture whose single mixture weight minimizes validation loss.
The constant mixture tests whether a global blend already explains any gain.

Two gate ablations remove passive certainty (payload-only gate) or payload
agreement (certainty-only gate). Both still use the same two experts; removing
a gate input does not remove watermark information from the fusion expert.

The archived protocol asks whether the full mixture has higher full-test AUC
than the passive probe and whether its AUC on the prior reduced-recovery slice
is within 0.005 of passive AUC. That slice contains crop, 32x32 round trip,
compound grid shift and spectral wipe. It is a previously identified,
retrospective diagnostic slice, not a newly preregistered population.
The method is also compared against the logit-plus-payload and constant-mixture
baselines. Improvement on one representation or slice must not be generalized
to every representation, generator or removal method.

Two thousand bootstrap draws resample sources within each collection,
bringing all selected derivatives together. Paired AUC differences use the
same draws for both scores. These intervals condition on fitted models and
the screened collection, and exclude fitting-seed uncertainty. Test-ROC TPR
at 5% FPR remains a descriptive ROC point; fixed validation thresholds have
their own test FPR and TPR.

## Reproduction and artifacts

Run `scripts/experimental_reliability.py --bootstrap-samples 2000` in the
project's research environment. It reads `outputs/conference_clean/` and the
original frozen caches, and writes only `outputs/extension/reliability/`.
`protocol.json` records the design before test access; `validation_folds.csv`
records source isolation; `selection.json` records all candidate losses and
thresholds; fitted NPZ files retain expert and gate parameters;
`test_predictions.csv` records every system; and `results.json` records full,
reduced-recovery, held-out attack, held-out generator, and clean-unmarked
comparisons. Original runs and the manuscript are preserved.

This method does not resolve collection or encoding confounds, establish
cryptographic provenance, or measure a stronger watermark or diffusion
reconstruction. Those require separate data controls and experiments. A
paper could present a useful empirical contribution if comparisons support
it, but algorithmic novelty and conference competitiveness still require
related-work assessment and external evidence.

## Completed experiment and negative findings

The full two-thousand-draw experiment completed for both representations.
All gate variants selected regularization 0.001 using source-held-out
validation loss. The numerical bootstrap implementation was checked against
scikit-learn on one hundred integer-weighted, tied-score examples. Stable
mixture log-odds were checked against the predicted probabilities. All expert
and gate fits converged. These checks support arithmetic and execution;
they do not validate generalization to new collections.

For ResNet, the learned mixture reaches full-test AUC 0.9695, versus 0.9427
for the passive probe and 0.9646 for the original static fusion. Its paired
gain over original fusion is 0.00485, with source-bootstrap interval
[0.00402, 0.00572]. However, on reduced-recovery images its AUC is 0.9026,
below passive AUC 0.9223. The paired difference from passive is -0.01974
[-0.02459, -0.01501]. It fails the declared 0.005 preservation criterion.

For CLIP, the mixture improves full-test AUC from passive 0.9940 to 0.9957.
Its reduced-recovery AUC is 0.9724, below passive 0.9884; the paired
difference is -0.01602 [-0.02149, -0.01126]. It also fails the preservation
criterion. The learned gates continue to give substantial weight to the
fusion expert under weak payload recovery, so a smooth gate alone does not
solve the failure being studied. The weaker gate ablations do not reverse
that conclusion.

A simpler, stronger comparator using passive logit plus payload agreement
outperforms the learned mixture on the full test: AUC 0.9719 for ResNet
and 0.9974 for CLIP. For CLIP it retains reduced-recovery AUC 0.9855,
with difference -0.00298 [-0.00391, -0.00205] from passive. For ResNet
its reduced-recovery AUC remains 0.9018, below passive. Fixed-threshold
errors must accompany these AUCs: for example, the CLIP logit-plus-payload
comparator's FPR is 0.0260 overall and 0.0766 on reduced-recovery images.

The defensible conclusion is an empirical comparison and a negative result:
the learned mixture improves the original probability-based fusion but does
not preserve passive ranking when payload recovery is poor, and it is beaten
overall by a simpler comparator. It should not be advertised as a solved
reliability mechanism or a superior novel detector. A subsequent constrained
or risk-sensitive gate would be a new retrospective design step and should
be evaluated with a genuinely separate external test rather than silently
tuned to this diagnostic slice.
