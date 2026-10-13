# Additional simple comparisons before the final test

Two secondary comparators were added while final test covers were still being
encoded, before `test/inference_started.json` and before any final test detector
or watermark score. Independent photographic calibration had already been
computed. This addition responds to a comparator gap, not a favorable test
finding. It changes neither the proposed method, original strongest eligible
comparator, nor the confirmatory endpoints.

**Complete-message union.** Each received image gets score 2 if any of the same
three registered queries completely recovers the expected native message,
otherwise the sigmoid of its frozen passive logit. TrustMark requires ECC,
schema and the entire padded message; classical recovery is its full16bits.
Each system has its own strict source-max photographic cut under the same
four-view, per-collection rule. If full-message false matches become too common
in calibration, tied score2 may force a conservative cut. Native recovery does
not authenticate an issuer, and a deliberately forged watermark remains an
untested threat.

**Confidence weighting with equal bound range.** Add
`B*max_query((2*agreement-1)_+*received_view_stability)` to the passive logit,
using `B={1,2,4,8,16}`. Selection uses only the existing tuning sources, the
same source-FPR and per-generator/view unmarked-loss constraints, greatest
marked recall, and then smallest bound. If none is feasible, use the probe.
This avoids comparing the proposed16bound only with a fixed2weight.

`scripts/phase6_additional_baselines.py --stage calibrate` seals source/model/
implementation hashes, all tuning candidates and independent cuts before final
test scores. Evaluation reuses the completed final feature cache with the same
2000paired, collection-stratified source draws. These comparisons are secondary
and cannot replace the fixed primary comparator. Their superiority or failure
must still influence the interpretation of method novelty.
