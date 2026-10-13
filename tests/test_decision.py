import pandas as pd

from aiwstudy.decision import DecisionPolicy, decision_metrics, fit_decision_policy


def test_keyed_payload_beats_forensic_when_verified():
    policy = DecisionPolicy(tau_active=0.8, margin=0.1)
    assert policy.decide(0.95, 0.2) == "AI_MARKED"
    assert policy.decide(0.1, 0.9) == "AI_FORENSIC"
    assert policy.decide(0.1, 0.1) == "REAL"
    assert policy.decide(0.1, 0.5) == "ABSTAIN"


def test_policy_keeps_matched_real_attacks_from_inflating_fpr():
    rows = []
    for index in range(40):
        rows.append(
            {
                "label": 0,
                "group": "R-C",
                "active_score": 0.45,
                "fusion_score": 0.12,
            }
        )
        rows.append(
            {
                "label": 0,
                "group": "R-T",
                "active_score": 0.45,
                "fusion_score": 0.48 if index < 20 else 0.18,
            }
        )
        rows.append(
            {
                "label": 1,
                "group": "S-W",
                "active_score": 0.95,
                "fusion_score": 0.9,
            }
        )
    validation = pd.DataFrame(rows)
    policy = fit_decision_policy(validation, max_real_fpr=0.05, min_coverage=0.5, shortcut_slack=0.02)
    metrics = decision_metrics(validation, policy)
    shortcut = metrics["artifact_shortcut"]["delta_positive_rate_rt_minus_rc"]
    assert metrics["coverage"] >= 0.5
    assert shortcut <= 0.02 + 1e-6
