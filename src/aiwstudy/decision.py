from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DECISIONS = ("AI_MARKED", "AI_FORENSIC", "REAL", "ABSTAIN")
POSITIVE = {"AI_MARKED", "AI_FORENSIC"}


@dataclass(frozen=True)
class DecisionPolicy:
    """Abstaining origin policy fitted only on validation images.

    If the expected payload is recovered, the policy labels the image marked
    AI. Otherwise fusion is used, with abstention inside its gray band.
    The band is chosen to keep matched real-image attacks from inflating the
    false-positive rate (the RobustSora shortcut).
    """

    tau_active: float
    margin: float
    midpoint: float = 0.5

    def decide(self, active: float, fusion: float) -> str:
        if active >= self.tau_active:
            return "AI_MARKED"
        if fusion >= self.midpoint + self.margin:
            return "AI_FORENSIC"
        if fusion <= self.midpoint - self.margin:
            return "REAL"
        return "ABSTAIN"

    def decide_many(self, active: np.ndarray, fusion: np.ndarray) -> np.ndarray:
        return np.asarray([self.decide(float(a), float(f)) for a, f in zip(active, fusion)])


def _false_positive_rate(frame: pd.DataFrame, group: str) -> float:
    subset = frame[frame["group"] == group]
    if subset.empty:
        return float("nan")
    return float((subset["decision"].isin(POSITIVE)).mean())


def _balanced_accuracy(frame: pd.DataFrame) -> float:
    decided = frame[frame["decision"] != "ABSTAIN"]
    if decided.empty or decided["label"].nunique() < 2:
        return 0.0
    pred = decided["decision"].isin(POSITIVE).to_numpy()
    labels = decided["label"].to_numpy()
    tpr = pred[labels == 1].mean() if (labels == 1).any() else 0.0
    tnr = (~pred[labels == 0]).mean() if (labels == 0).any() else 0.0
    return float((tpr + tnr) / 2.0)


def fit_decision_policy(
    validation: pd.DataFrame,
    max_real_fpr: float = 0.05,
    min_coverage: float = 0.70,
    shortcut_slack: float = 0.02,
) -> DecisionPolicy:
    if validation[validation["label"] == 0].empty:
        raise ValueError("Validation must include real images to set the watermark FAR.")

    best: tuple[float, float, float, float, float] | None = None
    fallback: tuple[float, float, float, float, float] | None = None
    for tau_active in np.linspace(0.75, 0.98, 12):
        for margin in np.linspace(0.02, 0.40, 20):
            policy = DecisionPolicy(tau_active=float(tau_active), margin=float(margin))
            scored = validation.copy()
            scored["decision"] = policy.decide_many(
                scored["active_score"].to_numpy(), scored["fusion_score"].to_numpy()
            )
            coverage = float((scored["decision"] != "ABSTAIN").mean())
            edited = _false_positive_rate(scored, "R-T")
            clean = _false_positive_rate(scored, "R-C")
            shortcut = 0.0 if np.isnan(edited) or np.isnan(clean) else float(edited - clean)
            edited_rate = 0.0 if np.isnan(edited) else float(edited)
            accuracy = _balanced_accuracy(scored)
            # Prefer a small edited-real gap, then accuracy, then coverage.
            rank = (-max(shortcut, 0.0), -max(edited_rate - max_real_fpr, 0.0), accuracy, coverage)
            candidate = (*rank, float(tau_active), float(margin))
            if fallback is None or candidate[:4] > fallback[:4]:
                fallback = candidate
            if (
                coverage + 1e-9 >= min_coverage
                and shortcut <= shortcut_slack + 1e-9
                and edited_rate <= max_real_fpr + 1e-9
            ):
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
    chosen = best if best is not None else fallback
    return DecisionPolicy(tau_active=chosen[-2], margin=chosen[-1])


def decision_metrics(frame: pd.DataFrame, policy: DecisionPolicy) -> dict:
    scored = frame.copy()
    scored["decision"] = policy.decide_many(
        scored["active_score"].to_numpy(), scored["fusion_score"].to_numpy()
    )
    decided = scored[scored["decision"] != "ABSTAIN"]
    counts = {name: int((scored["decision"] == name).sum()) for name in DECISIONS}
    report: dict = {
        "tau_active": policy.tau_active,
        "margin": policy.margin,
        "coverage": float(len(decided) / len(scored)) if len(scored) else float("nan"),
        "counts": counts,
        "balanced_accuracy_decided": _balanced_accuracy(scored),
        "group_positive_rate": {},
        "artifact_shortcut": {},
    }
    for group in ("R-C", "R-T", "R-S", "S-U", "S-W", "S-A"):
        subset = scored[scored["group"] == group]
        if subset.empty:
            continue
        report["group_positive_rate"][group] = {
            "n": int(len(subset)),
            "positive_rate": float(subset["decision"].isin(POSITIVE).mean()),
            "abstain_rate": float((subset["decision"] == "ABSTAIN").mean()),
            "marked_rate": float((subset["decision"] == "AI_MARKED").mean()),
            "forensic_rate": float((subset["decision"] == "AI_FORENSIC").mean()),
        }
    rc = report["group_positive_rate"].get("R-C", {}).get("positive_rate", float("nan"))
    rt = report["group_positive_rate"].get("R-T", {}).get("positive_rate", float("nan"))
    report["artifact_shortcut"] = {
        "delta_positive_rate_rt_minus_rc": float(rt - rc) if np.isfinite(rc) and np.isfinite(rt) else float("nan"),
        "spoof_accept_rate": report["group_positive_rate"].get("R-S", {}).get("positive_rate", float("nan")),
        "spoof_marked_rate": report["group_positive_rate"].get("R-S", {}).get("marked_rate", float("nan")),
    }
    return report
