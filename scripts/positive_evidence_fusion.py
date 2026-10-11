"""Frozen positive-evidence fusion candidate; old-test analysis is retrospective.

Parameter selection is source-grouped and uses corrected seen validation only.
The score adds no watermark residual below its fitted agreement threshold.
The matched-real FPR constraint is empirical, not a population-risk guarantee.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiwstudy.evaluation import select_threshold
from experimental_reliability import (
    append_exact_validation_logits, balanced_loss, balanced_weights, digest,
    fit_logistic, grouped_folds, grouped_summary,
)

RIDGES = [0.0, 0.0001, 0.001, 0.01, 0.1]
HELD_OUT = {("lowres", 2), ("regenerate", 2), ("spectral", 1)}
RISK = 0.05
BITS = 16


def empirical_risk_threshold(scores: np.ndarray, risk: float) -> float:
    """Smallest conservative threshold allowing at most floor(risk*n) errors."""
    scores = np.sort(np.asarray(scores, dtype=float))
    if not len(scores):
        raise ValueError("Real controls are required for empirical risk selection.")
    allowed = int(np.floor(risk * len(scores)))
    return float(np.nextafter(scores[len(scores) - allowed - 1], np.inf))


class PositiveEvidence:
    def __init__(self, regularization: float, cap_risk: bool = True, threshold_support: bool = True):
        self.regularization = float(regularization)
        self.cap_risk, self.threshold_support = cap_risk, threshold_support

    def evidence(self, agreement: np.ndarray) -> np.ndarray:
        if not self.threshold_support:
            return np.asarray(agreement, dtype=float)
        # Below the validation threshold there is exactly zero residual.
        # First supported bit count receives one unit of a normalized bit ramp.
        a = np.asarray(agreement, dtype=float)
        return np.where(a >= self.payload_threshold, (a - self.support_floor) / (1 - self.support_floor), 0.0)

    def fit(self, logit: np.ndarray, active: np.ndarray, labels: np.ndarray, groups: np.ndarray):
        self.base = fit_logistic(logit[:, None], labels)
        self.base_coefficient = float(self.base.coef_[0, 0])
        self.base_intercept = float(self.base.intercept_[0])
        if self.base_coefficient <= 0:
            raise RuntimeError("Passive calibration must preserve original score ranking.")
        self.payload_threshold = select_threshold(labels, active)
        self.support_floor = (int(np.ceil(self.payload_threshold * BITS)) - 1) / BITS
        if not 0 <= self.support_floor < 1:
            raise RuntimeError("The validation payload threshold cannot define a bit ramp.")
        base_scores = self.base_intercept + self.base_coefficient * logit
        real_masks = {name: (labels == 0) & (groups == name) for name in ("R-C", "R-T", "R-S")}
        real_masks = {name: mask for name, mask in real_masks.items() if mask.any()}
        real_masks["all_real"] = labels == 0
        self.risk_threshold = max(empirical_risk_threshold(base_scores[mask], RISK) for mask in real_masks.values())
        evidence = self.evidence(active)
        upper = 40.0
        if self.cap_risk:
            for mask in real_masks.values():
                score, h = base_scores[mask], evidence[mask]
                allowed = int(np.floor(RISK * len(score)))
                existing = int(np.sum(score >= self.risk_threshold))
                candidates = (h > 0) & (score < self.risk_threshold)
                crossings = np.sort((self.risk_threshold - score[candidates]) / h[candidates])
                remaining = allowed - existing
                if remaining < 0:
                    raise RuntimeError("Passive empirical risk threshold is infeasible.")
                if len(crossings) > remaining:
                    upper = min(upper, max(float(crossings[remaining]) * (1 - 1e-10), 0.0))
        self.coefficient_cap = upper
        weights = balanced_weights(labels)

        def objective(coefficient):
            z = base_scores + coefficient * evidence
            # Stable logistic loss; no sigmoid clipping is needed.
            return float(np.sum(weights * (np.logaddexp(0, z) - labels * z)) + self.regularization * coefficient ** 2 / 2)

        if upper <= 1e-12:
            optimum, success = 0.0, True
        else:
            optimized = minimize_scalar(objective, bounds=(0, upper), method="bounded", options={"xatol": 1e-10})
            if not optimized.success:
                raise RuntimeError("Positive-evidence coefficient did not converge.")
            choices = [0.0, upper, float(optimized.x)]
            optimum = min(choices, key=objective)
            success = bool(optimized.success)
        self.positive_coefficient = float(optimum)
        self.fit_diagnostics = {"optimizer_converged": success, "coefficient": optimum, "coefficient_cap": upper,
            "payload_threshold": self.payload_threshold, "risk_threshold_logit": self.risk_threshold,
            "empirical_real_fpr": {name: float(np.mean((base_scores + optimum * evidence)[mask] >= self.risk_threshold)) for name, mask in real_masks.items()}}
        if self.cap_risk and any(rate > RISK + 1e-12 for rate in self.fit_diagnostics["empirical_real_fpr"].values()):
            raise RuntimeError("Numerical rounding violated an empirical FPR constraint.")
        return self

    def decision_function(self, logit: np.ndarray, active: np.ndarray) -> np.ndarray:
        return self.base_intercept + self.base_coefficient * logit + self.positive_coefficient * self.evidence(active)

    def save(self, path: Path):
        np.savez(path, regularization=self.regularization, base_coefficient=self.base_coefficient,
                 base_intercept=self.base_intercept, positive_coefficient=self.positive_coefficient,
                 payload_threshold=self.payload_threshold, support_floor=self.support_floor,
                 risk_threshold=self.risk_threshold, risk=RISK, bits=BITS,
                 cap_risk=self.cap_risk, threshold_support=self.threshold_support)

    @classmethod
    def load(cls, path: Path):
        with np.load(path) as values:
            result = cls(float(values["regularization"]), bool(values["cap_risk"]), bool(values["threshold_support"]))
            for name in ("base_coefficient", "base_intercept", "positive_coefficient", "payload_threshold", "support_floor", "risk_threshold"):
                setattr(result, name, float(values[name]))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--freeze-only", action="store_true", help="Fit/archive the candidate without reading old test predictions")
    args = parser.parse_args()
    corrected = ROOT / "outputs" / "conference_clean"
    destination = ROOT / "outputs" / "extension" / "positive_evidence"
    destination.mkdir(parents=True, exist_ok=True)
    validation_path = corrected / "validation_predictions.csv"
    validation = pd.read_csv(validation_path, keep_default_na=False)
    if any(key in HELD_OUT for key in zip(validation["attack"], validation["severity"])) or (validation["generator"] == "midjourney").any():
        raise RuntimeError("Held-out images entered positive-evidence fitting.")
    protocol = {"retrospective_design": True, "old_test_status": "Already inspected before this candidate was conceived; diagnostic only.",
        "external_evaluation_requirement": "Freeze candidate before producing or inspecting new paired-content and encoding/removal evaluation scores.",
        "score": "calibrated passive logit + nonnegative coefficient * supported payload bit ramp",
        "support": "Exactly zero below a payload threshold chosen from eligible validation only.",
        "regularization_candidates": RIDGES, "selection": "5-fold source-grouped validation balanced logistic loss",
        "risk_constraint": "Empirical FPR <= .05 at a passive validation threshold, separately on clean, edited, opposite-payload and all real controls; no population or shifted-distribution guarantee.",
        "bits": BITS, "source_hash": digest(validation_path), "script_hash": digest(Path(__file__)),
        "helper_script_hash": digest(ROOT / "scripts" / "experimental_reliability.py"),
        "literature_basis": [
            {"paper": "Wolpert, Stacked generalization, 1992", "url": "https://www.sciencedirect.com/science/article/pii/S0893608005800231", "scope": "Supervised score combination is established."},
            {"paper": "Zadrozny and Elkan, Transforming classifier scores into accurate multiclass probability estimates, 2002", "url": "https://www.cs.columbia.edu/~djhsu/coms4771-f25/handouts/zadrozny2002kdd.pdf", "scope": "Score calibration and combination are established."},
            {"paper": "Tong, Feng and Li, Neyman-Pearson classification algorithms and NP receiver operating characteristics, 2018", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/", "scope": "An empirical FPR constraint alone does not establish population type-I-error control."}],
        "novelty_claim": "None for the formula; potential empirical contribution is matched-control evaluation and externally tested constrained positive-evidence fusion versus stronger baselines."}
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    validation = append_exact_validation_logits(validation, corrected)
    folds = grouped_folds(validation)
    validation.assign(fold=folds)[["image_id", "source_id", "generator", "attack", "severity", "fold"]].to_csv(destination / "validation_folds.csv", index=False)
    models, selections, thresholds, baselines = {}, {}, {}, {}
    for backbone, probability in (("resnet", "probe"), ("clip", "clip_probe")):
        logit = validation[probability + "_logit"].to_numpy(float)
        active = validation["active_score"].to_numpy(float)
        labels, groups = validation["label"].to_numpy(int), validation["group"].to_numpy(str)
        candidates = []
        for ridge in RIDGES:
            oof_logits, oof_decisions = np.empty(len(validation)), np.empty(len(validation), dtype=bool)
            fold_details = []
            for fold in range(5):
                fit, held = folds != fold, folds == fold
                model = PositiveEvidence(ridge).fit(logit[fit], active[fit], labels[fit], groups[fit])
                oof_logits[held] = model.decision_function(logit[held], active[held])
                oof_decisions[held] = oof_logits[held] >= model.risk_threshold
                fold_details.append(model.fit_diagnostics)
            loss = float(np.sum(balanced_weights(labels) * (np.logaddexp(0, oof_logits) - labels * oof_logits)))
            candidates.append({"regularization": ridge, "held_source_balanced_log_loss": loss,
                "held_source_fixed_fpr": float(oof_decisions[labels == 0].mean()), "fit_fold_diagnostics": fold_details})
        best = min(candidates, key=lambda record: (record["held_source_balanced_log_loss"], -record["regularization"]))
        variants = {"positive_evidence": PositiveEvidence(best["regularization"]),
                    "no_risk_cap": PositiveEvidence(best["regularization"], cap_risk=False),
                    "no_support_gate": PositiveEvidence(best["regularization"], threshold_support=False)}
        for name, model in variants.items():
            model.fit(logit, active, labels, groups)
            model.save(destination / f"{backbone}_{name}.npz")
            models[backbone, name] = model
        center, scale = float(logit.mean()), max(float(logit.std()), 1e-8)
        strong = fit_logistic(np.column_stack([(logit - center) / scale, active]), labels)
        np.savez(destination / f"{backbone}_logit_payload_baseline.npz", coefficients=strong.coef_, intercept=strong.intercept_, center=center, scale=scale)
        baselines[backbone] = strong, center, scale
        primary = models[backbone, "positive_evidence"]
        validation_scores = {name: model.decision_function(logit, active) for name, model in variants.items()}
        validation_scores["calibrated_passive"] = primary.base_intercept + primary.base_coefficient * logit
        validation_scores["logit_payload_fusion"] = strong.decision_function(np.column_stack([(logit - center) / scale, active]))
        thresholds[backbone] = {name: float(expit(model.risk_threshold)) for name, model in variants.items()}
        thresholds[backbone]["calibrated_passive"] = float(expit(primary.risk_threshold))
        thresholds[backbone]["logit_payload_fusion"] = select_threshold(labels, expit(validation_scores["logit_payload_fusion"]))
        thresholds[backbone]["probe"] = select_threshold(labels, validation[probability].to_numpy())
        original = "fusion" if backbone == "resnet" else "clip_fusion"
        thresholds[backbone]["original_fusion"] = select_threshold(labels, validation[original].to_numpy())
        # grouped_summary requires a constant-mixture reference; this is explicitly
        # the previously frozen negative-extension constant-mixture baseline.
        with np.load(ROOT / "outputs" / "extension" / "reliability" / f"{backbone}_full.npz") as old:
            pz = (logit[:, None] @ old["passive_coefficients"].T + old["passive_intercept"]).ravel()
            fz = (np.column_stack([validation[probability].to_numpy(), active]) @ old["fusion_coefficients"].T + old["fusion_intercept"]).ravel()
            q = float(old["constant_gate"])
            validation_constant = (1-q) * expit(pz) + q * expit(fz)
        thresholds[backbone]["constant_mixture"] = select_threshold(labels, validation_constant)
        selections[backbone] = {"selected_regularization": best["regularization"], "candidates": candidates,
            "models": {name: model.fit_diagnostics for name, model in variants.items()}}
        print(backbone, "frozen candidate", primary.fit_diagnostics, flush=True)
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "selection": selections,
        "thresholds_probability": thresholds, "test_was_not_read_during_this_candidate_fit": True,
        "npz_sha256": {path.name: digest(path) for path in sorted(destination.glob("*.npz"))}}
    (destination / "frozen_selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("FROZEN", frozen["frozen_utc"], destination, flush=True)
    if args.freeze_only:
        return
    test = pd.read_csv(corrected / "test_predictions.csv", keep_default_na=False)
    if set(test["source_id"]) & set(validation["source_id"]):
        raise RuntimeError("Validation/test sources overlap.")
    output = {"frozen_selection": frozen, "old_test_is_diagnostic_only": True, "results": {}}
    predictions = test[["image_id", "source_id", "label", "generator", "attack", "severity", "group"]].copy()
    for backbone, probability in (("resnet", "probe"), ("clip", "clip_probe")):
        logit, active = test[probability + "_logit"].to_numpy(float), test["active_score"].to_numpy(float)
        primary = models[backbone, "positive_evidence"]
        ranks = {name: models[backbone, name].decision_function(logit, active) for name in ("positive_evidence", "no_risk_cap", "no_support_gate")}
        ranks["calibrated_passive"] = primary.base_intercept + primary.base_coefficient * logit
        strong, center, scale = baselines[backbone]
        ranks["logit_payload_fusion"] = strong.decision_function(np.column_stack([(logit-center)/scale, active]))
        ranks["probe"] = logit
        original = "fusion" if backbone == "resnet" else "clip_fusion"
        ranks["original_fusion"] = test[original + "_logit"].to_numpy(float)
        with np.load(ROOT / "outputs" / "extension" / "reliability" / f"{backbone}_full.npz") as old:
            pz = (logit[:, None] @ old["passive_coefficients"].T + old["passive_intercept"]).ravel()
            fz = (np.column_stack([test[probability].to_numpy(), active]) @ old["fusion_coefficients"].T + old["fusion_intercept"]).ravel()
            q = float(old["constant_gate"])
            logpos = np.logaddexp(np.log1p(-q)-np.logaddexp(0,-pz), np.log(q)-np.logaddexp(0,-fz))
            logneg = np.logaddexp(np.log1p(-q)-np.logaddexp(0,pz), np.log(q)-np.logaddexp(0,fz))
            ranks["constant_mixture"] = logpos-logneg
        scores = {name: expit(rank) for name, rank in ranks.items()}
        scores["probe"] = test[probability].to_numpy(float)
        scores["original_fusion"] = test[original].to_numpy(float)
        masks = {"full_test": np.ones(len(test), bool),
            "reduced_recovery": ((test["attack"] == "crop") | ((test["attack"] == "lowres") & (test["severity"] == 2)) | test["attack"].isin(["regenerate", "spectral"])).to_numpy(),
            "weak_payload_support": active < primary.payload_threshold,
            "held_out_attacks": np.array([key in HELD_OUT for key in zip(test["attack"], test["severity"])]),
            "held_out_generator": test["generator"].isin(["real", "midjourney"]).to_numpy(),
            "clean_unmarked": test["group"].isin(["R-C", "S-U"]).to_numpy()}
        results = {}
        for name, mask in masks.items():
            results[name] = grouped_summary(test.loc[mask], {key:value[mask] for key,value in scores.items()}, {key:value[mask] for key,value in ranks.items()}, thresholds[backbone], args.bootstrap_samples)
            print(backbone, name, {key:round(value["roc_auc"],4) for key,value in results[name]["systems"].items()}, flush=True)
        results["exact_fallback_identity"] = bool(np.array_equal(ranks["positive_evidence"][masks["weak_payload_support"]], ranks["calibrated_passive"][masks["weak_payload_support"]]))
        results["fraction_with_supported_payload"] = float((active >= primary.payload_threshold).mean())
        output["results"][backbone] = results
        for name, rank in ranks.items():
            predictions[f"{backbone}_{name}_logit"] = rank
    predictions.to_csv(destination / "diagnostic_test_predictions.csv", index=False)
    (destination / "diagnostic_old_test_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
