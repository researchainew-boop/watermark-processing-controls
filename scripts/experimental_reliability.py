"""Retrospective reliability-gated fusion extension using frozen cached fits.

Only the corrected, seen-condition validation rows fit/select the extension.
All test labels are read after choices and fitted models have been archived.
No original predictions, fits or manuscript files are overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiwstudy.evaluation import classification_metrics, select_threshold

SEED = 20261008
RIDGES = [0.001, 0.01, 0.1, 1.0]
GATE_KINDS = ["full", "payload_only", "uncertainty_only"]
HELD_OUT = {("lowres", 2), ("regenerate", 2), ("spectral", 1)}


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2)
    if not counts.all():
        raise ValueError("Both outcome classes are required.")
    return 1 / (2 * counts[labels])


def balanced_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(-np.sum(balanced_weights(labels) * (labels * np.log(p) + (1 - labels) * np.log1p(-p))))


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=SEED)
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features.astype(np.float64), labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in found):
        raise RuntimeError("A calibration expert did not converge.")
    return model


def grouped_folds(frame: pd.DataFrame, folds: int = 5) -> np.ndarray:
    sources = frame[["source_id", "generator", "label"]].drop_duplicates("source_id")
    rng = np.random.default_rng(SEED)
    assignments = {}
    for _, subset in sources.groupby("generator", sort=True):
        ids = rng.permutation(subset["source_id"].to_numpy())
        assignments.update({str(source): i % folds for i, source in enumerate(ids)})
    return frame["source_id"].astype(str).map(assignments).to_numpy(dtype=int)


def append_exact_validation_logits(frame: pd.DataFrame, corrected: Path) -> pd.DataFrame:
    result = frame.copy()
    for backbone, probability in (("resnet", "probe"), ("clip", "clip_probe")):
        cache = ROOT / "outputs" / ("resnet_embeddings.npz" if backbone == "resnet" else "clip_eval_embeddings.npz")
        with np.load(cache, allow_pickle=True) as values:
            ids = values["image_id"].astype(str)
            lookup = {key: index for index, key in enumerate(ids)}
            indices = [lookup[str(key)] for key in frame["image_id"]]
            vectors = values["embedding"][indices].astype(np.float64)
        with np.load(corrected / f"{backbone}_full.npz") as model:
            result[probability + "_logit"] = (vectors @ model["coefficients"].astype(float).T + model["intercept"].astype(float)).ravel()
        del vectors
    return result


class ReliabilityMixture:
    """Two calibrated experts and a smooth, regularized, learned gate.

    The gate never receives attack identities or origin/generator labels.
    It can depend on expected-payload agreement and passive-score uncertainty.
    """
    def __init__(self, ridge: float, kind: str = "full"):
        self.ridge, self.kind = float(ridge), kind

    def gate_features(self, logit: np.ndarray, active: np.ndarray) -> np.ndarray:
        agreement = 2 * (active - 0.5)
        uncertainty = np.clip(np.abs(logit) / self.logit_scale, 0, 3)
        if self.kind == "payload_only":
            columns = [np.ones(len(logit)), agreement]
        elif self.kind == "uncertainty_only":
            columns = [np.ones(len(logit)), uncertainty]
        else:
            columns = [np.ones(len(logit)), agreement, uncertainty, agreement * uncertainty]
        return np.column_stack(columns)

    def fit(self, logit: np.ndarray, probability: np.ndarray, active: np.ndarray, labels: np.ndarray):
        self.logit_scale = max(float(np.percentile(np.abs(logit), 90)), 1.0)
        self.passive = fit_logistic(logit[:, None], labels)
        self.fusion = fit_logistic(np.column_stack([probability, active]), labels)
        passive = self.passive.predict_proba(logit[:, None])[:, 1]
        fusion = self.fusion.predict_proba(np.column_stack([probability, active]))[:, 1]
        features = self.gate_features(logit, active)
        weights = balanced_weights(labels)

        def objective(theta):
            gate = expit(features @ theta)
            mixed = np.clip(passive + gate * (fusion - passive), 1e-12, 1 - 1e-12)
            loss = -np.sum(weights * (labels * np.log(mixed) + (1 - labels) * np.log1p(-mixed)))
            penalty = self.ridge * np.dot(theta[1:], theta[1:]) / 2
            derivative = weights * (mixed - labels) / (mixed * (1 - mixed)) * (fusion - passive) * gate * (1 - gate)
            gradient = features.T @ derivative
            gradient[1:] += self.ridge * theta[1:]
            return float(loss + penalty), gradient

        optimized = minimize(objective, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
        if not optimized.success:
            raise RuntimeError(f"Gate optimization failed: {optimized.message}")
        self.theta = optimized.x
        self.optimization = {"success": bool(optimized.success), "iterations": int(optimized.nit), "objective": float(optimized.fun)}
        constant = minimize_scalar(lambda q: balanced_loss(labels, passive + q * (fusion - passive)), bounds=(0, 1), method="bounded")
        self.constant_gate = float(constant.x)
        self.logit_center = float(np.mean(logit))
        self.logit_std = max(float(np.std(logit)), 1e-8)
        self.logit_fusion = fit_logistic(np.column_stack([(logit - self.logit_center) / self.logit_std, active]), labels)
        return self

    def predict(self, logit: np.ndarray, probability: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
        passive = self.passive.predict_proba(logit[:, None])[:, 1]
        static = self.fusion.predict_proba(np.column_stack([probability, active]))[:, 1]
        gate = expit(self.gate_features(logit, active) @ self.theta)
        logit_fusion = self.logit_fusion.predict_proba(np.column_stack([(logit - self.logit_center) / self.logit_std, active]))[:, 1]
        return {"calibrated_passive": passive, "refit_static_fusion": static,
                "constant_mixture": passive + self.constant_gate * (static - passive),
                "logit_payload_fusion": logit_fusion,
                "reliability_mixture": passive + gate * (static - passive), "gate": gate}

    def ranking_scores(self, logit: np.ndarray, probability: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate mixture odds in log space, preserving extreme-score ranks."""
        passive = self.passive.decision_function(logit[:, None])
        fused = self.fusion.decision_function(np.column_stack([probability, active]))
        gate = self.gate_features(logit, active) @ self.theta

        def mixture_odds(gate_logit):
            log_gate = -np.logaddexp(0, -gate_logit)
            log_fallback = -np.logaddexp(0, gate_logit)
            log_positive = np.logaddexp(log_gate - np.logaddexp(0, -fused), log_fallback - np.logaddexp(0, -passive))
            log_negative = np.logaddexp(log_gate - np.logaddexp(0, fused), log_fallback - np.logaddexp(0, passive))
            return log_positive - log_negative

        constant_logit = np.log(self.constant_gate) - np.log1p(-self.constant_gate)
        return {"calibrated_passive": passive, "refit_static_fusion": fused,
                "constant_mixture": mixture_odds(constant_logit),
                "logit_payload_fusion": self.logit_fusion.decision_function(np.column_stack([(logit - self.logit_center) / self.logit_std, active])),
                "reliability_mixture": mixture_odds(gate)}

    def save(self, path: Path):
        np.savez(path, ridge=self.ridge, kind=self.kind, gate_coefficients=self.theta,
                 logit_scale=self.logit_scale, passive_coefficients=self.passive.coef_, passive_intercept=self.passive.intercept_,
                 fusion_coefficients=self.fusion.coef_, fusion_intercept=self.fusion.intercept_,
                 logit_fusion_coefficients=self.logit_fusion.coef_, logit_fusion_intercept=self.logit_fusion.intercept_,
                 logit_center=self.logit_center, logit_std=self.logit_std, constant_gate=self.constant_gate)


def ranking(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def prepare_auc(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """Sort once; source bootstrap changes integer weights, not score order."""
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    return order, starts, labels[order] == 1


def weighted_auc(prepared: tuple, weights: np.ndarray) -> float:
    order, starts, positive = prepared
    weight = weights[order]
    positives = np.add.reduceat(weight * positive, starts)
    negatives = np.add.reduceat(weight * (~positive), starts)
    preceding_negatives = np.cumsum(negatives) - negatives
    return float(np.sum(positives * (preceding_negatives + negatives / 2)) / (positives.sum() * negatives.sum()))


def grouped_summary(frame: pd.DataFrame, scores: dict, ranks: dict, thresholds: dict, samples: int) -> dict:
    labels = frame["label"].to_numpy(dtype=int)
    report = {name: classification_metrics(labels, probabilities, thresholds[name], ranking_scores=ranks[name]) for name, probabilities in scores.items()}
    for name in report:
        report[name]["balanced_log_loss"] = balanced_loss(labels, scores[name])
    source_codes, sources = pd.factorize(frame["source_id"], sort=True)
    collections = [np.unique(source_codes[frame["generator"].to_numpy() == name]) for name in sorted(frame["generator"].unique())]
    prepared = {name: prepare_auc(labels, ranks[name]) for name in scores}
    rng = np.random.default_rng(SEED)
    aucs = {name: [] for name in scores}
    for _ in range(samples):
        source_weights = np.zeros(len(sources), dtype=int)
        for group in collections:
            picked = rng.integers(0, len(group), len(group))
            source_weights[group] = np.bincount(picked, minlength=len(group))
        weights = source_weights[source_codes]
        for name in scores:
            aucs[name].append(weighted_auc(prepared[name], weights))
    for name in scores:
        report[name]["roc_auc_ci95"] = np.percentile(aucs[name], [2.5, 97.5]).tolist()
        comparisons = {}
        for reference in ("probe", "original_fusion", "logit_payload_fusion", "constant_mixture"):
            difference = np.asarray(aucs[name]) - np.asarray(aucs[reference])
            comparisons[reference] = {"estimate": report[name]["roc_auc"] - report[reference]["roc_auc"], "ci95": np.percentile(difference, [2.5, 97.5]).tolist()}
        report[name]["paired_auc_vs_reference"] = comparisons
        report[name]["paired_auc_minus_probe"] = comparisons["probe"]
    return {"rows": int(len(frame)), "sources": int(frame["source_id"].nunique()), "systems": report}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    corrected = ROOT / "outputs" / "conference_clean"
    destination = ROOT / "outputs" / "extension" / "reliability"
    destination.mkdir(parents=True, exist_ok=True)
    validation_path = corrected / "validation_predictions.csv"
    validation = pd.read_csv(validation_path, keep_default_na=False)
    keys = list(zip(validation["attack"].astype(str), validation["severity"].astype(int)))
    if any(key in HELD_OUT for key in keys) or (validation["generator"] == "midjourney").any():
        raise RuntimeError("Held-out attack/generator entered validation fitting.")
    if validation.groupby("source_id")["label"].nunique().max() != 1:
        raise RuntimeError("Origin labels conflict within validation sources.")
    protocol = {"design": "Smooth validation-trained mixture of calibrated passive and static-fusion experts",
        "retrospective": True, "limitations": "Designed after previous test results were examined; grouped CV prevents extension selection on held-out rows but does not create a fresh untouched test or establish novelty.",
        "seed": SEED, "folds": 5, "selection_metric": "mean held-source class-balanced log loss", "ridge_candidates": RIDGES,
        "gate_features": ["payload agreement centered on 0.5", "absolute passive logit normalized by training-fold 90th percentile", "their interaction"],
        "forbidden_fit_features": ["attack identity", "generator identity", "test labels"],
        "held_out_attacks": sorted(HELD_OUT), "held_out_generator": "midjourney", "bootstrap_samples": args.bootstrap_samples,
        "validation_input_sha256": digest(validation_path), "script_sha256": digest(Path(__file__)),
        "falsifiable_criteria": ["Greater full-test ROC-AUC than passive alone", "No more than 0.005 lower AUC than passive on the prior reduced-recovery diagnostic slice", "Compare against logit+payload logistic and constant-mixture baselines; report failures"]}
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    validation = append_exact_validation_logits(validation, corrected)
    folds = grouped_folds(validation)
    validation["extension_fold"] = folds
    validation[["image_id", "source_id", "generator", "attack", "severity", "extension_fold"]].to_csv(destination / "validation_folds.csv", index=False)
    choices, models, thresholds = {}, {}, {}
    for backbone, probability in (("resnet", "probe"), ("clip", "clip_probe")):
        logit = validation[probability + "_logit"].to_numpy(dtype=float)
        prob = validation[probability].to_numpy(dtype=float)
        active = validation["active_score"].to_numpy(dtype=float)
        labels = validation["label"].to_numpy(dtype=int)
        selection = {}
        for kind in GATE_KINDS:
            candidates = []
            for ridge in RIDGES:
                oof = np.empty(len(validation))
                oof_ranking = np.empty(len(validation))
                for fold in range(5):
                    fit, held = folds != fold, folds == fold
                    model = ReliabilityMixture(ridge, kind).fit(logit[fit], prob[fit], active[fit], labels[fit])
                    oof[held] = model.predict(logit[held], prob[held], active[held])["reliability_mixture"]
                    oof_ranking[held] = model.ranking_scores(logit[held], prob[held], active[held])["reliability_mixture"]
                candidates.append({"ridge": ridge, "held_source_balanced_log_loss": balanced_loss(labels, oof), "held_source_auc": float(roc_auc_score(labels, oof_ranking))})
            best = min(candidates, key=lambda candidate: (candidate["held_source_balanced_log_loss"], -candidate["ridge"]))
            selection[kind] = {"selected": best, "candidates": candidates}
            model = ReliabilityMixture(best["ridge"], kind).fit(logit, prob, active, labels)
            model.save(destination / f"{backbone}_{kind}.npz")
            models[backbone, kind] = model
            print(backbone, kind, "selected", best, flush=True)
        choices[backbone] = selection
        predicted = models[backbone, "full"].predict(logit, prob, active)
        predicted.pop("gate")
        predicted["payload_only_gate"] = models[backbone, "payload_only"].predict(logit, prob, active)["reliability_mixture"]
        predicted["uncertainty_only_gate"] = models[backbone, "uncertainty_only"].predict(logit, prob, active)["reliability_mixture"]
        predicted["probe"] = prob
        predicted["original_fusion"] = validation["fusion" if backbone == "resnet" else "clip_fusion"].to_numpy(dtype=float)
        thresholds[backbone] = {name: select_threshold(labels, score) for name, score in predicted.items()}
    (destination / "selection.json").write_text(json.dumps({"choices": choices, "thresholds": thresholds, "test_was_not_loaded_during_selection": True}, indent=2), encoding="utf-8")
    # Test access is deliberately after all model/hyperparameter/threshold files.
    test_path = corrected / "test_predictions.csv"
    test = pd.read_csv(test_path, keep_default_na=False)
    if set(test["source_id"]) & set(validation["source_id"]):
        raise RuntimeError("Source overlap between validation and test.")
    summary = {"protocol": protocol, "selection": choices, "thresholds": thresholds, "test_input_sha256": digest(test_path), "results": {}}
    all_predictions = test[["image_id", "source_id", "label", "generator", "attack", "severity", "group"]].copy()
    for backbone, probability in (("resnet", "probe"), ("clip", "clip_probe")):
        logit = test[probability + "_logit"].to_numpy(dtype=float)
        prob = test[probability].to_numpy(dtype=float)
        active = test["active_score"].to_numpy(dtype=float)
        scores = models[backbone, "full"].predict(logit, prob, active)
        gate = scores.pop("gate")
        scores["payload_only_gate"] = models[backbone, "payload_only"].predict(logit, prob, active)["reliability_mixture"]
        scores["uncertainty_only_gate"] = models[backbone, "uncertainty_only"].predict(logit, prob, active)["reliability_mixture"]
        scores["probe"] = prob
        original = "fusion" if backbone == "resnet" else "clip_fusion"
        scores["original_fusion"] = test[original].to_numpy(dtype=float)
        ranks = models[backbone, "full"].ranking_scores(logit, prob, active)
        ranks["payload_only_gate"] = models[backbone, "payload_only"].ranking_scores(logit, prob, active)["reliability_mixture"]
        ranks["uncertainty_only_gate"] = models[backbone, "uncertainty_only"].ranking_scores(logit, prob, active)["reliability_mixture"]
        ranks["probe"], ranks["original_fusion"] = logit, test[original + "_logit"].to_numpy(dtype=float)
        masks = {"full_test": np.ones(len(test), dtype=bool),
                 "reduced_recovery": ((test["attack"] == "crop") | ((test["attack"] == "lowres") & (test["severity"] == 2)) | test["attack"].isin(["regenerate", "spectral"])).to_numpy(),
                 "held_out_attacks": np.array([key in HELD_OUT for key in zip(test["attack"].astype(str), test["severity"].astype(int))]),
                 "held_out_generator": test["generator"].isin(["real", "midjourney"]).to_numpy(),
                 "clean_unmarked": test["group"].isin(["R-C", "S-U"]).to_numpy()}
        results = {}
        for name, mask in masks.items():
            frame = test.loc[mask].copy()
            results[name] = grouped_summary(frame, {key: value[mask] for key, value in scores.items()}, {key: value[mask] for key, value in ranks.items()}, thresholds[backbone], args.bootstrap_samples)
            print(backbone, name, {key: round(value["roc_auc"], 4) for key, value in results[name]["systems"].items()}, flush=True)
        results["gate_by_condition"] = {f"{attack}:{severity}": {"rows": len(sub), "mean_fusion_weight": float(gate[sub.index].mean())} for (attack, severity), sub in test.groupby(["attack", "severity"])}
        selected = results["full_test"]["systems"]["reliability_mixture"]["roc_auc"]
        weak = results["reduced_recovery"]["systems"]
        results["criteria"] = {"full_auc_above_probe": selected > results["full_test"]["systems"]["probe"]["roc_auc"],
            "reduced_recovery_auc_within_0p005_of_probe": weak["reliability_mixture"]["roc_auc"] >= weak["probe"]["roc_auc"] - 0.005,
            "full_auc_above_logit_payload_fusion": selected > results["full_test"]["systems"]["logit_payload_fusion"]["roc_auc"],
            "full_auc_above_constant_mixture": selected > results["full_test"]["systems"]["constant_mixture"]["roc_auc"]}
        summary["results"][backbone] = results
        for name, score in scores.items():
            all_predictions[f"{backbone}_{name}"] = score
        all_predictions[f"{backbone}_fusion_weight"] = gate
    all_predictions.to_csv(destination / "test_predictions.csv", index=False)
    (destination / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Archived extension at", destination, flush=True)


if __name__ == "__main__":
    main()
