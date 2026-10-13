"""Frozen, paired active/passive fusion with separate development and final stages.

fit reads development features only; calibrate reads independent calibration;
evaluate reads a locked final test without fitting or threshold selection.
Metadata controls pairing, weighting and reporting, never prediction features.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/phase6/fusion_v2"
METHODS = ("dwt_dct", "trustmark")
BACKBONES = ("resnet", "clip")
FEATURES = ("agreement", "min_view_agreement", "signed_margin", "q10_margin",
            "mean_abs_margin", "stability", "margin_stability", "full_match",
            "full_match_views", "ecc_valid", "codeword_agreement")
STABILITY = {"min_view_agreement", "stability", "margin_stability", "full_match_views"}
FULL_MATCH = {"full_match", "full_match_views", "ecc_valid", "codeword_agreement"}
VARIANTS = ("primary", "no_null", "no_pair", "no_stability", "no_full_match")
SCHEMA = "phase6_paired_fusion_v1"
SEED = 20261011
RISK = .05
NONINFERIORITY = .01
BOOTSTRAPS = 2000
RIDGE = .001
PAIR_STRENGTH = .25
CORRECTION_BOUNDS = (1., 2., 4., 8., 16.)
COMMON_CONDITIONS = ("identity", "jpeg50", "crop", "lowres32")
PRIMARY_SYSTEM = "trustmark/resnet"
ORIGIN_BASELINES = ("probe", "static_fusion", "hard_13_gate", "simple_confidence", "old_reliability_mixture", "old_capped_positive", "unconstrained_full_logistic")
REQUIRED = ("image_id", "source_id", "role", "collection", "label", "watermark_method",
            "embedded_payload", "condition", "detector", "query_payload", "resnet_logit", "clip_logit")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_json(value):
    """Represent unavailable scalar statistics explicitly, never JSON NaN."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_features(path, roles=None, queries=None):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED + FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if frame.empty or frame[list(REQUIRED[:4])].eq("").any().any():
        raise ValueError("Empty feature input or identity metadata")
    if roles is not None and not set(frame.role).issubset(set(roles)):
        raise ValueError(f"Unexpected roles: {sorted(set(frame.role)-set(roles))}")
    for name in ("label", "resnet_logit", "clip_logit") + FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError(f"Nonfinite features: {name}")
    if not frame.label.isin((0, 1)).all():
        raise ValueError("Labels must be binary")
    if not frame.detector.isin(METHODS).all():
        raise ValueError("Unknown decoder")
    unmarked = frame.watermark_method.isin(("", "unmarked", "none"))
    own_mark = frame.watermark_method.eq(frame.detector)
    if not (unmarked | own_mark).all():
        raise ValueError("Only own-mark or unmarked received images are eligible")
    query_set = sorted(frame.query_payload.unique())
    if len(query_set) != 3 or any(len(q) != 16 or set(q) - {"0", "1"} for q in query_set):
        raise ValueError("Exactly three registered 16-bit queries are required")
    if queries is not None and query_set != list(queries):
        raise ValueError("Registered query set changed")
    if not frame.loc[~unmarked, "embedded_payload"].isin(query_set).all():
        raise ValueError("Marked covers have unregistered payloads")
    if frame.groupby("source_id").role.nunique().max() != 1:
        raise ValueError("One source appears in multiple roles")
    if frame.groupby("source_id").label.nunique().max() != 1:
        raise ValueError("Source origin labels disagree")
    if frame.groupby("source_id").collection.nunique().max() != 1:
        raise ValueError("Source collection labels disagree")
    for name in ("agreement", "min_view_agreement", "stability", "full_match", "full_match_views", "ecc_valid", "codeword_agreement"):
        if not frame[name].between(0, 1).all():
            raise ValueError(f"Expected [0,1] feature: {name}")
    invariant = ("source_id", "role", "collection", "label", "watermark_method", "embedded_payload", "condition", "resnet_logit", "clip_logit")
    groups = frame.groupby(["detector", "image_id"], sort=False)
    if groups[list(invariant)].nunique().to_numpy().max() != 1:
        raise ValueError("Received-image metadata/logits vary across queries")
    if not groups.size().eq(3).all() or frame.duplicated(["detector", "image_id", "query_payload"]).any():
        raise ValueError("Each received image/decoder needs exactly three unique queries")
    frame = frame.sort_values(["detector", "image_id", "query_payload"], kind="mergesort").reset_index(drop=True)
    frame["is_unmarked"] = frame.watermark_method.isin(("", "unmarked", "none"))
    return frame, query_set


def view(frame, detector, backbone, role=None):
    subset = frame.loc[frame.detector.eq(detector)]
    if role is not None:
        subset = subset.loc[subset.role.eq(role)]
    subset = subset.sort_values(["image_id", "query_payload"], kind="mergesort")
    if subset.empty:
        raise ValueError(f"Missing data for {detector}/{backbone}/{role}")
    meta = subset.iloc[::3].reset_index(drop=True).copy()
    values = subset[list(FEATURES)].to_numpy(float).reshape(len(meta), 3, len(FEATURES))
    return {"meta": meta, "features": values, "z": meta[f"{backbone}_logit"].to_numpy(float),
            "y": meta.label.to_numpy(int), "unmarked": meta.is_unmarked.to_numpy(bool),
            "queries": sorted(subset.query_payload.unique())}


def weights(data):
    """Equal class, source and marked/unmarked-state weights, then equal images."""
    meta = data["meta"].copy()
    sources = meta[["source_id", "label"]].drop_duplicates()
    counts = sources.groupby("label").size().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError("Fitting requires both origins")
    state_counts = meta.groupby("source_id").is_unmarked.nunique()
    within_state = meta.groupby(["source_id", "is_unmarked"]).size()
    result = np.asarray([1 / (2 * counts[int(y)] * state_counts[s] * within_state[s, u])
                         for s, y, u in zip(meta.source_id, meta.label, meta.is_unmarked)], float)
    if not np.isclose(result.sum(), 1):
        raise ValueError("Source-balanced weights do not sum to one")
    return result


def pairs(data):
    meta = data["meta"]
    unmarked = meta.loc[meta.is_unmarked & meta.label.eq(1)]
    if unmarked.duplicated(["source_id", "condition"]).any():
        raise ValueError("Ambiguous source/condition unmarked pairing; include seed in condition")
    lookup = {(row.source_id, row.condition): i for i, row in unmarked.iterrows()}
    first, second, identities = [], [], []
    for i, row in meta.loc[~meta.is_unmarked & meta.label.eq(1)].iterrows():
        key = row.source_id, row.condition
        if key not in lookup:
            raise ValueError(f"Marked fitting image lacks an unmarked pair: {key}")
        first.append(i); second.append(lookup[key]); identities.append(row.source_id)
    if not first:
        raise ValueError("No marked/unmarked AI fitting pairs")
    counts = pd.Series(identities).value_counts()
    w = np.asarray([1 / (len(counts) * counts[s]) for s in identities], float)
    return np.asarray(first), np.asarray(second), w


def standardizer(data, names):
    raw = data["features"][:, :, [FEATURES.index(n) for n in names]]
    mean = raw.reshape(-1, len(names)).mean(axis=0)
    scale = raw.reshape(-1, len(names)).std(axis=0)
    scale = np.maximum(scale, 1e-8)
    return mean, scale


def design(data, model):
    indices = [FEATURES.index(n) for n in model["features"]]
    values = (data["features"][:, :, indices] - np.asarray(model["mean"])) / np.asarray(model["scale"])
    if model.get("include_logit"):
        passive = (data["z"] - model["z_mean"]) / model["z_scale"]
        values = np.concatenate([np.broadcast_to(passive[:, None, None], (*values.shape[:2], 1)), values], axis=2)
    return np.concatenate([np.ones((*values.shape[:2], 1)), np.clip(values, -20, 20)], axis=2)


def query_max(raw):
    j = np.argmax(raw, axis=1)
    return raw[np.arange(len(raw)), j], j


def correction_objective(data, model):
    """Return the exact fit loss and analytic max-query subgradient."""
    x = design(data, model)
    w = weights(data)
    pm, pu, pw = pairs(data)
    uw = w * data["unmarked"]
    uw /= uw.sum()
    y, z = data["y"], data["z"]
    bound, null_strength = model["bound"], model["null_strength"]

    def objective(theta):
        q = expit(np.einsum("nqf,f->nq", x, theta))
        correction, index = query_max(bound * q)
        selected_x = x[np.arange(len(x)), index]
        derivative = correction * (1 - correction / bound)
        s = z + correction
        loss = np.sum(w * (np.logaddexp(0, s) - y * s))
        dc = w * (expit(s) - y)
        if null_strength:
            loss += null_strength * np.sum(uw * correction ** 2)
            dc += 2 * null_strength * uw * correction
        if model["pair_strength"]:
            difference = correction[pm] - correction[pu]
            loss += model["pair_strength"] * np.sum(pw * np.logaddexp(0, -difference))
            d = -model["pair_strength"] * pw * expit(-difference)
            np.add.at(dc, pm, d)
            np.add.at(dc, pu, -d)
        loss += model["ridge"] * np.dot(theta, theta) / 2
        gradient = selected_x.T @ (dc * derivative) + model["ridge"] * theta
        return float(loss), gradient
    return objective


def fit_correction(data, bound, null_strength, variant):
    names = [n for n in FEATURES if not (variant == "no_stability" and n in STABILITY)
             and not (variant == "no_full_match" and n in FULL_MATCH)]
    mean, scale = standardizer(data, names)
    model = {"kind": "correction", "features": names, "mean": mean.tolist(), "scale": scale.tolist(),
             "include_logit": True, "z_mean": float(data["z"].mean()), "z_scale": max(float(data["z"].std()), 1e-8),
             "bound": float(bound), "null_strength": float(null_strength), "pair_strength": 0.0 if variant == "no_pair" else PAIR_STRENGTH,
             "ridge": RIDGE, "variant": variant}
    objective = correction_objective(data, model)
    x = design(data, model)

    initial = np.zeros(x.shape[2]); initial[0] = -2
    # A fixed, query-symmetric reliability prior avoids an all-query tie at
    # initialization. It is identical for every candidate and never tuned.
    initial[2+names.index("agreement")] = .1
    initial[2+names.index("signed_margin")] = .1
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=[(-20, 20)] * len(initial),
                      options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7, "maxls": 50})
    model["parameters"] = result.x.tolist()
    model["optimizer"] = {"converged": bool(result.success), "iterations": int(result.nit),
                          "objective": float(result.fun), "message": str(result.message)}
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        model["eligible"] = False
    else:
        model["eligible"] = True
    return model


def fit_logistic(data, full=False):
    if full:
        names = list(FEATURES)
        mean, scale = standardizer(data, names)
        model = {"kind": "full_logistic", "features": names, "mean": mean.tolist(), "scale": scale.tolist(),
                 "include_logit": True, "z_mean": float(data["z"].mean()), "z_scale": max(float(data["z"].std()), 1e-8)}
        x = design(data, model)
    else:
        a = data["features"][:, :, FEATURES.index("agreement")]
        p = np.broadcast_to(expit(data["z"])[:, None], a.shape)
        x = np.stack([np.ones_like(a), p, a], axis=2)
        model = {"kind": "static"}
    w, y = weights(data), data["y"]

    def objective(theta):
        raw = np.einsum("nqf,f->nq", x, theta)
        score, j = query_max(raw)
        loss = np.sum(w * (np.logaddexp(0, score) - y * score)) + RIDGE * np.dot(theta[1:], theta[1:]) / 2
        gradient = x[np.arange(len(x)), j].T @ (w * (expit(score) - y))
        gradient[1:] += RIDGE * theta[1:]
        return float(loss), gradient
    result = minimize(objective, np.zeros(x.shape[2]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10, "maxls": 50})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Baseline logistic did not converge: {result.message}")
    model["parameters"] = result.x.tolist()
    model["optimizer"] = {"converged": True, "iterations": int(result.nit)}
    return model


def static_queries(data, model):
    a = data["features"][:, :, FEATURES.index("agreement")]
    return model["parameters"][0] + model["parameters"][1] * expit(data["z"])[:, None] + model["parameters"][2] * a


def fit_reliability(data, static):
    a = data["features"][:, :, FEATURES.index("agreement")]
    uncertainty = np.clip(np.abs(data["z"]) / max(float(np.percentile(np.abs(data["z"]), 90)), 1.0), 0, 3)
    u = np.broadcast_to(uncertainty[:, None], a.shape)
    x = np.stack([np.ones_like(a), 2*(a-.5), u, 2*(a-.5)*u], axis=2)
    passive = expit(data["z"])[:, None]
    fused = expit(static_queries(data, static))
    w, y = weights(data), data["y"]
    def objective(theta):
        gate = expit(np.einsum("nqf,f->nq", x, theta))
        mixture = passive + gate * (fused-passive)
        probability, j = query_max(mixture)
        probability = np.clip(probability, 1e-12, 1-1e-12)
        loss = -np.sum(w * (y*np.log(probability)+(1-y)*np.log1p(-probability))) + RIDGE*np.dot(theta[1:], theta[1:])/2
        derivative = w*(probability-y)/(probability*(1-probability)) * (fused-passive)[np.arange(len(x)), j] * gate[np.arange(len(x)), j] * (1-gate[np.arange(len(x)), j])
        gradient = x[np.arange(len(x)), j].T @ derivative
        gradient[1:] += RIDGE*theta[1:]
        return float(loss), gradient
    result = minimize(objective, np.zeros(4), jac=True, method="L-BFGS-B", bounds=[(-15, 15)]*4,
                      options={"maxiter": 500, "ftol": 1e-10, "maxls": 50})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Reliability baseline did not converge: {result.message}")
    return {"kind": "reliability", "parameters": result.x.tolist(), "static": static,
            "logit_scale": max(float(np.percentile(np.abs(data["z"]), 90)), 1.0),
            "optimizer": {"converged": True, "iterations": int(result.nit)},
            "adaptation": "Local smooth reliability-mixture formula refitted on identical Phase6 fitting sources; passive expert remains frozen."}


def empirical_threshold(meta, scores, risk=RISK):
    real = meta.label.eq(0)
    if not real.any():
        raise ValueError("Photographic controls are required for thresholds")
    by_source = pd.DataFrame({"source_id": meta.loc[real, "source_id"].to_numpy(), "score": scores[real.to_numpy()]}).groupby("source_id").score.max().to_numpy()
    allowed = int(np.floor(risk * len(by_source)))
    ordered = np.sort(by_source)
    return float(np.nextafter(ordered[len(ordered)-allowed-1], np.inf))


def bit_ramp(data):
    a = data["features"][:, :, FEATURES.index("agreement")]
    return np.where(a >= 13/16, np.clip((a-12/16)/(1-12/16), 0, 1), 0)


def fit_capped(data):
    ramp = bit_ramp(data).max(axis=1)
    threshold = empirical_threshold(data["meta"], data["z"])
    real = data["y"] == 0
    records = pd.DataFrame({"source_id": data["meta"].source_id, "z": data["z"], "ramp": ramp, "real": real})
    allowed = int(np.floor(RISK * data["meta"].loc[real, "source_id"].nunique()))
    crossings = []
    already = 0
    for _, group in records.loc[real].groupby("source_id"):
        if (group.z >= threshold).any():
            already += 1
        else:
            supported = group.ramp > 0
            if supported.any():
                crossings.append(float(((threshold-group.loc[supported, "z"])/group.loc[supported, "ramp"]).min()))
    upper = 40.0
    remaining = allowed-already
    if remaining < 0:
        raise RuntimeError("Capped baseline passive threshold is infeasible")
    if len(crossings) > remaining:
        upper = max(min(upper, float(np.sort(crossings)[remaining])*(1-1e-10)), 0)
    w = weights(data)
    def objective(b):
        s = data["z"]+b*ramp
        return float(np.sum(w*(np.logaddexp(0, s)-data["y"]*s)) + RIDGE*b*b/2)
    if upper < 1e-12:
        coefficient = 0.0
    else:
        result = minimize_scalar(objective, bounds=(0, upper), method="bounded", options={"xatol": 1e-9})
        if not result.success:
            raise RuntimeError("Capped baseline failed")
        coefficient = min((0., upper, float(result.x)), key=objective)
    return {"kind": "capped", "coefficient": coefficient, "cap": upper, "fit_threshold": threshold,
            "adaptation": "Local capped positive-evidence formula refitted with frozen passive score and source-level max-query calibration."}


def score(data, model):
    kind = model["kind"]
    if kind == "probe":
        result = data["z"].copy()
    elif kind == "correction":
        raw = model["bound"]*expit(np.einsum("nqf,f->nq", design(data, model), np.asarray(model["parameters"])))
        result = data["z"]+raw.max(axis=1)
    elif kind == "static":
        result = static_queries(data, model).max(axis=1)
    elif kind == "full_logistic":
        result = np.einsum("nqf,f->nq", design(data, model), np.asarray(model["parameters"])).max(axis=1)
    elif kind == "hard_gate":
        agreement = data["features"][:, :, FEATURES.index("agreement")]
        result = np.where(agreement >= 13/16, static_queries(data, model["static"]), data["z"][:, None]).max(axis=1)
    elif kind == "simple_confidence":
        agreement = data["features"][:, :, FEATURES.index("agreement")]
        stability = data["features"][:, :, FEATURES.index("stability")]
        result = data["z"] + (2*np.maximum(2*agreement-1, 0)*stability).max(axis=1)
    elif kind == "watermark_only":
        result = data["features"][:, :, FEATURES.index(model["feature"])].max(axis=1)
    elif kind == "capped":
        result = data["z"] + model["coefficient"]*bit_ramp(data).max(axis=1)
    elif kind == "reliability":
        agreement = data["features"][:, :, FEATURES.index("agreement")]
        u = np.broadcast_to(np.clip(np.abs(data["z"])/model["logit_scale"], 0, 3)[:, None], agreement.shape)
        x = np.stack([np.ones_like(agreement), 2*(agreement-.5), u, 2*(agreement-.5)*u], axis=2)
        gate = np.einsum("nqf,f->nq", x, np.asarray(model["parameters"]))
        f = static_queries(data, model["static"])
        z = data["z"][:, None]
        log_positive = np.logaddexp(-np.logaddexp(0, -gate)-np.logaddexp(0, -f), -np.logaddexp(0, gate)-np.logaddexp(0, -z))
        log_negative = np.logaddexp(-np.logaddexp(0, -gate)-np.logaddexp(0, f), -np.logaddexp(0, gate)-np.logaddexp(0, z))
        result = (log_positive-log_negative).max(axis=1)
    else:
        raise ValueError(f"Unknown model: {kind}")
    if not np.isfinite(result).all():
        raise RuntimeError("A model produced nonfinite scores")
    return result


def rates(data, scores, threshold, base_decision=None, strict=False):
    decision = scores > threshold if strict else scores >= threshold
    meta = data["meta"]
    real = data["y"] == 0
    ai = data["y"] == 1
    marked = ai & ~data["unmarked"]
    unmarked = ai & data["unmarked"]
    def avg(mask):
        if not mask.any():
            return None
        return float(pd.DataFrame({"source_id": meta.loc[mask, "source_id"].to_numpy(), "d": decision[mask]}).groupby("source_id").d.mean().mean())
    result = {"marked_tpr": avg(marked), "unmarked_tpr": avg(unmarked), "real_image_fpr": avg(real),
              "real_source_fpr": float(pd.DataFrame({"s": meta.loc[real, "source_id"].to_numpy(), "d": decision[real]}).groupby("s").d.max().mean()) if real.any() else None}
    group_results = []
    for (collection, condition), indices in meta.loc[unmarked].groupby(["collection", "condition"]).groups.items():
        ii = np.asarray(list(indices), int)
        row = {"collection": collection, "condition": condition, "sources": int(meta.loc[ii, "source_id"].nunique()), "tpr": float(decision[ii].mean())}
        if base_decision is not None:
            row["tpr_minus_probe"] = float((decision[ii].astype(float)-base_decision[ii]).mean())
            row["harmful_switch_rate"] = float((base_decision[ii] & ~decision[ii]).mean())
        group_results.append(row)
    result["unmarked_groups"] = group_results
    return result


def eligible_tune(data, model, probe_threshold):
    scores = score(data, model)
    threshold = empirical_threshold(data["meta"], scores)
    baseline = data["z"] >= probe_threshold
    report = rates(data, scores, threshold, baseline)
    losses = [r["tpr_minus_probe"] for r in report["unmarked_groups"]]
    report["empirically_feasible"] = bool(losses and min(losses) >= -NONINFERIORITY-1e-12 and report["real_source_fpr"] <= RISK+1e-12)
    return threshold, report


def verify_complete(out, filename, inputs=None):
    path = out / filename
    if not path.exists():
        return None
    lock = load_json(path)
    if lock["schema"] != SCHEMA or lock["script_sha256"] != digest(__file__):
        raise RuntimeError("Locked fusion artifacts use changed source/schema; preserve old run and choose a new output directory")
    for name, expected in lock["outputs_sha256"].items():
        if not (out/name).exists() or digest(out/name) != expected:
            raise RuntimeError(f"Locked artifact changed: {name}")
    for name, expected in (inputs or {}).items():
        if lock.get(name) != expected:
            raise RuntimeError(f"Locked input changed: {name}")
    return lock


def fit(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    feature_sha = digest(args.features)
    if verify_complete(out, "fit_complete.json", {"development_features_sha256": feature_sha}):
        print("Verified existing frozen learner; fitting not repeated", flush=True); return
    frame, queries = read_features(args.features, ("fit", "tune", "constraint"))
    if set(frame.role) != {"fit", "tune", "constraint"}:
        raise ValueError("All three development roles are required")
    base_paths = {b: ROOT/f"outputs/conference_clean/{b}_full.npz" for b in BACKBONES}
    heads = {b: digest(p) for b, p in base_paths.items()}
    protocol = {"schema": SCHEMA, "created_utc": utc(), "design_status": "Retrospective design; existing sources are development, final test must be new and locked.",
                "registered_queries": queries, "risk": RISK, "unmarked_tpr_tolerance": NONINFERIORITY,
                "correction_bounds": list(CORRECTION_BOUNDS), "null_strengths": [1., 10.], "ridge": RIDGE, "pair_strength": PAIR_STRENGTH,
                "pre_fresh_grid_amendment": "Expanded B to8 and16 before fresh scores because frozen passive logits can be strongly negative; selection remains development-only.",
                "pre_fresh_comparator_amendment": "Strongest comparator is selected among origin/fusion baselines passing the same tune unmarked-harm and source-FPR requirements. Watermark-only scores remain benchmark rows, not eligible origin-fusion comparators. Initial fit retained separately; no fresh scores motivated this correction.",
                "selection": "Marked source-balanced tune TPR at empirical5% source FPR; require <=1pp unmarked TPR loss in each seen collection/condition; B=0 if none feasible.",
                "constraint_role": "Assess selected model at unchanged tune cuts on all four views; never select a different model from this result.",
                "prediction_inputs": list(FEATURES)+["frozen passive logit"], "forbidden_prediction_inputs": ["source identity", "role", "collection", "origin label", "embedded payload", "condition", "clean reference", "sibling edits"],
                "aggregation": "Maximum over the same three registered query scores per received image; calibration source maxima cover declared conditions.",
                "primary_common_conditions": list(COMMON_CONDITIONS), "bootstrap_samples": BOOTSTRAPS, "bootstrap_seed": SEED,
                "confirmatory_primary": {"system": PRIMARY_SYSTEM, "fresh_AI_sources": 600, "fresh_AI_collections": 3,
                                         "views": list(COMMON_CONDITIONS), "marked_gain": "Two-sided paired source-bootstrap CI lower > 0 versus probe",
                                         "unmarked_noninferiority": "One-sided paired loss upper < .01; zero any-view harm has a conditional pooled finite-sample bound",
                                         "strongest_baseline_gain": "Two-sided paired CI lower > 0 versus the feasible origin-fusion baseline with greatest marked tune TPR, frozen before final scores",
                                         "secondary": "CLIP, DWT, per-family/condition, marked shares, attacks, abstention and verification"},
                "final_calibration": "Independent negative sources, frozen scoring functions, strict source-max split-conformal cut per photographic collection; no unseen-shift guarantee.",
                "development_features_sha256": feature_sha, "base_heads_sha256": heads, "script_sha256": digest(__file__)}
    write_json(out/"protocol.json", protocol)
    models, selections, constraints = {}, {}, {}
    for detector in METHODS:
        for backbone in BACKBONES:
            key = f"{detector}/{backbone}"
            training, tuning, constraint = [view(frame, detector, backbone, role) for role in ("fit", "tune", "constraint")]
            probe_threshold = empirical_threshold(tuning["meta"], tuning["z"])
            suite = {"probe": {"kind": "probe"}}
            selected = {}; tune_cuts = {"probe": probe_threshold}
            for variant in VARIANTS:
                candidates = []
                for bound in CORRECTION_BOUNDS:
                    for null_strength in ((0.,) if variant == "no_null" else (1., 10.)):
                        model = fit_correction(training, bound, null_strength, variant)
                        if not model["eligible"]:
                            report = {"empirically_feasible": False, "optimizer_failure": model["optimizer"]}
                            cut = None
                        else:
                            cut, report = eligible_tune(tuning, model, probe_threshold)
                        candidates.append({"model": model, "tune_threshold": cut, "tune": report})
                feasible = [c for c in candidates if c["tune"]["empirically_feasible"]]
                if feasible:
                    chosen = min(feasible, key=lambda c: (-c["tune"]["marked_tpr"], c["model"]["bound"], -c["model"]["null_strength"]))
                    suite[variant] = chosen["model"]
                    tune_cuts[variant] = chosen["tune_threshold"]
                    fallback = False
                else:
                    suite[variant] = {"kind": "probe", "fallback": "No eligible correction met prespecified tune criteria"}
                    tune_cuts[variant] = probe_threshold
                    fallback = True
                selected[variant] = {"zero_fallback": fallback, "candidates": candidates, "selected_model": suite[variant], "tune_threshold": tune_cuts[variant]}
                print(key, variant, "zero fallback" if fallback else f"B={suite[variant]['bound']}, null={suite[variant]['null_strength']}", flush=True)
            static = fit_logistic(training)
            suite.update({"static_fusion": static, "hard_13_gate": {"kind": "hard_gate", "static": static},
                          "simple_confidence": {"kind": "simple_confidence"}, "old_reliability_mixture": fit_reliability(training, static),
                          "old_capped_positive": fit_capped(training), "unconstrained_full_logistic": fit_logistic(training, full=True),
                          "watermark_only_agreement": {"kind": "watermark_only", "feature": "agreement"},
                          "watermark_only_margin": {"kind": "watermark_only", "feature": "signed_margin"},
                          "watermark_only_full_match": {"kind": "watermark_only", "feature": "full_match"}})
            baseline_tune = {}
            for name, model in suite.items():
                if name not in tune_cuts:
                    tune_cuts[name] = empirical_threshold(tuning["meta"], score(tuning, model))
                if name not in VARIANTS:
                    baseline_tune[name] = rates(tuning, score(tuning, model), tune_cuts[name], tuning["z"] >= probe_threshold)
            baseline_eligibility = {}
            for name in ORIGIN_BASELINES:
                record = baseline_tune[name]
                losses = [row["tpr_minus_probe"] for row in record["unmarked_groups"]]
                baseline_eligibility[name] = bool(losses and min(losses) >= -NONINFERIORITY-1e-12 and record["real_source_fpr"] <= RISK+1e-12)
            eligible_baselines = [name for name in ORIGIN_BASELINES if baseline_eligibility[name]]
            strongest_baseline = min(eligible_baselines, key=lambda name: (-baseline_tune[name]["marked_tpr"], name)) if eligible_baselines else "probe"
            baseline = constraint["z"] >= probe_threshold
            diagnostics = {name: rates(constraint, score(constraint, model), tune_cuts[name], baseline) for name, model in suite.items()}
            primary = diagnostics["primary"]
            primary["strict_development_feasibility"] = bool(primary["real_source_fpr"] <= RISK+1e-12 and all(r["tpr_minus_probe"] >= -NONINFERIORITY-1e-12 for r in primary["unmarked_groups"]))
            primary["claim_status"] = "Development diagnostic only; no final noninferiority or shift guarantee. Constraint failure does not change frozen model."
            models[key] = suite; selections[key] = {"variants": selected, "tune_thresholds": tune_cuts,
                                                  "baseline_tune": baseline_tune, "strongest_baseline": strongest_baseline,
                                                  "origin_baseline_eligibility": baseline_eligibility,
                                                  "watermark_only_is_benchmark_not_origin_comparator": True}; constraints[key] = diagnostics
    source_record = frame[["source_id", "role", "collection", "label"]].drop_duplicates().sort_values("source_id")
    source_record.to_csv(out/"development_sources.csv", index=False)
    write_json(out/"models.json", finite_json(models)); write_json(out/"selection.json", finite_json(selections)); write_json(out/"constraint_diagnostic.json", finite_json(constraints))
    names = ("protocol.json", "models.json", "selection.json", "constraint_diagnostic.json", "development_sources.csv")
    complete = {"schema": SCHEMA, "frozen_utc": utc(), "script_sha256": digest(__file__), "development_features_sha256": feature_sha,
                "base_heads_sha256": heads, "source_count": source_record.source_id.nunique(), "registered_queries": queries,
                "outputs_sha256": {name: digest(out/name) for name in names}, "final_test_read": False}
    write_json(out/"fit_complete.json", finite_json(complete))
    print("FROZEN", complete["frozen_utc"], out, flush=True)


def frozen(out):
    lock = verify_complete(out, "fit_complete.json")
    if lock is None:
        raise RuntimeError("Fit lock is missing")
    for b, expected in lock["base_heads_sha256"].items():
        if digest(ROOT/f"outputs/conference_clean/{b}_full.npz") != expected:
            raise RuntimeError("Frozen original head changed")
    return lock, load_json(out/"models.json"), load_json(out/"protocol.json")


def conformal_cut(meta, scores, risk=RISK):
    real = meta.label.eq(0)
    rows = pd.DataFrame({"source_id": meta.loc[real, "source_id"].to_numpy(), "collection": meta.loc[real, "collection"].to_numpy(), "score": scores[real.to_numpy()]})
    records = []
    for collection, group in rows.groupby("collection", sort=True):
        maxima = np.sort(group.groupby("source_id").score.max().to_numpy())
        k = int(math.ceil((len(maxima)+1)*(1-risk)))
        cut = float(maxima[k-1]) if k <= len(maxima) else None
        records.append({"collection": collection, "sources": len(maxima), "rank": k, "threshold": cut})
    if not records:
        raise ValueError("Independent calibration has no photographic controls")
    threshold = None if any(r["threshold"] is None for r in records) else max(r["threshold"] for r in records)
    return {"threshold": threshold, "strict_exceedance": True, "collection_cuts": records,
            "interpretation": "Marginal source-level control conditional on a fixed scoring function, registered query/view family and within-collection source exchangeability; no realized-split or arbitrary-shift guarantee."}


def calibrate(args):
    out = args.output.resolve()
    fit_lock, models, protocol = frozen(out)
    feature_sha = digest(args.features)
    if verify_complete(out, "calibration_complete.json", {"calibration_features_sha256": feature_sha, "fit_complete_sha256": digest(out/"fit_complete.json")}):
        print("Verified existing fixed final calibration; cuts not recomputed", flush=True); return
    frame, queries = read_features(args.features, ("calibration", "calibrate", "cal"), protocol["registered_queries"])
    dev = pd.read_csv(out/"development_sources.csv", dtype=str, keep_default_na=False)
    if set(frame.source_id) & set(dev.source_id):
        raise ValueError("Independent calibration overlaps development sources")
    if not frame.label.eq(0).all() or not frame.is_unmarked.all():
        raise ValueError("Final origin calibration must contain only unmarked photographs")
    conditions = sorted(frame.condition.unique())
    if not set(COMMON_CONDITIONS).issubset(conditions):
        raise ValueError("Final calibration must include all four primary common conditions")
    coverage = frame.groupby(["detector", "source_id"]).condition.agg(set)
    if any(not set(COMMON_CONDITIONS).issubset(v) for v in coverage):
        raise ValueError("Every calibration source/decoder must include all four primary views")
    policies = {}
    for detector in METHODS:
        for backbone in BACKBONES:
            key = f"{detector}/{backbone}"; data = view(frame, detector, backbone)
            policies[key] = {}
            for name, model in models[key].items():
                values = score(data, model)
                policy = conformal_cut(data["meta"], values)
                threshold = policy["threshold"] if policy["threshold"] is not None else np.inf
                policy["calibration_rates"] = rates(data, values, threshold, strict=True)
                # A descriptive fixed abstention band, not a selective-risk guarantee.
                policy["abstention_width"] = float(np.quantile(np.abs(values-threshold), .2)) if np.isfinite(threshold) else None
                policies[key][name] = policy
    verification = {}
    for detector in METHODS:
        data = view(frame, detector, "resnet")
        margin = data["features"][:, :, FEATURES.index("signed_margin")].max(axis=1)
        verification[detector] = conformal_cut(data["meta"], margin)
        verification[detector]["null_coverage"] = "Photographs only; no guarantee for unmarked AI or wrong-payload marks. Those test errors are reported separately."
    sources = frame[["source_id", "collection", "label"]].drop_duplicates().sort_values("source_id")
    sources.to_csv(out/"calibration_sources.csv", index=False)
    write_json(out/"calibration.json", finite_json({"frozen_utc": utc(), "conditions": conditions, "origin_policies": policies, "verification_policies": verification,
                                                  "abstention": "Width fixed from negative-only calibration to discard the closest20% of negative margins; descriptive coverage, no selective-risk certificate."}))
    write_json(out/"thresholds_locked.json", {"schema": SCHEMA, "frozen_utc": utc(), "calibration_json_sha256": digest(out/"calibration.json"),
                                             "fit_complete_sha256": digest(out/"fit_complete.json"), "test_read": False})
    complete = {"schema": SCHEMA, "frozen_utc": utc(), "script_sha256": digest(__file__), "calibration_features_sha256": feature_sha,
                "fit_complete_sha256": digest(out/"fit_complete.json"), "final_test_read": False,
                "outputs_sha256": {n: digest(out/n) for n in ("calibration.json", "calibration_sources.csv", "thresholds_locked.json")}}
    write_json(out/"calibration_complete.json", complete)
    print("CALIBRATION FROZEN", complete["frozen_utc"], flush=True)


def source_bootstrap(data, decisions, base_decisions, samples, seed):
    """One resample per collection; shared draws across every model/endpoint."""
    meta = data["meta"].copy(); meta["i"] = np.arange(len(meta))
    source_rows = meta[["source_id", "collection", "label"]].drop_duplicates().sort_values("source_id").reset_index(drop=True)
    lookup = {s: i for i, s in enumerate(source_rows.source_id)}
    endpoints = {name: np.full((len(source_rows), 4), np.nan) for name in decisions}
    for sid, group in meta.groupby("source_id", sort=False):
        si = lookup[sid]; ii = group.i.to_numpy(int)
        marked = (~group.is_unmarked & group.label.eq(1)).to_numpy()
        unmarked = (group.is_unmarked & group.label.eq(1)).to_numpy()
        real = group.label.eq(0).to_numpy()
        for name, values in decisions.items():
            if marked.any(): endpoints[name][si, 0] = values[ii[marked]].mean()
            if unmarked.any(): endpoints[name][si, 1] = values[ii[unmarked]].mean()
            if real.any():
                endpoints[name][si, 2] = values[ii[real]].mean()
                endpoints[name][si, 3] = values[ii[real]].max()
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy(int) for _, group in source_rows.groupby("collection", sort=True)]
    draws = np.zeros((samples, len(source_rows)), dtype=np.int32)
    for stratum in strata:
        picked = rng.integers(0, len(stratum), size=(samples, len(stratum)))
        for j in range(samples): draws[j, stratum] = np.bincount(picked[j], minlength=len(stratum))
    result, names = {}, ("marked_tpr", "unmarked_tpr", "real_image_fpr", "real_source_fpr")
    bootstrap_values = {}
    base = endpoints["probe"]
    for method, values in endpoints.items():
        result[method] = {}
        for j, name in enumerate(names):
            valid = np.isfinite(values[:, j])
            if not valid.any():
                result[method][name] = None; continue
            w = draws[:, valid]
            boot = (w @ values[valid, j]) / w.sum(axis=1)
            bootstrap_values[method, name] = boot
            point = float(values[valid, j].mean())
            difference = values[valid, j]-base[valid, j]
            db = (w @ difference) / w.sum(axis=1)
            result[method][name] = {"estimate": point, "ci95": np.quantile(boot, [.025, .975]).tolist(), "sources": int(valid.sum()),
                                    "minus_probe": {"estimate": float(difference.mean()), "ci95": np.quantile(db, [.025, .975]).tolist(),
                                                    "one_sided_lower95": float(np.quantile(db, .05))}}
    for endpoint in names:
        if result["primary"][endpoint] is not None:
            comparisons = {}
            for reference in result:
                if reference == "primary" or result[reference][endpoint] is None:
                    continue
                difference = bootstrap_values["primary", endpoint]-bootstrap_values[reference, endpoint]
                comparisons[reference] = {"estimate": result["primary"][endpoint]["estimate"]-result[reference][endpoint]["estimate"],
                                          "ci95": np.quantile(difference, [.025, .975]).tolist()}
            result["primary"][endpoint]["comparisons"] = comparisons
    for method in result:
        shares = []
        if (method, "marked_tpr") in bootstrap_values and (method, "unmarked_tpr") in bootstrap_values:
            for share in (0., .25, .5, .75, 1.):
                mixture = share*bootstrap_values[method, "marked_tpr"]+(1-share)*bootstrap_values[method, "unmarked_tpr"]
                reference = share*bootstrap_values["probe", "marked_tpr"]+(1-share)*bootstrap_values["probe", "unmarked_tpr"]
                point = share*result[method]["marked_tpr"]["estimate"]+(1-share)*result[method]["unmarked_tpr"]["estimate"]
                probe_point = share*result["probe"]["marked_tpr"]["estimate"]+(1-share)*result["probe"]["unmarked_tpr"]["estimate"]
                shares.append({"marked_share": share, "tpr": point, "ci95": np.quantile(mixture, [.025, .975]).tolist(),
                               "gain_vs_probe": point-probe_point, "gain_ci95": np.quantile(mixture-reference, [.025, .975]).tolist(),
                               "interpretation": "Fixed-cut reweighting of paired marked and unmarked source outcomes; no threshold retuning."})
        result[method]["marked_shares_secondary"] = shares
    return result


def cp_upper(events, n, delta=.05):
    if not n: return None
    if events == n: return 1.0
    return float(beta.ppf(1-delta, events+1, n-events))


def harmful_switch_assessment(data, decision, baseline):
    meta = data["meta"]
    unmarked = (data["y"] == 1)&data["unmarked"]
    collections = sorted(meta.loc[unmarked, "collection"].unique())
    if not collections:
        return {"finite_sample_supported": False, "reason": "No unmarked AI sources"}
    delta = .05/len(collections)
    total = 0; weighted_upper = 0.; records = []
    for collection in collections:
        mask = unmarked & meta.collection.eq(collection).to_numpy()
        rows = pd.DataFrame({"source": meta.loc[mask, "source_id"].to_numpy(), "harm": baseline[mask]&~decision[mask]})
        events = rows.groupby("source").harm.max()
        n, count = len(events), int(events.sum())
        upper = cp_upper(count, n, delta)
        records.append({"collection": collection, "sources": n, "source_any_harm_events": count, "one_sided_upper": upper, "delta": delta})
        total += n; weighted_upper += n*upper
    upper = weighted_upper/total
    count = sum(r["source_any_harm_events"] for r in records)
    # With independent source events and fixed collection quotas,
    # product_g (1-p_g)^n_g <= (1-sum_g n_g*p_g/N)^N by log concavity.
    # Hence zero events bound the quota-weighted risk without assuming a
    # common Bernoulli risk across collections. This does NOT justify pooled
    # Clopper-Pearson bounds when any event is observed.
    zero_upper = float(-np.expm1(np.log(.05)/total)) if count == 0 else None
    chosen_upper = zero_upper if zero_upper is not None else upper
    return {"simultaneous_confidence": .95, "collection_bounds": records, "fixed_quota_weighted_harm_upper": upper,
            "sources": total, "source_any_harm_events": count, "pooled_zero_event_upper95": zero_upper,
            "unmarked_tpr_drop_upper": chosen_upper, "finite_sample_supported": bool(chosen_upper < NONINFERIORITY),
            "pooled_zero_bound": "Only when every source has zero harmful switches: 1-.05^(1/N), via log concavity at fixed quotas; no pooled nonzero-event binomial claim.",
            "per_collection_noninferiority_unresolved": any(r["one_sided_upper"] >= NONINFERIORITY for r in records),
            "assumptions": "Independent sources; iid/common-risk events within each named collection; fixed comparison, views and quotas. Per-collection bounds use a union bound. The pooled zero-event bound targets the fixed-quota mean any-view harm, which upper-bounds average-view TPR drop. Does not cover arbitrary shift."}


def evaluate_scope(data, models, policies, samples, seed, strongest_baseline="probe"):
    scores = {n: score(data, m) for n, m in models.items()}
    decisions = {n: s > (policies[n]["threshold"] if policies[n]["threshold"] is not None else np.inf) for n, s in scores.items()}
    bootstrap = source_bootstrap(data, decisions, decisions["probe"], samples, seed)
    output = {}
    for name in models:
        threshold = policies[name]["threshold"] if policies[name]["threshold"] is not None else np.inf
        record = rates(data, scores[name], threshold, decisions["probe"], strict=True)
        record["source_bootstrap"] = bootstrap[name]
        # Diagnostic AUC never determines the frozen operating point.
        real = data["y"] == 0
        record["auc"] = {}
        for state, positives in (("marked", (data["y"] == 1)&~data["unmarked"]), ("unmarked", (data["y"] == 1)&data["unmarked"])):
            mask = real | positives
            record["auc"][state] = float(roc_auc_score(data["y"][mask], scores[name][mask])) if real.any() and positives.any() else None
        width = policies[name]["abstention_width"]
        selected = np.abs(scores[name]-threshold) >= width if width is not None else np.zeros(len(scores[name]), bool)
        curve = []
        for multiplier in (0., .5, 1., 2., 4.):
            retained = np.abs(scores[name]-threshold) >= multiplier*width if width is not None else np.zeros(len(scores[name]), bool)
            curve.append({"calibration_band_multiplier": multiplier, "coverage": float(retained.mean()), "retained_images": int(retained.sum()),
                          "error_rate": float((decisions[name][retained] != data["y"][retained]).mean()) if retained.any() else None})
        retained_counts = []
        for (collection, condition, label), ii in data["meta"].groupby(["collection", "condition", "label"]).groups.items():
            ii = np.asarray(list(ii), int)
            retained_counts.append({"collection": collection, "condition": condition, "label": int(label), "images": len(ii), "retained": int(selected[ii].sum())})
        record["selective"] = {"coverage": float(selected.mean()), "selected_images": int(selected.sum()),
                               "error_rate": float((decisions[name][selected] != data["y"][selected]).mean()) if selected.any() else None,
                               "risk_coverage_curve": curve, "retained_counts": retained_counts,
                               "interpretation": "Descriptive fixed negative-calibration band; no selective-risk or shifted-coverage guarantee."}
        source_events = pd.DataFrame({"source": data["meta"].source_id, "label": data["y"], "decision": decisions[name]})
        real_events = source_events.loc[source_events.label.eq(0)].groupby("source").decision.max()
        record["real_source_binomial"] = {"events": int(real_events.sum()), "sources": len(real_events), "one_sided_upper95": cp_upper(int(real_events.sum()), len(real_events)),
                                          "interpretation": "Model-based iid common-risk source bound; pooled fixed collection quotas do not establish exact mixture coverage."}
        output[name] = record
    primary = output["primary"]["source_bootstrap"]
    gain = primary["marked_tpr"]["minus_probe"] if primary["marked_tpr"] else None
    ni = primary["unmarked_tpr"]["minus_probe"] if primary["unmarked_tpr"] else None
    independent_harm = harmful_switch_assessment(data, decisions["primary"], decisions["probe"])
    analytic_zero = bool(models["primary"]["kind"] == "probe" and policies["primary"]["threshold"] == policies["probe"]["threshold"])
    if analytic_zero:
        independent_harm["finite_sample_supported"] = True
        independent_harm["analytic_identity"] = "The primary fallback and its threshold exactly equal the probe for every input. No new-method gain."
    strongest = primary["marked_tpr"].get("comparisons", {}).get(strongest_baseline) if primary["marked_tpr"] else None
    bootstrap_ni = bool(ni and ni["one_sided_lower95"] > -NONINFERIORITY)
    nonzero_harm = independent_harm.get("source_any_harm_events", 0) > 0
    ni_supported = bool(independent_harm["finite_sample_supported"] or (nonzero_harm and bootstrap_ni))
    conclusion = {"marked_gain_supported": bool(gain and gain["ci95"][0] > 0),
                  "strongest_baseline_selected_on_tune": strongest_baseline,
                  "gain_vs_tune_selected_baseline_supported": bool(strongest and strongest["ci95"][0] > 0),
                  "bootstrap_noninferiority_descriptive": bootstrap_ni,
                  "unmarked_loss_one_sided_upper95": -ni["one_sided_lower95"] if ni else None,
                  "unmarked_noninferiority_supported": ni_supported,
                  "finite_sample_harm_noninferiority_supported": independent_harm["finite_sample_supported"],
                  "noninferiority_inference": "Conditional finite-sample source-harm bound" if independent_harm["finite_sample_supported"] else ("Paired source bootstrap approximation; no finite-sample certificate" if nonzero_harm else "Zero-event bound too wide; degenerate zero-loss bootstrap does not establish noninferiority"),
                  "independent_source_harm_assessment": independent_harm,
                  "realized_source_fpr_within_nominal": bool(output["primary"]["real_source_fpr"] is not None and output["primary"]["real_source_fpr"] <= RISK),
                  "unmarked_tolerance": NONINFERIORITY, "no_model_selection_from_this_test": True,
                  "caveat": "Preregistered primary/NI criteria; secondary multi-model and subgroup intervals are descriptive unless a simultaneous correction is separately declared."}
    if models["primary"]["kind"] == "probe":
        conclusion["candidate_status"] = "Safe zero fallback; no new-method gain can be claimed"
    else:
        conclusion["candidate_status"] = "Primary criteria met" if all(conclusion[k] for k in ("marked_gain_supported", "gain_vs_tune_selected_baseline_supported", "unmarked_noninferiority_supported", "realized_source_fpr_within_nominal")) else "One or more prespecified criteria failed or are unresolved"
    return {"images": len(data["meta"]), "sources": data["meta"].source_id.nunique(), "systems": output, "primary_assessment": conclusion}, scores, decisions


def subset_data(data, mask):
    return {"meta": data["meta"].loc[mask].reset_index(drop=True), "features": data["features"][mask], "z": data["z"][mask],
            "y": data["y"][mask], "unmarked": data["unmarked"][mask], "queries": data["queries"]}


def confirmatory_coverage(data):
    """Check the fixed primary sample and paired-view coverage; never select it."""
    meta = data["meta"]
    ai = meta.loc[meta.label.eq(1)]
    sources = ai[["source_id", "collection"]].drop_duplicates()
    family_counts = sources.groupby("collection").size().to_dict()
    problems = []
    if len(sources) != 600 or len(family_counts) != 3 or any(n != 200 for n in family_counts.values()):
        problems.append("The primary protocol requires 600 AI sources, with 200 in each of three held-out collections")
    for sid, group in ai.groupby("source_id", sort=False):
        marked = group.loc[~group.is_unmarked]
        unmarked = group.loc[group.is_unmarked]
        if set(marked.embedded_payload) != set(data["queries"]):
            problems.append(f"AI source {sid} lacks a registered marked payload")
        if set(unmarked.condition) != set(COMMON_CONDITIONS) or unmarked.condition.duplicated().any():
            problems.append(f"AI source {sid} lacks exactly one unmarked image per common view")
        for payload, images in marked.groupby("embedded_payload"):
            if set(images.condition) != set(COMMON_CONDITIONS) or images.condition.duplicated().any():
                problems.append(f"AI source {sid}/{payload} lacks exactly one marked image per common view")
    real_coverage = meta.loc[meta.label.eq(0)].groupby("source_id").condition.agg(set)
    if real_coverage.empty or any(v != set(COMMON_CONDITIONS) for v in real_coverage):
        problems.append("Every supplied real source must cover all common views")
    return {"valid": not problems, "AI_sources": len(sources), "family_counts": family_counts,
            "problems": problems, "scope": "The four fixed primary views and every registered payload"}


def verification_report(frame, calibration):
    reports = {}
    for detector in METHODS:
        subset = frame.loc[frame.detector.eq(detector)].copy()
        policy = calibration["verification_policies"][detector]
        cut = policy["threshold"] if policy["threshold"] is not None else np.inf
        subset["accepted_margin"] = subset.signed_margin > cut
        subsets = {"photographs_unmarked": subset.loc[subset.label.eq(0)&subset.is_unmarked],
                   "AI_unmarked": subset.loc[subset.label.eq(1)&subset.is_unmarked],
                   "wrong_payload_queries": subset.loc[~subset.is_unmarked & subset.query_payload.ne(subset.embedded_payload)],
                   "correct_payload_queries": subset.loc[~subset.is_unmarked & subset.query_payload.eq(subset.embedded_payload)]}
        records = {}
        for name, group in subsets.items():
            if group.empty:
                records[name] = {"images": 0, "sources": 0}; continue
            by_image = group.groupby("image_id")[["accepted_margin", "full_match"]].max()
            by_source = group.groupby("source_id")[["accepted_margin", "full_match"]].max()
            records[name] = {"images": len(by_image), "sources": len(by_source), "margin_image_rate": float(by_image.accepted_margin.mean()),
                             "margin_source_any_rate": float(by_source.accepted_margin.mean()), "native_full_match_image_rate": float(by_image.full_match.mean()),
                             "native_full_match_source_any_rate": float(by_source.full_match.mean())}
        reports[detector] = {"policy": policy, "results": records, "claim": "Expected registered-payload evidence; neither margin acceptance nor native recovery authenticates an issuer. Photographic calibration does not guarantee AI or wrong-payload false matches."}
    return reports


def evaluate(args):
    out = args.output.resolve()
    fit_lock, models, protocol = frozen(out)
    calibration_lock = verify_complete(out, "calibration_complete.json", {"fit_complete_sha256": digest(out/"fit_complete.json")})
    if calibration_lock is None:
        raise RuntimeError("Independent final calibration is not locked")
    feature_sha = digest(args.features)
    result_dir = out/"evaluations"/args.name
    result_dir.mkdir(parents=True, exist_ok=True)
    if (result_dir/"complete.json").exists():
        old = load_json(result_dir/"complete.json")
        if old["test_features_sha256"] != feature_sha or old["fit_complete_sha256"] != digest(out/"fit_complete.json") or old["calibration_complete_sha256"] != digest(out/"calibration_complete.json") or old["script_sha256"] != digest(__file__):
            raise RuntimeError("Existing evaluation lock differs; final test may not be silently rerun")
        for n, sha in old["outputs_sha256"].items():
            if digest(result_dir/n) != sha: raise RuntimeError("Evaluation output changed")
        print("Verified completed evaluation; scores not recomputed", flush=True); return
    frame, queries = read_features(args.features, ("test", "fresh_test", "final_test"), protocol["registered_queries"])
    dev = pd.read_csv(out/"development_sources.csv", dtype=str, keep_default_na=False)
    cal = pd.read_csv(out/"calibration_sources.csv", dtype=str, keep_default_na=False)
    if set(frame.source_id) & (set(dev.source_id)|set(cal.source_id)):
        raise ValueError("Final test overlaps fitting/selection/calibration sources")
    calibration = load_json(out/"calibration.json")
    selection = load_json(out/"selection.json")
    unseen_conditions = sorted(set(frame.condition)-set(calibration["conditions"]))
    if unseen_conditions and not args.allow_unseen_conditions:
        raise ValueError(f"Test conditions outside calibration family: {unseen_conditions}; use explicit diagnostic flag")
    report = {"schema": SCHEMA, "evaluated_utc": utc(), "test_features_sha256": feature_sha,
              "conditions_outside_calibration_family": unseen_conditions, "no_test_tuning": True,
              "bootstrap_samples": args.bootstrap_samples, "primary_common_conditions": list(COMMON_CONDITIONS),
              "confirmatory_primary_system": PRIMARY_SYSTEM, "models": {}}
    predictions = []
    for di, detector in enumerate(METHODS):
        for bi, backbone in enumerate(BACKBONES):
            key = f"{detector}/{backbone}"; data = view(frame, detector, backbone)
            mask = data["meta"].condition.isin(COMMON_CONDITIONS).to_numpy()
            common = subset_data(data, mask)
            if common["meta"].empty:
                raise ValueError("Final test lacks primary common conditions")
            record, _, _ = evaluate_scope(common, models[key], calibration["origin_policies"][key], args.bootstrap_samples, SEED+100*di+bi, selection[key]["strongest_baseline"])
            if mask.all():
                all_record = record
                scores = {n: score(data, m) for n, m in models[key].items()}
                decisions = {n: s > (calibration["origin_policies"][key][n]["threshold"] if calibration["origin_policies"][key][n]["threshold"] is not None else np.inf) for n, s in scores.items()}
            else:
                all_record, scores, decisions = evaluate_scope(data, models[key], calibration["origin_policies"][key], args.bootstrap_samples, SEED+100*di+bi, selection[key]["strongest_baseline"])
            record["interpretation"] = "Single confirmatory system" if key == PRIMARY_SYSTEM else "Secondary system; no primary selection from this result"
            report["models"][key] = {"primary_common": record, "all_supplied_conditions_diagnostic": all_record}
            if key == PRIMARY_SYSTEM:
                coverage = confirmatory_coverage(common)
                report["confirmatory_assessment"] = {"system": key, "coverage": coverage, **record["primary_assessment"]}
                if not coverage["valid"]:
                    report["confirmatory_assessment"]["candidate_status"] = "Non-confirmatory: fixed sample or paired-view coverage incomplete"
            for name in models[key]:
                output = data["meta"][["image_id", "source_id", "collection", "label", "watermark_method", "embedded_payload", "condition"]].copy()
                output["detector"] = detector; output["backbone"] = backbone; output["system"] = name
                output["score"] = scores[name]; output["decision"] = decisions[name].astype(int)
                predictions.append(output)
            print(key, record["primary_assessment"], flush=True)
    report["verification"] = verification_report(frame, calibration)
    write_json(result_dir/"results.json", finite_json(report))
    pd.concat(predictions, ignore_index=True).to_csv(result_dir/"predictions.csv", index=False)
    sources = frame[["source_id", "collection", "label"]].drop_duplicates().sort_values("source_id")
    sources.to_csv(result_dir/"test_sources.csv", index=False)
    complete = {"schema": SCHEMA, "completed_utc": utc(), "script_sha256": digest(__file__), "test_features_sha256": feature_sha,
                "fit_complete_sha256": digest(out/"fit_complete.json"), "calibration_complete_sha256": digest(out/"calibration_complete.json"),
                "bootstrap_samples": args.bootstrap_samples, "source_count": len(sources), "retuning": False,
                "outputs_sha256": {n: digest(result_dir/n) for n in ("results.json", "predictions.csv", "test_sources.csv")}}
    write_json(result_dir/"complete.json", complete)
    print("EVALUATION COMPLETE", result_dir, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("fit", "calibrate", "evaluate"):
        child = sub.add_parser(stage)
        child.add_argument("--features", type=Path, required=True)
        if stage == "evaluate":
            child.add_argument("--name", default="fresh_test")
            child.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAPS)
            child.add_argument("--allow-unseen-conditions", action="store_true")
    args = parser.parse_args()
    if args.stage == "evaluate":
        if args.bootstrap_samples < 1 or not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
            raise ValueError("Positive bootstrap count and a plain evaluation name required")
    {"fit": fit, "calibrate": calibrate, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
