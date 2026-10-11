"""Retrospective, source-disjoint fusion fitting and negative-only calibration.

Earlier probes, splits, images and results are read-only. The optional missing
CLIP extraction is scheduled separately by the root agent; this script is CPU
only. Core-13 and all-16 calibration are separate prespecified diagnostics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from phase4_domain_generalization import (
    ROOT, CONTROLS, ORIGINAL, HELD, assemble_frame, digest, features, logits,
    read_matrix, systems, fpr_threshold, write_json,
)

DEST = ROOT / "outputs/phase5/calibration"
FRESH = ROOT / "outputs/phase4/fresh_replication"
SEEDS = list(range(20260920, 20260925))
ALPHA = .05
BOOTSTRAP = 1000
COLLECTIONS = ["coco", "openimages"]
LOO = DEST / "sd_family_excluded"


def utc():
    return datetime.now(timezone.utc).isoformat()


def ranked(ids, seed, collection):
    return sorted(ids, key=lambda value: hashlib.sha256(
        f"phase5|{seed}|validation-half|{collection}|{value}".encode()).hexdigest())


def old_head_paths():
    return (list(ORIGINAL.glob("*.npz")) +
            [CONTROLS / "split_seeds" / str(seed) / f"{b}{suffix}_full.npz"
             for seed in SEEDS for b in ["resnet", "clip"] for suffix in ["", "_fusion"]])


def lock_protocol():
    target = DEST / "protocol.json"
    if target.exists():
        saved = json.loads(target.read_text())
        for name, value in saved["input_sha256"].items():
            if digest(ROOT / name) != value:
                raise ValueError(f"Locked input changed: {name}")
        if digest(DEST / "source_roles.csv") != saved["source_roles_sha256"]:
            raise ValueError("Locked calibration partitions changed")
        return saved
    frame, frames, paths, eligible = assemble_frame()
    sources = frame[["source_id", "collection", "label"]].drop_duplicates("source_id")
    assignments = pd.read_csv(CONTROLS / "split_seeds/assignments.csv", keep_default_na=False)
    result = []
    for seed in SEEDS:
        split = assignments.loc[assignments.seed.eq(seed)].set_index("source_id").split
        src = sources.assign(locked_split=sources.source_id.map(split)).copy()
        role = dict(zip(src.source_id, src.locked_split))
        for collection, part in src.loc[src.locked_split.eq("val")].groupby("collection"):
            ids = ranked(part.source_id, seed, collection)
            for i, identifier in enumerate(ids):
                role[identifier] = "fusion_fit" if i < len(ids) // 2 else "calibration"
        src["role"] = src.source_id.map(role)
        src.insert(0, "seed", seed)
        result.append(src)
        if src.loc[src.collection.eq("midjourney"), "role"].ne("test").any():
            raise ValueError("Midjourney entered fitting/calibration")
        counts = src.loc[src.role.eq("calibration") & src.label.eq(0)].collection.value_counts()
        if counts.to_dict() != {"coco": 144, "openimages": 75}:
            raise ValueError("Unexpected calibration source counts")
        for role_name in ["train", "fusion_fit", "calibration"]:
            ids = set(src.loc[src.role.eq(role_name), "source_id"])
            if not frame.loc[eligible & frame.source_id.isin(ids)].groupby("source_id").size().eq(13).all():
                raise ValueError("Expected thirteen fitting/calibration views per source")
    roles = pd.concat(result, ignore_index=True)
    roles.to_csv(DEST / "source_roles.csv", index=False)
    inputs = list(paths) + [CONTROLS / "split_seeds/assignments.csv", ROOT / "data/manifest_derived.csv",
        ROOT / "outputs/active_scores.csv", FRESH / "manifest.csv", FRESH / "summary.json"]
    inputs += old_head_paths()
    inputs += [ROOT / "outputs/resnet_embeddings.npz", ROOT / "outputs/clip_train_embeddings.npz",
        ROOT / "outputs/clip_eval_embeddings.npz", CONTROLS / "clip_heldout_embeddings.npz"]
    inputs += [CONTROLS / f"{b}_{kind}_embeddings.npz"
        for b in ["resnet", "clip"] for kind in ["openimages", "unmarked", "union"]]
    inputs += [FRESH / f"{b}_embeddings.npz" for b in ["resnet", "clip"]]
    protocol = {
        "locked_at_utc": utc(), "seeds": SEEDS, "alpha": ALPHA,
        "source_roles_sha256": digest(DEST / "source_roles.csv"),
        "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs},
        "role_counts_per_seed": roles.groupby(["seed", "role", "collection"]).size().reset_index(name="sources").to_dict("records"),
        "partition_rule": "Preserve the locked original training/test source split. Within each validation collection, SHA256-sort by phase5|seed|validation-half|collection|source_id. First floor(n/2) fit the new fusion; the remainder calibrate. Every derivative follows its source. Midjourney remains test only.",
        "probe_rule": "Existing full Phase3 probe for each seed/backbone stays fixed. Fit ten new balanced logistic fusion heads on only the fusion-fit source half and thirteen eligible views, C=1 L2, max_iter=3000, random_state=0. Inputs are unchanged probe probability and payload bit agreement. Gate remains agreement>=13/16.",
        "processing_families": {
            "core13": "Benchmark's two identity views (real includes wrong-payload spoof) plus eleven fitting-eligible edits. Excludes lowres severity2, grid-shift/JPEG/noise regenerate severity2, and spectral severity1.",
            "all16": "Same two identity and all fourteen edit views. Weak three views are excluded from model fitting but deliberately included in this new retrospective negative calibration, so they are no longer held out from calibration here."},
        "policies": {
            "per_image_core13": "Minimum observed negative core13 calibration logit with <=5% empirical copy FPR; classify score>=threshold, preserving whole ties. Independent of fitting; finite-source population control is not claimed.",
            "conformal_core13": "For each real collection, take maximum score over each calibration source's complete core13 bundle; k=ceil((n+1)*(1-alpha)). Cut is kth smallest max (1-indexed), or +infinity if k>n. Classify score strictly greater than max(COCO cut,Open Images cut), preserving whole ties.",
            "conformal_all16": "Same negative-source maximum rule over complete all16 bundles. One global cut deploys without an image collection or attack label."},
        "guarantee": "For a fixed fitted score function and exchangeable calibration/new real source bundles from either prespecified collection and the same calibrated processing family, strict exceedance controls the marginal probability of at least one false positive in a new bundle by alpha (with finite bound (n+1-k)/(n+1)). This is neither a high-confidence population-FPR guarantee nor a guarantee for new collections, changed encoding, arbitrary attacks or adversarial source selection. Max of collection cuts is conservative within either calibrated collection; no mixture labels are used at decision time.",
        "evaluation": "All five original seed tests (sources overlap; Midjourney reused) and already-evaluated new400 cached sample as retrospective follow-up. Evaluate core13/all16 source bundles and fresh four views. The fresh JPEG95 acquisition/encoding and four-view protocol differ from calibration; transfer is measured and not guaranteed. No test score selects a model, scope, hyperparameter or threshold.",
        "bootstrap": {"replicates": BOOTSTRAP, "seed": 20261020,
            "unit": "Source, stratified by collection; reuse the same global source bootstrap weights across all five overlapping tests and all policies. Confidence intervals describe fixed-model/fixed-threshold test sampling only; they do not cover calibration uncertainty."},
        "retrospective": True, "new400_is_untouched": False, "test_threshold_tuning": False,
        "script_sha256_at_lock": digest(__file__),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "python": platform.python_version(), "device": "CPU fits/analysis; any missing CLIP vectors separately scheduled",
    }
    write_json(target, protocol)
    return protocol


def prepare_missing():
    protocol = lock_protocol()
    target = DEST / "calibration_extra_manifest.csv"
    if target.exists():
        saved = json.loads((DEST / "missing_features.json").read_text())
        if digest(target) != saved["manifest_sha256"]:
            raise ValueError("Missing-feature manifest changed")
        print(json.dumps(saved), flush=True)
        return
    roles = pd.read_csv(DEST / "source_roles.csv")
    cal = set(roles.loc[roles.role.eq("calibration") & roles.label.eq(0), "source_id"])
    frame, _, _, _ = assemble_frame()
    raw = pd.read_csv(ROOT / "data/manifest_derived.csv", keep_default_na=False, dtype={"payload": str})
    missing = raw.loc[raw.source_id.isin(cal) & ~raw.image_id.isin(frame.image_id)].copy()
    if not all((a, int(s)) in HELD for a, s in zip(missing.attack, missing.severity)):
        raise ValueError("Unexpected missing calibration view")
    known = pd.read_csv(ROOT / "outputs/active_scores.csv", keep_default_na=False)
    active = dict(zip(known.image_id, known.active_score))
    import cv2
    from aiwstudy.watermark import DwtDctWatermark
    cv2.setNumThreads(1)
    def prepare(row):
        pixels = cv2.imread(row["path"])
        if pixels is None:
            raise ValueError("Existing calibration pixels unavailable")
        row["active_score"] = active.get(row["image_id"])
        if row["active_score"] is None:
            row["active_score"] = DwtDctWatermark("1011010010110100", 12).detect(pixels).bit_accuracy
        row["collection"] = "coco"
        row["output_file_sha256"] = digest(row["path"])
        return row
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(prepare, missing.to_dict("records")))
    pd.DataFrame(records, columns=[*raw.columns, "active_score", "collection", "output_file_sha256"]).to_csv(target, index=False)
    audit = {"locked_at_utc": utc(), "protocol_sha256": digest(DEST / "protocol.json"),
        "manifest_sha256": digest(target), "rows": len(records), "sources": missing.source_id.nunique(),
        "inference_started": False, "new_images_generated": False,
        "resnet": "All requested vectors already exist in outputs/resnet_embeddings.npz.",
        "clip": "Extract only these missing original real-calibration weak views with the unchanged study CLIP preprocessing/checkpoint; exact row order and manifest hash required.",
        "clip_output": "outputs/phase5/calibration/clip_calibration_extra_embeddings.npz",
        "output_keys": ["image_id", "embedding", "manifest_sha256"],
        "purpose": "Complete all16 negative bundles without selecting sources on feature coverage or test score."}
    write_json(DEST / "missing_features.json", audit)
    print(json.dumps(audit), flush=True)


def base_frame(extra=False):
    frame, frames, paths, eligible = assemble_frame()
    if extra:
        new = pd.read_csv(DEST / "calibration_extra_manifest.csv", keep_default_na=False)
        frame = pd.concat([frame, new], ignore_index=True)
    return frame, frames, paths


def fit_fusions():
    protocol = lock_protocol()
    if (DEST / "heads_locked.json").exists():
        saved = json.loads((DEST / "heads_locked.json").read_text())
        for name, value in saved["heads_sha256"].items():
            if digest(ROOT / name) != value:
                raise ValueError("Frozen Phase5 fusion changed")
        print("Reused ten previously locked source-disjoint fusion heads.", flush=True)
        return
    roles = pd.read_csv(DEST / "source_roles.csv")
    frame, frames, paths = base_frame()
    eligible = np.array([(a, int(s)) not in HELD for a, s in zip(frame.attack, frame.severity)]) & ~frame.group.eq("S-U-A").to_numpy()
    records = []
    for backbone in ["resnet", "clip"]:
        x = features(backbone, frames, paths)
        for seed in SEEDS:
            probe = CONTROLS / "split_seeds" / str(seed) / f"{backbone}_full.npz"
            fusion = DEST / str(seed) / f"{backbone}_fusion.npz"
            fusion.parent.mkdir(parents=True, exist_ok=True)
            fit_ids = set(roles.loc[roles.seed.eq(seed) & roles.role.eq("fusion_fit"), "source_id"])
            chosen = frame.source_id.isin(fit_ids).to_numpy() & eligible
            design = np.column_stack([expit(logits(x[chosen], probe)), frame.loc[chosen, "active_score"]])
            labels = frame.loc[chosen, "label"].to_numpy()
            meta_path = fusion.with_suffix(".json")
            if fusion.exists():
                meta = json.loads(meta_path.read_text())
                if meta["head_sha256"] != digest(fusion) or meta["protocol_sha256"] != digest(DEST / "protocol.json"):
                    raise ValueError("New locked fusion changed")
            else:
                started, begin = utc(), time.perf_counter()
                model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(design, labels)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise ValueError("Fusion failed to converge")
                np.savez(fusion, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)
                meta = {"fit_started_at_utc": started, "fit_completed_at_utc": utc(),
                    "seed": seed, "backbone": backbone, "sources": len(fit_ids), "rows": len(labels),
                    "real_rows": int((labels == 0).sum()), "AI_rows": int((labels == 1).sum()),
                    "n_iter": model.n_iter_.tolist(), "converged": True,
                    "seconds": time.perf_counter() - begin, "protocol_sha256": digest(DEST / "protocol.json"),
                    "head_sha256": digest(fusion), "probe_sha256": digest(probe),
                    "fusion_fit_source_hash": hashlib.sha256("\n".join(sorted(fit_ids)).encode()).hexdigest(),
                    "calibration_or_test_sources_used_for_fit": False}
                write_json(meta_path, meta)
            records.append(meta)
            print(f"Frozen {seed} {backbone} fusion, sources={len(fit_ids)} rows={len(labels)}", flush=True)
        del x
    old = {p.relative_to(ROOT).as_posix(): digest(p) for p in old_head_paths()}
    if any(protocol["input_sha256"][name] != value for name, value in old.items()):
        raise ValueError("Earlier model changed")
    write_json(DEST / "heads_locked.json", {"locked_at_utc": utc(), "fits": records,
        "protocol_sha256": digest(DEST / "protocol.json"), "earlier_heads_unchanged": True,
        "heads_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in DEST.glob("*/*_fusion.npz")},
        "earlier_probe_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in old_head_paths() if "fusion" not in p.name},
        "usable_systems": "Fixed original probe, new source-disjoint fusion, and fixed agreement13/16 gate.",
        "test_scoring_started": False})


def conformal_cut(frame, scores, family):
    chosen = frame.label.eq(0).to_numpy(copy=True)
    if family == "core13":
        chosen &= np.array([(a, int(s)) not in HELD for a, s in zip(frame.attack, frame.severity)])
    real = frame.loc[chosen, ["source_id", "collection"]].assign(score=scores[chosen])
    by_source = real.groupby(["collection", "source_id"]).score.agg(["max", "count"])
    if not by_source["count"].eq(13 if family == "core13" else 16).all():
        raise ValueError("Incomplete negative-source calibration bundle")
    cuts = {}
    for collection in COLLECTIONS:
        values = by_source.loc[collection, "max"].to_numpy()
        n = len(values)
        k = math.ceil((n + 1) * (1 - ALPHA))
        q = float(np.partition(values, k - 1)[k - 1]) if k <= n else None
        accepted = np.zeros(n, dtype=bool) if q is None else values > q
        cuts[collection] = {"threshold_logit": q, "calibration_real_sources": n,
            "k_one_indexed": k, "finite_source_marginal_bound": (n + 1 - k) / (n + 1) if k <= n else 0.,
            "calibration_source_any_fpr": float(accepted.mean()),
            "comparison": ">", "infinite_threshold": q is None}
    finite = all(value["threshold_logit"] is not None for value in cuts.values())
    threshold = max(value["threshold_logit"] for value in cuts.values()) if finite else None
    return {"threshold_logit": threshold, "infinite_threshold": threshold is None,
        "comparison": ">", "collection_cuts": cuts, "globally_deployable": True}


def complete_features(backbone, frames, paths):
    x = features(backbone, frames, paths)
    extra_path = DEST / "calibration_extra_manifest.csv"
    extra = pd.read_csv(extra_path, keep_default_na=False)
    if len(extra):
        if backbone == "resnet":
            z = read_matrix([ROOT / "outputs/resnet_embeddings.npz"], extra.image_id)
        else:
            target = DEST / "clip_calibration_extra_embeddings.npz"
            with np.load(target, allow_pickle=True) as saved:
                if str(saved["manifest_sha256"]) != digest(extra_path) or not np.array_equal(saved["image_id"].astype(str), extra.image_id.to_numpy()):
                    raise ValueError("Missing CLIP calibration features not locked to exact manifest")
                z = saved["embedding"].astype(np.float32)
        x = np.concatenate([x, z])
    return x


def calibrate():
    lock_protocol()
    if (DEST / "thresholds_locked.json").exists():
        saved = json.loads((DEST / "thresholds_locked.json").read_text())
        if saved["heads_locked_sha256"] != digest(DEST / "heads_locked.json"):
            raise ValueError("Calibrated head lock changed")
        if saved["protocol_sha256"] != digest(DEST / "protocol.json"):
            raise ValueError("Calibrated protocol changed")
        print("Reused all ninety locked calibration thresholds.", flush=True)
        return
    roles = pd.read_csv(DEST / "source_roles.csv")
    frame, frames, paths = base_frame(extra=True)
    answer = {}
    for backbone in ["resnet", "clip"]:
        x = complete_features(backbone, frames, paths)
        for seed in SEEDS:
            selected = roles.loc[roles.seed.eq(seed) & roles.role.eq("calibration")]
            chosen = frame.source_id.isin(selected.source_id).to_numpy()
            cal = frame.loc[chosen].reset_index(drop=True)
            # Only real sources need the three additional weak-view rows. AI core13
            # rows may coexist with already-cached extra views; AI never sets a cut.
            probe = CONTROLS / "split_seeds" / str(seed) / f"{backbone}_full.npz"
            fusion = DEST / str(seed) / f"{backbone}_fusion.npz"
            scores = systems(x[chosen], cal.active_score.to_numpy(), probe, fusion)
            entry = {"seed": seed, "backbone": backbone, "selected_at_utc": utc(),
                "fusion_sha256": digest(fusion), "probe_sha256": digest(probe), "systems": {}}
            core_real = cal.label.eq(0).to_numpy() & np.array([(a, int(s)) not in HELD for a, s in zip(cal.attack, cal.severity)])
            for name, score in scores.items():
                cut, rate = fpr_threshold(score[core_real])
                entry["systems"][name] = {"per_image_core13": {"threshold_logit": cut,
                    "comparison": ">=", "calibration_copy_fpr": rate, "calibration_real_sources": 219,
                    "calibration_real_rows": int(core_real.sum()), "infinite_threshold": False},
                    **{f"conformal_{family}": conformal_cut(cal, score, family) for family in ["core13", "all16"]}}
            answer[f"{seed}_{backbone}"] = entry
        del x
    write_json(DEST / "thresholds_locked.json", {"locked_at_utc": utc(),
        "protocol_sha256": digest(DEST / "protocol.json"), "heads_locked_sha256": digest(DEST / "heads_locked.json"),
        "missing_manifest_sha256": digest(DEST / "calibration_extra_manifest.csv"),
        "calibration_only": True, "test_threshold_tuning": False, "models": answer})
    print("Frozen 90 independent calibration thresholds (ten model pairs, three systems, three policies).", flush=True)


def decide(scores, policy):
    if policy["threshold_logit"] is None:
        return np.zeros(len(scores), dtype=bool)
    if policy["comparison"] == ">":
        return scores > policy["threshold_logit"]
    return scores >= policy["threshold_logit"]


def lock_loo_protocol():
    """Additional detector for the prospective paired SD-caption score stage."""
    lock_protocol()
    LOO.mkdir(parents=True, exist_ok=True)
    target = LOO / "protocol.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["source_roles_sha256"] != digest(LOO / "source_roles.csv"):
            raise ValueError("SD-exclusion role lock changed")
        if saved["parent_protocol_sha256"] != digest(DEST / "protocol.json"):
            raise ValueError("SD-exclusion parent protocol changed")
        return saved
    roles = pd.read_csv(DEST / "source_roles.csv")
    roles = roles.loc[roles.seed.eq(SEEDS[0])].copy()
    excluded = roles.collection.isin(["stable-diffusion", "midjourney"]) & ~roles.role.eq("test")
    roles.loc[excluded, "role"] = "excluded_generator_from_fitting_and_calibration"
    roles.to_csv(LOO / "source_roles.csv", index=False)
    protocol = {"locked_at_utc": utc(), "seed": SEEDS[0], "parent_protocol_sha256": digest(DEST / "protocol.json"),
        "source_roles_sha256": digest(LOO / "source_roles.csv"), "heldout_AI_families": ["stable-diffusion", "midjourney"],
        "training_rule": "Train new balanced L2 logistic probes on unchanged core13 cached vectors for all original locked training photographs and DALL-E 3 sources only. Exclude every Stable Diffusion and Midjourney source from probe training, fusion fitting and calibration. Midjourney stays in original tests. New fusion fits the existing source-disjoint validation fusion-fit half, also excluding SD. Real-negative calibration uses the same independent COCO144/Open Images75 source half and the same three prespecified policies.",
        "probe_hyperparameters": {"class_weight": "balanced", "penalty": "L2", "C": 1., "max_iter": 3000, "random_state": 0},
        "role_counts": roles.groupby(["role", "collection"]).size().reset_index(name="sources").to_dict("records"),
        "caveat": "DALL-E-only synthetic training is a restricted, different training mix, so a difference from the original detector cannot isolate the effect of generator holdout. The frozen ImageNet/CLIP backbone may have prior exposure to related generators or internet imagery; this control guarantees no SD-family supervised feature-head fitting, not generator ignorance of pretrained representations. The original source pool and new400 have already been examined. Future caption-paired SD images are to be scored after this head/threshold lock without retuning.",
        "test_threshold_tuning": False, "paired_test_scored": False,
        "script_sha256_at_lock": digest(__file__)}
    write_json(target, protocol)
    return protocol


def fit_loo():
    protocol = lock_loo_protocol()
    if (LOO / "heads_locked.json").exists():
        saved = json.loads((LOO / "heads_locked.json").read_text())
        for name, value in saved["heads_sha256"].items():
            if digest(ROOT / name) != value:
                raise ValueError("SD-exclusion head changed")
        print("Reused four locked SD-family-exclusion heads.", flush=True)
        return
    roles = pd.read_csv(LOO / "source_roles.csv")
    frame, frames, paths = base_frame()
    eligible = np.array([(a, int(s)) not in HELD for a, s in zip(frame.attack, frame.severity)]) & ~frame.group.eq("S-U-A").to_numpy()
    records = []
    for backbone in ["resnet", "clip"]:
        x = features(backbone, frames, paths)
        for name, role in [("probe", "train"), ("fusion", "fusion_fit")]:
            chosen = frame.source_id.isin(roles.loc[roles.role.eq(role), "source_id"]).to_numpy() & eligible
            if frame.loc[chosen, "collection"].isin(["stable-diffusion", "midjourney"]).any():
                raise ValueError("Excluded AI family entered fitting")
            head = LOO / f"{backbone}_{name}.npz"
            meta_path = head.with_suffix(".json")
            labels = frame.loc[chosen, "label"].to_numpy()
            design = x[chosen] if name == "probe" else np.column_stack([
                expit(logits(x[chosen], LOO / f"{backbone}_probe.npz")), frame.loc[chosen, "active_score"]])
            if head.exists():
                meta = json.loads(meta_path.read_text())
                if meta["head_sha256"] != digest(head) or meta["protocol_sha256"] != digest(LOO / "protocol.json"):
                    raise ValueError("SD-exclusion fit artifact changed")
            else:
                model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
                begin, started = time.perf_counter(), utc()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(design, labels)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise ValueError("SD-exclusion fit failed to converge")
                np.savez(head, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)
                meta = {"fit_started_at_utc": started, "fit_completed_at_utc": utc(), "backbone": backbone,
                    "model": name, "role": role, "sources": frame.loc[chosen, "source_id"].nunique(),
                    "rows": len(labels), "real_rows": int((labels == 0).sum()), "AI_rows": int((labels == 1).sum()),
                    "collection_counts": frame.loc[chosen].groupby("collection").source_id.nunique().to_dict(),
                    "head_sha256": digest(head), "protocol_sha256": digest(LOO / "protocol.json"),
                    "n_iter": model.n_iter_.tolist(), "converged": True, "seconds": time.perf_counter() - begin,
                    "SD_or_Midjourney_sources_used_for_fit": False}
                write_json(meta_path, meta)
            records.append(meta)
            print(f"Frozen SD-excluded {backbone} {name}: {len(labels)} copies.", flush=True)
        del x
    write_json(LOO / "heads_locked.json", {"locked_at_utc": utc(), "fits": records,
        "protocol_sha256": digest(LOO / "protocol.json"),
        "heads_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in LOO.glob("*.npz")},
        "paired_test_scored": False, "supervised_SD_family_excluded": True})


def calibrate_loo():
    lock_loo_protocol()
    target = LOO / "thresholds_locked.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["heads_locked_sha256"] != digest(LOO / "heads_locked.json"):
            raise ValueError("SD-exclusion calibration head lock changed")
        print("Reused eighteen locked SD-family-exclusion calibration policies.", flush=True)
        return
    roles = pd.read_csv(LOO / "source_roles.csv")
    frame, frames, paths = base_frame(extra=True)
    cal_ids = set(roles.loc[roles.role.eq("calibration"), "source_id"])
    chosen = frame.source_id.isin(cal_ids).to_numpy()
    cal = frame.loc[chosen].reset_index(drop=True)
    if cal.collection.isin(["stable-diffusion", "midjourney"]).any():
        raise ValueError("Excluded family entered calibration")
    answer = {}
    core_real = cal.label.eq(0).to_numpy() & np.array([(a, int(s)) not in HELD for a, s in zip(cal.attack, cal.severity)])
    for backbone in ["resnet", "clip"]:
        x = complete_features(backbone, frames, paths)
        probe, fusion = LOO / f"{backbone}_probe.npz", LOO / f"{backbone}_fusion.npz"
        score_map = systems(x[chosen], cal.active_score.to_numpy(), probe, fusion)
        answer[backbone] = {"probe_sha256": digest(probe), "fusion_sha256": digest(fusion), "systems": {}}
        for system, score in score_map.items():
            cut, rate = fpr_threshold(score[core_real])
            answer[backbone]["systems"][system] = {"per_image_core13": {"threshold_logit": cut,
                "comparison": ">=", "calibration_copy_fpr": rate, "calibration_real_sources": 219,
                "calibration_real_rows": int(core_real.sum()), "infinite_threshold": False},
                **{f"conformal_{family}": conformal_cut(cal, score, family) for family in ["core13", "all16"]}}
        del x
    write_json(target, {"locked_at_utc": utc(), "protocol_sha256": digest(LOO / "protocol.json"),
        "heads_locked_sha256": digest(LOO / "heads_locked.json"), "models": answer,
        "test_threshold_tuning": False, "paired_test_scored": False, "policy_count": 18,
        "calibration_scope": "Source-disjoint real-negative calibration, COCO144 plus Open Images75. Same core13/all16 bundle rules as the parent Phase5 protocol."})
    print("Frozen eighteen SD-exclusion calibration policies before paired test scores.", flush=True)


def evaluate(sd_excluded=False):
    lock_protocol()
    work = LOO if sd_excluded else DEST
    seeds = [SEEDS[0]] if sd_excluded else SEEDS
    if sd_excluded:
        lock_loo_protocol()
    locked = json.loads((work / "thresholds_locked.json").read_text())
    write_json(work / "analysis_started.json", {"test_scoring_started_at_utc": utc(),
        "thresholds_locked_sha256": digest(work / "thresholds_locked.json"),
        "heads_locked_sha256": digest(work / "heads_locked.json"),
        "protocol_sha256": digest(work / "protocol.json"),
        "new400_is_untouched": False, "test_threshold_tuning": False,
        "purpose": "Prespecified retrospective scoring of cached original tests and new400 follow-up after all heads and policies are locked."})
    roles = pd.read_csv(DEST / "source_roles.csv")
    roles = roles.loc[roles.seed.isin(seeds)]
    frame, frames, paths = base_frame(extra=True)
    fresh = pd.read_csv(FRESH / "manifest.csv", keep_default_na=False)
    fresh["group"] = np.where(fresh.label.eq(0), "R-C", np.where(fresh.mark.eq("marked"), "S-A", "S-U-A"))
    original_union = frame.loc[frame.source_id.isin(roles.loc[roles.role.eq("test"), "source_id"]), ["source_id", "collection", "label"]].drop_duplicates("source_id")
    src = pd.concat([original_union, fresh[["source_id", "collection", "label"]].drop_duplicates("source_id")], ignore_index=True)
    if src.source_id.duplicated().any():
        raise ValueError("Fresh test overlaps original diagnostic source pool")
    src = src.reset_index(drop=True)
    mapping = pd.Series(src.index, index=src.source_id)
    rng = np.random.default_rng(20261020)
    counts = np.zeros((BOOTSTRAP, len(src)), dtype=np.float32)
    # Fresh sources are resampled within their dataset and collection, preserving
    # the distinct acquisition protocol as well as overlapping original tests.
    src["dataset"] = np.where(src.source_id.isin(set(fresh.source_id)), "fresh400_followup", "original")
    for _, part in src.groupby(["dataset", "collection"], sort=True):
        ids = part.index.to_numpy()
        for rep in range(BOOTSTRAP):
            sampled = rng.choice(ids, len(ids), replace=True)
            counts[rep, ids] = np.bincount(sampled, minlength=len(src))[ids]
    metrics, draws = [], []
    started = utc()
    for backbone in ["resnet", "clip"]:
        x = complete_features(backbone, frames, paths)
        with np.load(FRESH / f"{backbone}_embeddings.npz", allow_pickle=True) as z:
            if str(z["manifest_sha256"]) != digest(FRESH / "manifest.csv") or not np.array_equal(z["image_id"].astype(str), fresh.image_id.to_numpy()):
                raise ValueError("Fresh cache identity mismatch")
            xf = z["embedding"].astype(np.float32)
        for seed in seeds:
            test_ids = set(roles.loc[roles.seed.eq(seed) & roles.role.eq("test"), "source_id"])
            chosen = frame.source_id.isin(test_ids).to_numpy()
            test = frame.loc[chosen].reset_index(drop=True)
            if not test.groupby("source_id").size().eq(test.groupby("source_id").label.first().map({0:16, 1:30})).all():
                raise ValueError("Incomplete original test bundle")
            probe = LOO / f"{backbone}_probe.npz" if sd_excluded else CONTROLS / "split_seeds" / str(seed) / f"{backbone}_full.npz"
            fusion = LOO / f"{backbone}_fusion.npz" if sd_excluded else DEST / str(seed) / f"{backbone}_fusion.npz"
            entry = locked["models"][backbone] if sd_excluded else locked["models"][f"{seed}_{backbone}"]
            if digest(fusion) != entry["fusion_sha256"] or digest(probe) != entry["probe_sha256"]:
                raise ValueError("Head changed after calibration")
            batches = [("original", test, systems(x[chosen], test.active_score.to_numpy(), probe, fusion)),
                       ("fresh400_followup", fresh, systems(xf, fresh.active_score.to_numpy(), probe, fusion))]
            for dataset, batch, score_map in batches:
                held = np.array([(a, int(s)) in HELD for a, s in zip(batch.attack, batch.severity)])
                scopes = [("core13", ~held), ("all16", np.ones(len(batch), dtype=bool))] if dataset == "original" else [("four_views_JPEG95", np.ones(len(batch), dtype=bool))]
                index = batch.source_id.map(mapping).to_numpy()
                for system, score in score_map.items():
                    for policy_name, policy in entry["systems"][system].items():
                        pred = decide(score, policy)
                        for scope_name, scope in scopes:
                            for group, mask in [
                                ("overall", np.ones(len(batch), dtype=bool)),
                                ("coco", batch.collection.eq("coco").to_numpy()),
                                ("openimages", batch.collection.eq("openimages").to_numpy()),
                                ("seen_generators", batch.label.eq(1).to_numpy() & (batch.collection.eq("dalle-3").to_numpy() if sd_excluded else ~batch.collection.eq("midjourney").to_numpy())),
                                ("heldout_stable_diffusion", batch.collection.eq("stable-diffusion").to_numpy() if sd_excluded else np.zeros(len(batch), dtype=bool)),
                                ("heldout_midjourney", batch.collection.eq("midjourney").to_numpy()),
                                ("marked_AI", batch.label.eq(1).to_numpy() & batch.group.isin(["S-W", "S-A"]).to_numpy()),
                                ("unmarked_AI", batch.label.eq(1).to_numpy() & batch.group.isin(["S-U", "S-U-A"]).to_numpy())]:
                                selected = scope & mask
                                for quantity, sign in [("copy_FPR", 0), ("copy_TPR", 1), ("source_any_FPR", 0)]:
                                    part = selected & batch.label.eq(sign).to_numpy()
                                    if not part.any():
                                        continue
                                    numerator = np.bincount(index[part], weights=pred[part], minlength=len(src))
                                    denominator = np.bincount(index[part], minlength=len(src))
                                    if quantity == "source_any_FPR":
                                        numerator = (numerator > 0).astype(float)
                                        denominator = (denominator > 0).astype(float)
                                    distribution = (counts @ numerator) / (counts @ denominator)
                                    row = {"seed": seed, "backbone": backbone, "system": system, "policy": policy_name,
                                        "dataset": dataset, "processing_scope": scope_name, "group": group, "quantity": quantity,
                                        "rate": float(numerator.sum() / denominator.sum()),
                                        "ci95_low": float(np.percentile(distribution, 2.5)), "ci95_high": float(np.percentile(distribution, 97.5)),
                                        "source_count": int((denominator > 0).sum()), "copy_count": int(part.sum()),
                                        "threshold_logit": policy["threshold_logit"], "comparison": policy["comparison"], "threshold_retuned": False}
                                    metrics.append(row)
                                    draws.append(distribution)
            print(f"Measured {seed} {backbone} with three locked policies; no threshold changes.", flush=True)
        del x, xf
    result = pd.DataFrame(metrics)
    result.to_csv(work / "per_seed_metrics.csv", index=False)
    keys = ["backbone", "system", "policy", "dataset", "processing_scope", "group", "quantity"]
    summaries = []
    summary_draws = {}
    draws = np.asarray(draws)
    for key, part in result.groupby(keys, sort=True):
        distribution = draws[part.index].mean(axis=0)
        summaries.append(dict(zip(keys, key), mean=float(part.rate.mean()), std=float(part.rate.std(ddof=1)) if len(part) > 1 else None,
            mean_ci95_low=float(np.percentile(distribution, 2.5)), mean_ci95_high=float(np.percentile(distribution, 97.5)),
            seeds=len(part), source_count_each_seed=part.source_count.tolist()))
        summary_draws[key] = distribution
    summary = pd.DataFrame(summaries)
    summary.to_csv(work / "mean_std_ci.csv", index=False)
    differences = []
    other_keys = [key for key in keys if key != "policy"]
    for key, part in summary.groupby(other_keys, sort=True):
        point = part.set_index("policy")
        for policy_name in ["conformal_core13", "conformal_all16"]:
            if policy_name not in point.index or "per_image_core13" not in point.index:
                continue
            factors = dict(zip(other_keys, key))
            case = tuple(factors.get(name, policy_name) for name in keys)
            baseline = tuple(factors.get(name, "per_image_core13") for name in keys)
            paired = summary_draws[case] - summary_draws[baseline]
            differences.append({**factors, "policy_minus_baseline": policy_name,
                "difference": float(point.loc[policy_name, "mean"] - point.loc["per_image_core13", "mean"]),
                "ci95_low": float(np.percentile(paired, 2.5)), "ci95_high": float(np.percentile(paired, 97.5)),
                "bootstrap": "Paired source weights, averaged across the same five overlapping seed tests."})
    pd.DataFrame(differences).to_csv(work / "paired_policy_differences.csv", index=False)
    compact = summary.loc[summary.group.eq("overall")].to_dict("records")
    proof = {"analysis_started_at_utc": started, "completed_at_utc": utc(), "seeds": seeds,
        "SD_family_excluded_from_supervised_fitting": sd_excluded,
        "thresholds_locked_sha256": digest(work / "thresholds_locked.json"),
        "heads_locked_sha256": digest(work / "heads_locked.json"), "protocol_sha256": digest(work / "protocol.json"),
        "metric_rows": len(result), "summary_rows": len(summary), "overall": compact,
        "new400_is_untouched": False, "test_threshold_tuning": False,
        "interpretation": "Independent fusion-fit/calibration halves remove within-seed validation reuse. Conformal source-max calibration is conservative under its stated marginal exchangeability and processing-family assumptions; measured test errors can exceed 5% after encoding/collection/processing shifts. Five tests overlap, and bootstrap weights preserve repeated source identity.",
        "bootstrap_replicates": BOOTSTRAP, "old_heads_unchanged": all(digest(ROOT / name) == value for name, value in lock_protocol()["input_sha256"].items() if name.endswith(".npz") and "full" in name)}
    write_json(work / "summary.json", proof)
    artifact_names = ["protocol.json", "source_roles.csv", "heads_locked.json", "thresholds_locked.json", "analysis_started.json", "per_seed_metrics.csv", "mean_std_ci.csv", "paired_policy_differences.csv", "summary.json"]
    if not sd_excluded:
        artifact_names += ["missing_features.json", "calibration_extra_manifest.csv"]
    write_json(work / "complete.json", {"completed_at_utc": utc(), "fit_count": 4 if sd_excluded else 10, "calibration_policy_count": 18 if sd_excluded else 90,
        "CPU_only": True, "test_threshold_tuning": False, "earlier_heads_unchanged": proof["old_heads_unchanged"],
        "completed_script_sha256": digest(__file__),
        "output_sha256": {name: digest(work / name) for name in artifact_names},
        "new_heads_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in (LOO.glob("*.npz") if sd_excluded else DEST.glob("202*/*_fusion.npz"))}})
    print(summary.loc[summary.group.eq("overall") & summary.system.eq("probe"), ["backbone", "policy", "dataset", "processing_scope", "quantity", "mean", "std", "mean_ci95_low", "mean_ci95_high"]].to_string(index=False), flush=True)


def source_event_intervals():
    """Additional descriptive intervals; never calibrate or alter a decision."""
    for work in [DEST, LOO]:
        if not (work / "per_seed_metrics.csv").exists():
            continue
        source = work / "per_seed_metrics.csv"
        frame = pd.read_csv(source)
        write_json(work / "source_event_interval_protocol.json", {
            "saved_at_utc": utc(), "input_metrics_sha256": digest(source),
            "retrospective_secondary_analysis": True, "thresholds_changed": False,
            "purpose": "Ordinary fixed-threshold source bootstrap degenerates at zero observed events. Add collection-specific two-sided 95% Clopper-Pearson intervals for source-any error counts, requiring independent identically distributed future source bundles within the given collection. These are descriptive conditional-risk intervals; no threshold or model is selected using them.",
            "across_seed_rule": "Report each seed separately. Overlapping sources and model-dependent events are not pooled as independent successes. The original source bootstrap remains the across-seed mean interval."})
        rows = []
        for value in frame.loc[frame.quantity.eq("source_any_FPR") & frame.group.isin(COLLECTIONS)].to_dict("records"):
            n = int(value["source_count"])
            count = int(round(value["rate"] * n))
            if not math.isclose(value["rate"] * n, count, abs_tol=1e-8):
                raise ValueError("Source event rate cannot recover integer count")
            low = float(beta.ppf(.025, count, n - count + 1)) if count else 0.
            high = float(beta.ppf(.975, count + 1, n - count)) if count < n else 1.
            rows.append({key: value[key] for key in ["seed", "backbone", "system", "policy", "dataset", "processing_scope", "group", "rate", "source_count", "ci95_low", "ci95_high"]} |
                {"source_any_positive_count": count, "exact_binomial_ci95_low": low, "exact_binomial_ci95_high": high,
                 "iid_within_collection_assumption_required": True})
        target = work / "source_event_intervals.csv"
        pd.DataFrame(rows).to_csv(target, index=False)
        complete_path = work / "complete.json"
        complete = json.loads(complete_path.read_text())
        complete["output_sha256"].update({name: digest(work / name) for name in ["source_event_interval_protocol.json", "source_event_intervals.csv"]})
        complete["completed_script_sha256"] = digest(__file__)
        write_json(complete_path, complete)
        print(f"Saved {len(rows)} collection-specific source-event intervals in {work.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "fit", "calibrate", "evaluate", "loo-fit", "loo-calibrate", "loo-evaluate", "source-event-intervals", "all"], default="all")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.stage in ["prepare", "all"]:
        prepare_missing()
    if args.stage in ["fit", "all"]:
        fit_fusions()
    if args.stage in ["calibrate", "all"]:
        calibrate()
    if args.stage in ["evaluate", "all"]:
        evaluate()
    if args.stage == "loo-fit":
        fit_loo()
    if args.stage == "loo-calibrate":
        calibrate_loo()
    if args.stage == "loo-evaluate":
        evaluate(sd_excluded=True)
    if args.stage in ["source-event-intervals", "all"]:
        source_event_intervals()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
