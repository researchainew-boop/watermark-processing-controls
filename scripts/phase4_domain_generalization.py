"""Locked, cached-feature controls for real-collection transfer and calibration.

This retrospective diagnostic writes new heads/results only. The five earlier
split assignments, all original heads, and all feature caches are read-only.
No image inference, GPU use, or test-selected threshold is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "outputs/phase3/controls"
ORIGINAL = ROOT / "outputs/conference_clean"
DEST = ROOT / "outputs/phase4/domain_generalization"
SEED = 20260920
BOOTSTRAP = 1000
HELD = {("lowres", 2), ("regenerate", 2), ("spectral", 1)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_manifest(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"payload": str})


def assemble_frame(include_unmarked=True):
    paths = [ORIGINAL / "manifest_derived.csv", CONTROLS / "heldout_clip_manifest.csv",
             CONTROLS / "openimages_derived_manifest.csv"]
    if include_unmarked:
        paths += [CONTROLS / "unmarked_test_manifest.csv",
                  CONTROLS / "unmarked_union_extra_manifest.csv"]
    frames = [read_manifest(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["collection"] = np.where(frame.generator.eq("real"), "coco", frame.generator)
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate feature identifiers")
    assignment = read_manifest(CONTROLS / "split_seeds/assignments.csv")
    assignment = assignment.loc[assignment.seed.eq(SEED)].set_index("source_id")
    frame["locked_split"] = frame.source_id.map(assignment.split)
    if frame.locked_split.isna().any():
        raise ValueError("Missing locked source assignment")
    eligible = np.array([(a, int(s)) not in HELD for a, s in zip(frame.attack, frame.severity)])
    eligible &= ~frame.group.eq("S-U-A").to_numpy()
    return frame, frames, paths, eligible


def read_matrix(paths, identifiers):
    requested = identifiers.astype(str).to_numpy()
    lookup = {}
    blocks = []
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as values:
            vectors = values["embedding"].astype(np.float32)
            ids = values["image_id"].astype(str)
        blocks.append(vectors)
        for i, identifier in enumerate(ids):
            lookup.setdefault(identifier, offset + i)
        offset += len(vectors)
    indices = np.array([lookup[identifier] for identifier in requested])
    vectors = np.concatenate(blocks)
    return vectors[indices]


def locked_matrix(backbone, kind, frame, manifest_path):
    path = CONTROLS / f"{backbone}_{kind}_embeddings.npz"
    with np.load(path, allow_pickle=True) as values:
        if not np.array_equal(values["image_id"].astype(str), frame.image_id.to_numpy()):
            raise ValueError(f"Feature row mismatch: {path}")
        if str(values["manifest_sha256"]) != digest(manifest_path):
            raise ValueError(f"Feature manifest checksum mismatch: {path}")
        return values["embedding"].astype(np.float32)


def features(backbone, frames, paths):
    original_paths = ([ROOT / "outputs/resnet_embeddings.npz"] if backbone == "resnet" else
                      [ROOT / "outputs/clip_train_embeddings.npz", ROOT / "outputs/clip_eval_embeddings.npz"])
    blocks = [read_matrix(original_paths, frames[0].image_id)]
    blocks += [read_matrix(original_paths, frames[1].image_id) if backbone == "resnet" else
               locked_matrix(backbone, "heldout", frames[1], paths[1])]
    for kind, frame, path in zip(["openimages", "unmarked", "union"], frames[2:], paths[2:]):
        blocks.append(locked_matrix(backbone, kind, frame, path))
    return np.concatenate(blocks)


def logits(x, head):
    with np.load(head) as values:
        return (np.asarray(x, dtype=np.float64) @ values["coefficients"].reshape(-1)
                + float(values["intercept"].reshape(-1)[0]))


def systems(x, active, probe_head, fusion_head):
    p = logits(x, probe_head)
    f = logits(np.column_stack([expit(p), active]), fusion_head)
    return {"probe": p, "fusion": f, "gated": np.where(active >= 13 / 16, f, p)}


def fpr_threshold(scores, target=.05):
    """Lowest observed logit threshold with <=target empirical FPR; ties stay whole."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    unique, counts = np.unique(scores, return_counts=True)
    accept = np.cumsum(counts[::-1])[::-1]
    valid = np.flatnonzero(accept / len(scores) <= target)
    threshold = float(unique[valid[0]]) if len(valid) else float(np.nextafter(scores.max(), np.inf))
    actual = float((scores >= threshold).mean())
    if actual > target:
        raise ValueError("Calibration threshold violates target FPR")
    return threshold, actual


def frozen_calibration():
    target = DEST / "frozen_seed_thresholds.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if any(digest(ROOT / name) != value for name, value in saved["head_sha256"].items()):
            raise ValueError("Locked frozen calibration head changed")
        if digest(DEST / "frozen_calibration_scope.csv") != saved["scope_sha256"]:
            raise ValueError("Locked frozen calibration scope changed")
        print(json.dumps({"reused_locked_thresholds": str(target), "locked_at_utc": saved["locked_at_utc"]}), flush=True)
        return
    frame, frames, paths, eligible = assemble_frame(include_unmarked=False)
    val = frame.locked_split.eq("val").to_numpy() & eligible
    scope = frame.loc[val, ["image_id", "source_id", "label", "collection", "group", "attack", "severity"]]
    scope.to_csv(DEST / "frozen_calibration_scope.csv", index=False)
    old_lock = json.loads((CONTROLS / "protocol.json").read_text())["original_heads_sha256"]
    seed_heads = CONTROLS / "split_seeds" / str(SEED)
    heads = [seed_heads / f"{backbone}{suffix}_full.npz" for backbone in ["resnet", "clip"]
             for suffix in ["", "_fusion"]]
    protocol = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                "scope_sha256": digest(DEST / "frozen_calibration_scope.csv"),
                "head_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in heads},
                "threshold_rule": "Minimum unsaturated logit threshold with validation real-copy FPR <=0.05; classify logit >= threshold, retaining whole ties.",
                "threshold_selected_before_new_replication_source_selection": True,
                "scope": "All thirteen eligible conditions of existing locked seed-20260920 validation sources, both real collections; no test data used.",
                "dependence": "Validation is source-disjoint from probe training. Existing fusion heads were fitted on these same validation rows, so fusion/gate thresholds are retrospective calibration, not an independent fusion calibration set.",
                "validation_sources": int(scope.source_id.nunique()), "validation_real_sources": int(scope.loc[scope.label.eq(0), "source_id"].nunique()),
                "validation_rows": len(scope), "original_heads_sha256": old_lock}
    write_json(DEST / "frozen_calibration_protocol.json", protocol)
    result = {**protocol, "systems": {}}
    for backbone in ["resnet", "clip"]:
        x = features(backbone, frames, paths)
        scores = systems(x[val], frame.loc[val, "active_score"].to_numpy(),
                         seed_heads / f"{backbone}_full.npz", seed_heads / f"{backbone}_fusion_full.npz")
        labels = frame.loc[val, "label"].to_numpy()
        for name, score in scores.items():
            threshold, fpr = fpr_threshold(score[labels == 0])
            result["systems"][f"{backbone}_{name}"] = {"threshold_logit": threshold,
                "threshold_probability": float(expit(threshold)), "validation_fpr": fpr,
                "validation_tpr": float((score[labels == 1] >= threshold).mean()),
                "validation_real_copies": int((labels == 0).sum()), "validation_AI_copies": int((labels == 1).sum())}
        del x
    if any(digest(ORIGINAL / name) != value for name, value in old_lock.items()):
        raise ValueError("Original frozen head changed")
    write_json(DEST / "frozen_seed_thresholds.json", result)
    print(json.dumps({"saved": str(DEST / "frozen_seed_thresholds.json"), "systems": result["systems"]}), flush=True)


def ranked(identifiers, role):
    return sorted(identifiers, key=lambda identifier: hashlib.sha256(f"{SEED}:{role}:{identifier}".encode()).hexdigest())


def lock_domain_protocol(frame, eligible):
    sources = frame[["source_id", "collection", "label", "locked_split"]].drop_duplicates("source_id")
    cohorts = {}
    for split, count in [("train", 700), ("val", 150), ("test", 150)]:
        for collection in ["coco", "openimages"]:
            ids = sources.loc[sources.locked_split.eq(split) & sources.collection.eq(collection), "source_id"].tolist()
            if len(ids) < count:
                raise ValueError("Insufficient collection-matched source count")
            cohorts[f"{split}_{collection}"] = ranked(ids, f"{split}:{collection}")[:count]
    AI = {split: set(sources.loc[sources.label.eq(1) & sources.locked_split.eq(split), "source_id"])
          for split in ["train", "val", "test"]}
    fit_AI_val = set(ranked(AI["val"], "AI-validation-half")[::2])
    cal_AI_val = AI["val"] - fit_AI_val
    designs = {}
    for regime in ["coco_only", "openimages_only", "joint"]:
        if regime == "joint":
            train_real = set(cohorts["train_coco"][:350] + cohorts["train_openimages"][:350])
            fit_real = set(cohorts["val_coco"][:37] + cohorts["val_openimages"][:38])
            cal_real = set(cohorts["val_coco"][37:75] + cohorts["val_openimages"][38:75])
        else:
            collection = regime.removesuffix("_only")
            train_real = set(cohorts[f"train_{collection}"])
            fit_real = set(cohorts[f"val_{collection}"][:75])
            cal_real = set(cohorts[f"val_{collection}"][75:150])
        designs[regime] = {"train": AI["train"] | train_real,
                           "fusion_fit": fit_AI_val | fit_real,
                           "calibration": cal_AI_val | cal_real,
                           "test": AI["test"] | set(cohorts["test_coco"] + cohorts["test_openimages"])}
        for left in ["train", "fusion_fit", "calibration", "test"]:
            for right in ["train", "fusion_fit", "calibration", "test"]:
                if left != right and designs[regime][left] & designs[regime][right]:
                    raise ValueError("Source leakage between locked diagnostic roles")
    records = []
    for regime, roles in designs.items():
        for role, ids in roles.items():
            for row in sources.loc[sources.source_id.isin(ids)].to_dict("records"):
                records.append({"regime": regime, "role": role, **row})
    assignments = pd.DataFrame(records).sort_values(["regime", "role", "collection", "source_id"])
    target = DEST / "source_assignments.csv"
    if target.exists():
        if target.read_text(encoding="utf-8") != assignments.to_csv(index=False):
            raise ValueError("Locked assignment changed")
    else:
        assignments.to_csv(target, index=False)
    protocol = {"locked_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                "source_assignments_sha256": digest(target),
                "prior_split_assignments_sha256": digest(CONTROLS / "split_seeds/assignments.csv"),
                "design": "Same synthetic sources, source split, and thirteen training conditions for all regimes. Real training sources:700 COCO,700 OpenImages,or350+350. Fusion fitting and threshold calibration use separate source halves of the existing validation split:75 real sources each. Every regime is compared against identical held-out150 COCO+150 OpenImages and the same AI test sources.",
                "threshold": "Lowest calibration-real logit threshold retaining whole ties and empirical FPR<=5%; fixed before reading test scores. Fusion and probe fitting never see calibration sources.",
                "probe": {"model": "L2 logistic regression", "C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000, "random_state": 0},
                "fusion": "L2 logistic regression on probe probability and payload agreement, fitted only to fusion-fit validation sources; gated fallback uses fixed13/16 bits.",
                "copy_design": "Train,fusion-fit,calibration:13conditions/source. Test:16real or30AI copies/source, including fourteen matched unmarked AI edits and three attacks held out of fitting/calibration.",
                "source_inference": "All embeddings are existing cached features. No new images or backbone inference.",
                "bootstrap": {"replicates": BOOTSTRAP, "unit": "source including all copies", "strata": "real collection and AI generator", "seed": 20261014, "interval": "percentile95%"},
                "caveat": "Retrospective diagnostic on an already examined source pool; not a new untouched test. Source clustering does not establish semantic independence or eliminate pretraining overlap. Midjourney never enters fitting or calibration.",
                "script_sha256": digest(Path(__file__)),
                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]}, "python": platform.python_version()}
    path = DEST / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key in ["source_assignments_sha256", "prior_split_assignments_sha256", "probe", "threshold", "design"]:
            if protocol[key] != saved[key]:
                raise ValueError(f"Protocol changed: {key}")
    else:
        write_json(path, protocol)
    return designs, assignments


def fit(x, labels, head):
    if head.exists():
        return
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, labels)
    if any(issubclass(value.category, ConvergenceWarning) for value in found):
        raise RuntimeError("Diagnostic fit failed to converge")
    np.savez(head, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)


def weighted_auc(labels, scores, source_index, counts):
    order = np.argsort(scores, kind="stable")
    sorted_score = scores[order]
    sorted_label = labels[order]
    sorted_source = source_index[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    result = []
    for begin in range(0, len(counts), 32):
        weights = counts[begin:begin + 32, sorted_source]
        positives = np.add.reduceat(weights * sorted_label, starts, axis=1)
        negatives = np.add.reduceat(weights * (1 - sorted_label), starts, axis=1)
        below = np.cumsum(negatives, axis=1) - negatives
        auc = (positives * (below + .5 * negatives)).sum(axis=1) / (positives.sum(axis=1) * negatives.sum(axis=1))
        result.extend(auc.tolist())
    return np.asarray(result)


def bootstrap_counts(source_frame):
    rng = np.random.default_rng(20261014)
    counts = np.zeros((BOOTSTRAP, len(source_frame)), dtype=np.float32)
    for _, group in source_frame.groupby("collection", sort=True):
        ids = group.index.to_numpy()
        for rep in range(BOOTSTRAP):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            counts[rep, ids] = np.bincount(sampled, minlength=len(source_frame))[ids]
    return counts


def cluster_rate(labels, predictions, source_index, counts, positive):
    chosen = labels == positive
    numerator = np.bincount(source_index[chosen], weights=predictions[chosen], minlength=counts.shape[1])
    denominator = np.bincount(source_index[chosen], minlength=counts.shape[1])
    estimates = counts @ numerator / (counts @ denominator)
    return float(predictions[chosen].mean()), np.percentile(estimates, [2.5, 97.5]).tolist()


def describe_predictions(test, scores, threshold, counts):
    source_frame = test[["source_id", "collection", "label"]].drop_duplicates("source_id").reset_index(drop=True)
    lookup = pd.Series(source_frame.index, index=source_frame.source_id)
    index = test.source_id.map(lookup).to_numpy()
    y = test.label.to_numpy()
    pred = scores >= threshold
    auc_draws = weighted_auc(y, scores, index, counts)
    fpr, fpr_ci = cluster_rate(y, pred, index, counts, 0)
    tpr, tpr_ci = cluster_rate(y, pred, index, counts, 1)
    answer = {"auc": float(roc_auc_score(y, scores)), "auc_ci95": np.percentile(auc_draws, [2.5, 97.5]).tolist(),
              "test_fpr": fpr, "test_fpr_ci95": fpr_ci, "test_tpr": tpr, "test_tpr_ci95": tpr_ci,
              "test_rows": len(test), "test_sources": len(source_frame), "real_sources": int(source_frame.label.eq(0).sum()),
              "AI_sources": int(source_frame.label.eq(1).sum()), "threshold_logit": threshold,
              "bootstrap_replicates": BOOTSTRAP, "test_threshold_retuned": False, "subgroups": {}}
    for collection in ["coco", "openimages", "stable-diffusion", "dalle-3", "midjourney"]:
        selected = test.collection.eq(collection).to_numpy()
        if not selected.any():
            continue
        numerator = np.bincount(index[selected], weights=pred[selected], minlength=len(source_frame))
        denominator = np.bincount(index[selected], minlength=len(source_frame))
        draws = counts @ numerator / (counts @ denominator)
        answer["subgroups"][collection] = {"rate": float(pred[selected].mean()),
            "rate_ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "sources": int(test.loc[selected, "source_id"].nunique()),
            "rows": int(selected.sum()), "rate_kind": "FPR" if collection in ["coco", "openimages"] else "TPR"}
    for name, selected in [("seen_generators", test.label.eq(1).to_numpy() & ~test.collection.eq("midjourney").to_numpy()),
                           ("heldout_midjourney", test.collection.eq("midjourney").to_numpy()),
                           ("unmarked_AI_edits", test.group.eq("S-U-A").to_numpy()),
                           ("marked_AI_edits", test.group.eq("S-A").to_numpy())]:
        if not selected.any():
            continue
        numerator = np.bincount(index[selected], weights=pred[selected], minlength=len(source_frame))
        denominator = np.bincount(index[selected], minlength=len(source_frame))
        draws = counts @ numerator / (counts @ denominator)
        answer["subgroups"][name] = {"rate": float(pred[selected].mean()), "rate_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
            "sources": int(test.loc[selected, "source_id"].nunique()), "rows": int(selected.sum()), "rate_kind": "TPR"}
    return answer, auc_draws


def domain_controls(backbones):
    frame, frames, paths, eligible = assemble_frame()
    designs, assignments = lock_domain_protocol(frame, eligible)
    old_lock = json.loads((CONTROLS / "protocol.json").read_text())["original_heads_sha256"]
    common_test_ids = designs["joint"]["test"]
    test_mask = frame.source_id.isin(common_test_ids).to_numpy()
    test = frame.loc[test_mask].copy().reset_index(drop=True)
    source_frame = test[["source_id", "collection", "label"]].drop_duplicates("source_id").reset_index(drop=True)
    expected = test.groupby("source_id").label.first().map({0: 16, 1: 30})
    if not test.groupby("source_id").size().equals(expected):
        raise ValueError("Unequal source-copy test design")
    counts = bootstrap_counts(source_frame)
    models, metrics, auc_boot = {}, [], {}
    for backbone in backbones:
        x = features(backbone, frames, paths)
        for regime, roles in designs.items():
            work = DEST / backbone / regime
            work.mkdir(parents=True, exist_ok=True)
            mask = {role: frame.source_id.isin(ids).to_numpy() & (eligible if role != "test" else True)
                    for role, ids in roles.items()}
            probe, fusion = work / "probe.npz", work / "fusion.npz"
            print(f"Fit {backbone} {regime}: {mask['train'].sum()} training copies", flush=True)
            fit(x[mask["train"]], frame.loc[mask["train"], "label"].to_numpy(), probe)
            probe_val = expit(logits(x[mask["fusion_fit"]], probe))
            fit(np.column_stack([probe_val, frame.loc[mask["fusion_fit"], "active_score"]]),
                frame.loc[mask["fusion_fit"], "label"].to_numpy(), fusion)
            cal_mask = mask["calibration"]
            cal_y = frame.loc[cal_mask, "label"].to_numpy()
            cal_scores = systems(x[cal_mask], frame.loc[cal_mask, "active_score"].to_numpy(), probe, fusion)
            thresholds = {name: fpr_threshold(score[cal_y == 0]) for name, score in cal_scores.items()}
            lock = {"selected_at_utc": datetime.now(timezone.utc).isoformat(), "test_inference_performed": False,
                    "calibration_scope": "Separate held-out validation-source half, never used in either fitted head.",
                    "real_sources": 75, "real_rows": int((cal_y == 0).sum()),
                    "AI_sources": int(frame.loc[cal_mask & frame.label.eq(1).to_numpy(), "source_id"].nunique()),
                    "systems": {name: {"threshold_logit": threshold, "calibration_fpr": fpr,
                                "calibration_tpr": float((cal_scores[name][cal_y == 1] >= threshold).mean())}
                                for name, (threshold, fpr) in thresholds.items()},
                    "heads_sha256": {p.name: digest(p) for p in [probe, fusion]}}
            write_json(work / "thresholds.json", lock)
            test_scores = systems(x[test_mask], test.active_score.to_numpy(), probe, fusion)
            prediction = test[["image_id", "source_id", "label", "collection", "group", "attack", "severity"]].copy()
            report = {"backbone": backbone, "regime": regime, "thresholds": lock, "systems": {}}
            for name, score in test_scores.items():
                threshold, _ = thresholds[name]
                measured, draws = describe_predictions(test, score, threshold, counts)
                measured["calibration_fpr"] = thresholds[name][1]
                report["systems"][name] = measured
                auc_boot[(backbone, regime, name)] = draws
                prediction[name + "_logit"] = score
                prediction[name + "_accepted"] = score >= threshold
                metrics.append({"backbone": backbone, "regime": regime, "system": name,
                    **{key: value for key, value in measured.items() if key not in ["subgroups", "auc_ci95", "test_fpr_ci95", "test_tpr_ci95"]},
                    "auc_ci95_low": measured["auc_ci95"][0], "auc_ci95_high": measured["auc_ci95"][1],
                    "FPR_coco": measured["subgroups"]["coco"]["rate"], "FPR_openimages": measured["subgroups"]["openimages"]["rate"],
                    "TPR_seen_generators": measured["subgroups"]["seen_generators"]["rate"], "TPR_midjourney": measured["subgroups"]["heldout_midjourney"]["rate"]})
            write_json(work / "metrics.json", report)
            prediction.to_csv(work / "test_predictions.csv", index=False)
            models[f"{backbone}_{regime}"] = report
            pd.DataFrame(metrics).to_csv(DEST / "metrics.partial.csv", index=False)
        del x
    summary = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
               "protocol_sha256": digest(DEST / "protocol.json"), "models": models, "paired_auc_differences": {},
               "test_sources": len(source_frame), "test_rows": len(test), "test_real_sources_each_collection": 150,
               "test_AI_sources": int(source_frame.label.eq(1).sum()), "bootstrap_replicates": BOOTSTRAP,
               "caveat": "Retrospective diagnostics using cached features on previously examined source pools. Independent source halves prevent fitting/calibration leakage within this diagnostic; independent collection transfer remains empirical, not guaranteed.",
               "original_heads_unchanged": all(digest(ORIGINAL / name) == value for name, value in old_lock.items())}
    if not summary["original_heads_unchanged"]:
        raise ValueError("Original head changed")
    for backbone in backbones:
        for regime in ["coco_only", "openimages_only"]:
            for system in ["probe", "fusion", "gated"]:
                difference = auc_boot[(backbone, "joint", system)] - auc_boot[(backbone, regime, system)]
                point = models[f"{backbone}_joint"]["systems"][system]["auc"] - models[f"{backbone}_{regime}"]["systems"][system]["auc"]
                summary["paired_auc_differences"][f"{backbone}_joint_minus_{regime}_{system}"] = {
                    "difference": point, "ci95": np.percentile(difference, [2.5, 97.5]).tolist(),
                    "method": "Paired source-cluster bootstrap on identical test copies, stratified by collection."}
    pd.DataFrame(metrics).to_csv(DEST / "metrics.csv", index=False)
    write_json(DEST / "summary.json", summary)
    print(pd.DataFrame(metrics)[["backbone", "regime", "system", "auc", "test_fpr", "test_tpr", "FPR_coco", "FPR_openimages", "TPR_seen_generators", "TPR_midjourney"]].to_string(index=False), flush=True)


def attack_scope_rates():
    """Separate collection transfer from test attacks absent from calibration."""
    summary = json.loads((DEST / "summary.json").read_text())
    protocol = {"analysis_locked_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Describe fixed-threshold photograph false positives on the thirteen calibration-matched conditions separately from three held-out attacks. No head or threshold changes; test data already examined in the main retrospective diagnostic.",
                "heldout_conditions": sorted([list(value) for value in HELD]),
                "bootstrap_replicates": BOOTSTRAP, "unit": "source, stratified by collection", "bootstrap_seed": 20261014,
                "input_summary_sha256": digest(DEST / "summary.json")}
    write_json(DEST / "attack_scope_protocol.json", protocol)
    all_rows = []
    for key, model in summary["models"].items():
        backbone, regime = model["backbone"], model["regime"]
        work = DEST / backbone / regime
        frame = pd.read_csv(work / "test_predictions.csv", keep_default_na=False)
        source = frame[["source_id", "collection", "label"]].drop_duplicates("source_id").reset_index(drop=True)
        counts = bootstrap_counts(source)
        lookup = pd.Series(source.index, index=source.source_id)
        source_index = frame.source_id.map(lookup).to_numpy()
        held = np.array([(a, int(s)) in HELD for a, s in zip(frame.attack, frame.severity)])
        rates = {}
        for scope, scope_mask in [("matched_13_conditions", ~held), ("heldout_3_conditions", held)]:
            rates[scope] = {}
            for collection in ["overall", "coco", "openimages"]:
                chosen = scope_mask & frame.label.eq(0).to_numpy()
                if collection != "overall":
                    chosen &= frame.collection.eq(collection).to_numpy()
                if not chosen.any():
                    raise ValueError("Empty attack/collection scope")
                rates[scope][collection] = {}
                for system in ["probe", "fusion", "gated"]:
                    prediction = frame[system + "_accepted"].to_numpy(dtype=bool)
                    numerator = np.bincount(source_index[chosen], weights=prediction[chosen], minlength=len(source))
                    denominator = np.bincount(source_index[chosen], minlength=len(source))
                    draws = counts @ numerator / (counts @ denominator)
                    result = {"fpr": float(prediction[chosen].mean()),
                              "fpr_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
                              "real_sources": int(frame.loc[chosen, "source_id"].nunique()), "real_rows": int(chosen.sum()),
                              "threshold_logit": model["systems"][system]["threshold_logit"],
                              "threshold_changed": False}
                    rates[scope][collection][system] = result
                    all_rows.append({"backbone": backbone, "regime": regime, "system": system,
                                     "attack_scope": scope, "real_collection": collection,
                                     **{name: value for name, value in result.items() if name != "fpr_ci95"},
                                     "fpr_ci95_low": result["fpr_ci95"][0], "fpr_ci95_high": result["fpr_ci95"][1]})
        model["attack_scope_false_positives"] = rates
        report = json.loads((work / "metrics.json").read_text())
        report["attack_scope_false_positives"] = rates
        write_json(work / "metrics.json", report)
    summary["attack_scope_protocol_sha256"] = digest(DEST / "attack_scope_protocol.json")
    write_json(DEST / "summary.json", summary)
    pd.DataFrame(all_rows).to_csv(DEST / "attack_scope_false_positives.csv", index=False)
    write_json(DEST / "complete.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "backbones": sorted({model["backbone"] for model in summary["models"].values()}),
        "regimes": ["coco_only", "openimages_only", "joint"], "diagnostic_systems": len(summary["models"]) * 3,
        "test_sources": summary["test_sources"], "test_rows": summary["test_rows"], "thresholds_selected_on_test": False,
        "original_heads_unchanged": summary["original_heads_unchanged"], "device": "CPU; existing cached features only",
        "head_sha256": {path.relative_to(DEST).as_posix(): digest(path) for path in DEST.glob("*/*/*.npz")},
        "result_sha256": {name: digest(DEST / name) for name in ["protocol.json", "summary.json", "metrics.csv", "source_assignments.csv", "attack_scope_protocol.json", "attack_scope_false_positives.csv", "frozen_seed_thresholds.json"]},
        "script_sha256": digest(Path(__file__)), "inference": "All photograph rates, AUC differences, and generator TPRs retain fixed fitted heads and fixed calibration thresholds; intervals use source-cluster resampling."})
    print(pd.DataFrame(all_rows).loc[lambda f: f.system.eq("probe") & f.attack_scope.eq("matched_13_conditions"),
          ["backbone", "regime", "real_collection", "fpr", "fpr_ci95_low", "fpr_ci95_high"]].to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["frozen-calibration", "domains", "attack-scope", "all"], default="all")
    ap.add_argument("--backbone", choices=["both", "resnet", "clip"], default="both")
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.stage in ["frozen-calibration", "all"]:
        frozen_calibration()
    if args.stage in ["domains", "all"]:
        domain_controls(["resnet", "clip"] if args.backbone == "both" else [args.backbone])
    if args.stage in ["attack-scope", "all"]:
        attack_scope_rates()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
