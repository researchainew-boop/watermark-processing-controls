"""Equal-slot real-edit ablation on frozen features, with validation-only operating points.

This retrospective diagnostic preserves every earlier fit and result. Its
protocol is written before fitting. No image extraction or backbone training
is performed. Repeated clean rows control training slots and class weighting,
but cannot control the information content of distinct augmentations.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_encoding_controls import digest
from phase3_analyze_controls import HELD, SEEDS, locked_features, matrix

OLD = ROOT / "outputs/conference_clean"
CONTROLS = ROOT / "outputs/phase3/controls"
PRIOR = CONTROLS / "split_seeds"
OUT = ROOT / "outputs/phase4/size_controlled"
VARIANTS = ["full", "clean_repeat", "no_matched_reals"]
MEASURES = ["auc", "tpr_at_validation_fpr5", "test_fpr_at_validation_fpr5", "tpr_at_test_roc_fpr5"]
BOOT_SEED = 20261010


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def load_design():
    old = pd.read_csv(OLD / "manifest_derived.csv", keep_default_na=False, dtype={"payload": str})
    heldout = pd.read_csv(CONTROLS / "heldout_clip_manifest.csv", keep_default_na=False, dtype={"payload": str})
    oi = pd.read_csv(CONTROLS / "openimages_derived_manifest.csv", keep_default_na=False, dtype={"payload": str})
    unmarked = pd.read_csv(CONTROLS / "unmarked_test_manifest.csv", keep_default_na=False, dtype={"payload": str})
    union = pd.read_csv(CONTROLS / "unmarked_union_extra_manifest.csv", keep_default_na=False, dtype={"payload": str})
    parts = [old, heldout, oi, unmarked, union]
    frame = pd.concat(parts, ignore_index=True)
    frame["collection"] = np.where(frame.generator.eq("real"), "coco", frame.generator)
    if frame.image_id.duplicated().any():
        raise ValueError("Duplicate image identities in feature design")
    eligible = np.array([(a, int(s)) not in HELD for a, s in zip(frame.attack, frame.severity)]) & ~frame.group.eq("S-U-A").to_numpy()
    if not frame.loc[eligible].groupby("source_id").size().eq(13).all():
        raise ValueError("Thirteen eligible training slots required per source")
    assignments = pd.read_csv(PRIOR / "assignments.csv", keep_default_na=False)
    if digest(PRIOR / "assignments.csv") != json.loads((PRIOR / "protocol.json").read_text())["seed_assignment_sha256"]:
        raise ValueError("Previous source assignments changed")
    maps = {seed: sub.set_index("source_id").split.to_dict() for seed, sub in assignments.groupby("seed")}
    if sorted(maps) != SEEDS:
        raise ValueError("Expected all five original split seeds")
    split_arrays = {seed: frame.source_id.map(maps[seed]).to_numpy() for seed in SEEDS}
    for values in split_arrays.values():
        if pd.isna(values).any():
            raise ValueError("Unassigned source")
        test = frame.loc[values == "test"]
        if len(test) != 49158 or test.source_id.nunique() != 1843:
            raise ValueError("Unexpected test cohort")
        expected = test.groupby("source_id").label.first().map({0: 16, 1: 30})
        if not test.groupby("source_id").size().equals(expected):
            raise ValueError("Changed per-source test design")
    return frame, parts, eligible, split_arrays


def load_features(name, parts):
    old, heldout, oi, unmarked, union = parts
    h = matrix(name, heldout) if name == "resnet" else locked_features(name, "heldout", heldout)
    x = np.concatenate([matrix(name, old), h, locked_features(name, "openimages", oi),
                        locked_features(name, "unmarked", unmarked), locked_features(name, "union", union)])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite frozen features")
    return x


def frozen_inputs():
    paths = [OLD / "manifest_derived.csv", OLD / "manifest_sources.csv", PRIOR / "assignments.csv",
             PRIOR / "protocol.json", PRIOR / "complete.json",
             ROOT / "outputs/resnet_embeddings.npz", ROOT / "outputs/clip_train_embeddings.npz", ROOT / "outputs/clip_eval_embeddings.npz"]
    paths += list(CONTROLS.glob("*_embeddings.npz"))
    paths += [CONTROLS / name for name in ["heldout_clip_manifest.csv", "openimages_derived_manifest.csv",
        "unmarked_test_manifest.csv", "unmarked_union_extra_manifest.csv", "openimages_sources.csv"]]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def preserved_heads():
    paths = list(OLD.glob("*.npz")) + list(PRIOR.glob("*/*.npz"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def prepare_protocol(frame, eligible, split_arrays, samples):
    OUT.mkdir(parents=True, exist_ok=True)
    counts = []
    for seed, split in split_arrays.items():
        training = frame.loc[(split == "train") & eligible]
        reduced = training.loc[~training.group.eq("R-T")]
        counts.append({"seed": seed, "training_sources": int(training.source_id.nunique()),
            "full_rows": len(training), "clean_repeat_rows": len(training), "no_matched_reals_rows": len(reduced),
            "replaced_real_edit_slots": int(training.group.eq("R-T").sum()),
            "full_class_counts": {str(k): int(v) for k, v in training.label.value_counts().items()},
            "reduced_class_counts": {str(k): int(v) for k, v in reduced.label.value_counts().items()},
            "validation_rows": int(((split == "val") & eligible).sum()), "test_rows": int((split == "test").sum())})
    protocol = {"frozen_at_utc": utc(), "status": "retrospective follow-up, fixed before new fits",
        "seeds": SEEDS, "backbones": ["resnet", "clip"], "variants": VARIANTS,
        "primary_contrast": "full minus clean_repeat",
        "full": "All thirteen eligible original slots per training source, including eleven matched real edits.",
        "clean_repeat": "Replace each R-T feature vector by the R-C frozen clean vector from the same real source. Keep row identities, source identities, labels, ordering, slots, all AI rows and R-S rows unchanged.",
        "no_matched_reals": "Drop the eleven R-T slots per real source, reproducing the earlier reduced-row baseline.",
        "unchanged_between_full_and_clean_repeat": ["source assignments", "real and AI source identities", "row ordering", "label sequence", "row count", "class counts", "balanced class weights", "solver", "regularization", "iteration cap", "convergence tolerance", "validation cohort", "test cohort"],
        "counts": counts, "logistic_regression_parameters": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0).get_params(),
        "fits": "Every full, repeated-clean and reduced-row head is independently fitted here, including convergence diagnostics; original heads are never overwritten.",
        "threshold_selection": "Select the smallest validation logit threshold with empirical validation FPR <=0.05 (no randomized ties), maximizing attainable validation TPR. Evaluate unchanged on test. Test-ROC TPR5 is separately descriptive and never used for threshold selection.",
        "bootstrap": {"replicates": samples, "seed": BOOT_SEED,
            "unit": "Unique source in the union of all five test cohorts, retaining every derivative and its membership in each split.",
            "stratification": "Resample sources within collection (COCO, Open Images, Stable Diffusion, DALL-E 3, Midjourney), using the same source multiplicities for every head and every split.",
            "estimator": "Paired differences per split and arithmetic mean of the five split metric differences; percentiles across shared source-cluster resamples.",
            "conditioning": "Fitted heads, validation-selected thresholds and cohort membership remain fixed; no training or threshold uncertainty is included. Individual split counts fluctuate when restricting union resamples. The same Midjourney holdout and overlapping sources are not treated as independent replications."},
        "limitations": ["Duplicating clean features controls training slots, class weight and optimization budget, not augmentation diversity or information content.",
            "This experiment does not distinguish the benefit of matching the specific AI edits from the benefit of other equally diverse real augmentations.",
            "The source pool and prior results had already been examined; this is neither preregistration nor independent validation.",
            "Only frozen-feature logistic probes are compared here; no fusion coefficients or backbone weights are trained."],
        "input_sha256": frozen_inputs(), "preserved_head_sha256": preserved_heads(),
        "script_sha256": digest(Path(__file__)), "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
        "device": "CPU", "thread_limit": 4}
    target = OUT / "protocol.json"
    if target.exists():
        previous = json.loads(target.read_text())
        for key in ["seeds", "variants", "counts", "input_sha256", "preserved_head_sha256", "script_sha256", "logistic_regression_parameters", "packages", "bootstrap"]:
            if previous[key] != protocol[key]:
                if key == "script_sha256" and (OUT / "jpeg_protocol_amendment.json").exists():
                    amendment = json.loads((OUT / "jpeg_protocol_amendment.json").read_text())
                    if (digest(OUT / "initial_script_snapshot.py") == previous[key]
                        and amendment["script_sha256"] == protocol[key]):
                        continue
                raise ValueError(f"Frozen protocol changed: {key}")
        return previous
    write_json(target, protocol)
    return protocol


def fit_head(x, y, path):
    record = path.with_suffix(".fit.json")
    if path.exists():
        if not record.exists() or json.loads(record.read_text())["model_sha256"] != digest(path):
            raise ValueError("Incomplete or changed new fit")
        return json.loads(record.read_text())
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    info = {"completed_at_utc": utc(), "fit_seconds": time.perf_counter() - start,
        "rows": len(y), "features": x.shape[1], "n_iter": model.n_iter_.astype(int).tolist(),
        "iteration_cap": 3000, "tolerance": float(model.tol), "warnings": [str(w.message) for w in found],
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in found),
        "class_counts": {str(c): int((y == c).sum()) for c in [0, 1]},
        "balanced_class_weights": {str(c): float(len(y) / (2 * (y == c).sum())) for c in [0, 1]}}
    if not info["converged"]:
        write_json(record, info)
        raise RuntimeError("Nonconverged new head; no result reported")
    np.savez(path, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)
    info["model_sha256"] = digest(path)
    write_json(record, info)
    return info


def linear_scores(x, path):
    with np.load(path) as model:
        return (x.astype(np.float64) @ model["coefficients"].astype(np.float64).T + model["intercept"]).ravel()


def choose_fpr5_threshold(y, score):
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
    candidates = np.flatnonzero((fpr <= .05) & np.isfinite(thresholds))
    if not len(candidates):
        threshold = float(np.nextafter(score.max(), np.inf))
    else:
        threshold = float(thresholds[candidates[-1]])
    predicted = score >= threshold
    return {"logit_threshold": threshold, "probability_threshold": float(expit(threshold)),
        "validation_fpr": float(predicted[y == 0].mean()), "validation_tpr": float(predicted[y == 1].mean()),
        "validation_real_rows": int((y == 0).sum()), "validation_AI_rows": int((y == 1).sum())}


def metric_values(y, score, threshold):
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    pred = score >= threshold
    return {"auc": float(roc_auc_score(y, score)), "tpr_at_validation_fpr5": float(pred[y == 1].mean()),
        "test_fpr_at_validation_fpr5": float(pred[y == 0].mean()),
        "tpr_at_test_roc_fpr5": float(tpr[fpr <= .05].max())}


def source_bootstrap_plan(frame, split_arrays, samples):
    used = np.any(np.stack([v == "test" for v in split_arrays.values()]), axis=0)
    sources = frame.loc[used, ["source_id", "collection", "label"]].drop_duplicates("source_id").sort_values(["collection", "source_id"]).reset_index(drop=True)
    source_map = {s: i for i, s in enumerate(sources.source_id)}
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.zeros((samples, len(sources)), dtype=np.int16)
    for _, group in sources.groupby("collection", sort=True):
        indices = group.index.to_numpy()
        for b in range(samples):
            draws[b, indices] = np.bincount(rng.integers(len(indices), size=len(indices)), minlength=len(indices))
    sources.to_csv(OUT / "bootstrap_source_union.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_source_multiplicities.npz", source_id=sources.source_id.to_numpy(dtype=str), multiplicities=draws)
    return sources, source_map, draws


def bootstrap_metrics(y, scores, threshold, source_codes, draws):
    """Weighted ROC metrics with ties, from one sort per head.

    Integer source multiplicities are exactly equivalent to duplicating every
    corresponding image row. All variants use identical resamples.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    sorted_codes = source_codes[order]
    positive = (y[order] == 1).astype(float)
    negative = 1 - positive
    ns = draws.shape[1]
    predictions = scores >= threshold
    source_pos = np.bincount(source_codes, weights=(y == 1), minlength=ns)
    source_neg = np.bincount(source_codes, weights=(y == 0), minlength=ns)
    source_tp = np.bincount(source_codes, weights=((y == 1) & predictions), minlength=ns)
    source_fp = np.bincount(source_codes, weights=((y == 0) & predictions), minlength=ns)
    fixed_tpr = (draws @ source_tp) / (draws @ source_pos)
    fixed_fpr = (draws @ source_fp) / (draws @ source_neg)
    result = np.empty((len(draws), len(MEASURES)))
    result[:, 1] = fixed_tpr
    result[:, 2] = fixed_fpr
    for b, mult in enumerate(draws):
        weight = mult[sorted_codes]
        pos = np.add.reduceat(weight * positive, starts)
        neg = np.add.reduceat(weight * negative, starts)
        cp, cn = np.cumsum(pos), np.cumsum(neg)
        result[b, 0] = np.sum(neg * (cp - .5 * pos)) / (cp[-1] * cn[-1])
        allowed = cn / cn[-1] <= .05
        result[b, 3] = float((cp[allowed] / cp[-1]).max()) if allowed.any() else 0.0
    # Internal numerical identity check using the unresampled cohort; this is
    # part of the requested measurement verification, not a separate test.
    ones = np.ones(ns, dtype=np.int16)
    weight = ones[sorted_codes]
    pos, neg = np.add.reduceat(weight * positive, starts), np.add.reduceat(weight * negative, starts)
    cp, cn = np.cumsum(pos), np.cumsum(neg)
    calculated = np.sum(neg * (cp - .5 * pos)) / (cp[-1] * cn[-1])
    if abs(calculated - roc_auc_score(y, scores)) > 1e-12:
        raise ValueError("Weighted ROC identity check failed")
    return result


def run(samples):
    if (OUT / "complete.json").exists() and "jpeg_only" in json.loads((OUT / "complete.json").read_text()).get("variants", []):
        raise RuntimeError("Completed four-arm results exist; use --jpeg-only to reproduce the amendment without overwriting them with the primary three-arm summary")
    started = time.perf_counter()
    frame, parts, eligible, split_arrays = load_design()
    protocol = prepare_protocol(frame, eligible, split_arrays, samples)
    union, source_map, draws = source_bootstrap_plan(frame, split_arrays, samples)
    records, fit_records, comparison_records, bootstrap_values = [], [], [], {}
    clean_lookup = frame.loc[frame.group.eq("R-C")].reset_index().set_index("source_id")["index"]
    if clean_lookup.index.duplicated().any():
        raise ValueError("Ambiguous clean real source mapping")
    for name in ["resnet", "clip"]:
        x = load_features(name, parts)
        print("Loaded", name, x.shape, flush=True)
        for seed, split in split_arrays.items():
            dest = OUT / str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            train = (split == "train") & eligible
            val = (split == "val") & eligible
            test = split == "test"
            tr_indices = np.flatnonzero(train)
            labels = frame.label.to_numpy(dtype=int)
            tr_frame = frame.loc[train]
            replacement_indices = tr_indices.copy()
            replace = tr_frame.group.eq("R-T").to_numpy()
            replacement_indices[replace] = clean_lookup.loc[tr_frame.loc[tr_frame.group.eq("R-T"), "source_id"]].to_numpy(dtype=int)
            if not np.array_equal(frame.source_id.iloc[replacement_indices].to_numpy(), tr_frame.source_id.to_numpy()):
                raise ValueError("Clean-repeat control changed source identities")
            if not np.array_equal(labels[replacement_indices], labels[tr_indices]):
                raise ValueError("Clean-repeat control changed labels")
            testframe = frame.loc[test, ["image_id", "source_id", "label", "generator", "collection", "group", "attack", "severity"]].copy()
            codes = testframe.source_id.map(source_map).to_numpy(dtype=int)
            for variant in VARIANTS:
                indices = replacement_indices if variant == "clean_repeat" else tr_indices
                if variant == "no_matched_reals":
                    indices = tr_indices[~replace]
                path = dest / f"{name}_{variant}.npz"
                print("Fit", seed, name, variant, len(indices), flush=True)
                info = fit_head(x[indices], labels[indices], path)
                if variant != "clean_repeat":
                    with np.load(PRIOR / str(seed) / path.name) as previous, np.load(path) as current:
                        info["max_abs_coefficient_difference_from_prior"] = float(np.max(np.abs(previous["coefficients"] - current["coefficients"])))
                        info["max_abs_intercept_difference_from_prior"] = float(np.max(np.abs(previous["intercept"] - current["intercept"])))
                fit_records.append({"seed": seed, "backbone": name, "variant": variant, **info})
                vscore = linear_scores(x[val], path)
                selected = choose_fpr5_threshold(labels[val], vscore)
                write_json(dest / f"{name}_{variant}_threshold.json", selected)
                tscore = linear_scores(x[test], path)
                testframe[variant + "_logit"] = tscore
                bootstrap_values[(name, seed, variant)] = bootstrap_metrics(labels[test], tscore, selected["logit_threshold"], codes, draws)
                for subset, sub in [("overall", testframe), ("coco", testframe.loc[testframe.label.eq(1) | testframe.collection.eq("coco")]),
                    ("openimages", testframe.loc[testframe.label.eq(1) | testframe.collection.eq("openimages")])]:
                    values = metric_values(sub.label.to_numpy(), sub[variant + "_logit"].to_numpy(), selected["logit_threshold"])
                    records.append({"seed": seed, "backbone": name, "variant": variant, "real_source": subset,
                        "train_rows": len(indices), "train_sources": int(tr_frame.source_id.nunique()), "test_rows": len(sub),
                        "test_sources": int(sub.source_id.nunique()), **selected, **values})
                pd.DataFrame(records).to_csv(OUT / "per_seed_metrics.partial.csv", index=False)
                write_json(OUT / "convergence.partial.json", fit_records)
                print("Measured", seed, name, variant, records[-3]["auc"], records[-3]["tpr_at_validation_fpr5"], "iterations", info["n_iter"], flush=True)
            testframe.to_csv(dest / f"{name}_test_predictions.csv", index=False)
        del x
    result = pd.DataFrame(records)
    result.to_csv(OUT / "per_seed_metrics.csv", index=False)
    summary = result.groupby(["backbone", "variant", "real_source"])[MEASURES].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(OUT / "mean_std_metrics.csv", index=False)
    for name in ["resnet", "clip"]:
        subset = result.loc[result.backbone.eq(name) & result.real_source.eq("overall")]
        for other in ["clean_repeat", "no_matched_reals"]:
            diffs = []
            pointdiffs = []
            for seed in SEEDS:
                difference = bootstrap_values[(name, seed, "full")] - bootstrap_values[(name, seed, other)]
                diffs.append(difference)
                full = subset.loc[subset.seed.eq(seed) & subset.variant.eq("full")].iloc[0]
                baseline = subset.loc[subset.seed.eq(seed) & subset.variant.eq(other)].iloc[0]
                point = np.array([full[m] - baseline[m] for m in MEASURES])
                pointdiffs.append(point)
                for j, measure in enumerate(MEASURES):
                    lo, hi = np.percentile(difference[:, j], [2.5, 97.5])
                    comparison_records.append({"seed": str(seed), "backbone": name, "contrast": "full_minus_" + other,
                        "metric": measure, "estimate": float(point[j]), "ci95_low": float(lo), "ci95_high": float(hi)})
            mean_difference = np.mean(diffs, axis=0)
            mean_point = np.mean(pointdiffs, axis=0)
            for j, measure in enumerate(MEASURES):
                lo, hi = np.percentile(mean_difference[:, j], [2.5, 97.5])
                comparison_records.append({"seed": "five_seed_mean", "backbone": name, "contrast": "full_minus_" + other,
                    "metric": measure, "estimate": float(mean_point[j]), "ci95_low": float(lo), "ci95_high": float(hi)})
    comparisons = pd.DataFrame(comparison_records)
    comparisons.to_csv(OUT / "paired_source_bootstrap.csv", index=False)
    write_json(OUT / "convergence.json", fit_records)
    if any(digest(ROOT / p) != h for p, h in protocol["preserved_head_sha256"].items()):
        raise ValueError("Earlier model head changed during follow-up")
    if any(digest(ROOT / p) != h for p, h in protocol["input_sha256"].items()):
        raise ValueError("Frozen input changed during follow-up")
    outputs = [p for p in OUT.rglob("*") if p.is_file() and p.name not in ["complete.json"]]
    complete = {"completed_at_utc": utc(), "runtime_seconds": time.perf_counter() - started,
        "protocol_frozen_before_all_new_fits": protocol["frozen_at_utc"] < min(r["completed_at_utc"] for r in fit_records),
        "fit_count": len(fit_records), "all_converged": all(r["converged"] for r in fit_records),
        "max_iterations_observed": max(max(r["n_iter"]) for r in fit_records),
        "fit_seconds_total": sum(r["fit_seconds"] for r in fit_records), "source_union": len(union),
        "bootstrap_replicates": samples, "source_union_counts": union.collection.value_counts().astype(int).to_dict(),
        "seed_count": 5, "backbones": ["resnet", "clip"], "variants": VARIANTS,
        "earlier_heads_and_inputs_unchanged": True,
        "main_results": summary.loc[summary.real_source.eq("overall")].to_dict("records"),
        "mean_paired_differences": comparisons.loc[comparisons.seed.eq("five_seed_mean")].to_dict("records"),
        "output_sha256": {p.relative_to(OUT).as_posix(): digest(p) for p in outputs},
        "protocol_sha256": digest(OUT / "protocol.json"), "script_sha256": digest(Path(__file__)),
        "caution": protocol["limitations"], "bootstrap_conditioning": protocol["bootstrap"]["conditioning"]}
    write_json(OUT / "complete.json", complete)
    archive = OUT / "primary_archive"
    archive.mkdir(exist_ok=True)
    for name in ["protocol.json", "complete.json", "convergence.json", "per_seed_metrics.csv", "mean_std_metrics.csv", "paired_source_bootstrap.csv"]:
        if not (archive / name).exists():
            shutil.copy2(OUT / name, archive / name)
    if not (OUT / "initial_script_snapshot.py").exists():
        shutil.copy2(Path(__file__), OUT / "initial_script_snapshot.py")
    print(summary.loc[summary.real_source.eq("overall")].to_string(index=False), flush=True)
    print(comparisons.loc[comparisons.seed.eq("five_seed_mean")].to_string(index=False), flush=True)
    print("Completed in seconds", complete["runtime_seconds"], flush=True)


def run_jpeg_only(samples):
    """Add a narrow augmentation arm after a separately frozen amendment.

    The three-arm primary protocol, script and aggregates remain archived.
    No primary fit, threshold or source resample is changed.
    """
    started = time.perf_counter()
    frame, parts, eligible, split_arrays = load_design()
    protocol = json.loads((OUT / "protocol.json").read_text())
    primary = json.loads((OUT / "primary_archive/complete.json").read_text())
    if samples != protocol["bootstrap"]["replicates"]:
        raise ValueError("JPEG arm must use original shared resamples")
    if digest(OUT / "initial_script_snapshot.py") != protocol["script_sha256"]:
        raise ValueError("Primary scientific source snapshot changed")
    for category in ["input_sha256", "preserved_head_sha256"]:
        if any(digest(ROOT / p) != h for p, h in protocol[category].items()):
            raise ValueError("Earlier inputs or heads changed before amendment")
    amendment = {"frozen_at_utc": utc(), "status": "retrospective amendment fixed before JPEG-only fits",
        "parent_protocol_sha256": digest(OUT / "protocol.json"),
        "preserved_primary_complete_sha256": digest(OUT / "primary_archive/complete.json"),
        "initial_script_snapshot_sha256": digest(OUT / "initial_script_snapshot.py"),
        "script_sha256": digest(Path(__file__)), "variant": "jpeg_only", "seeds": SEEDS,
        "backbones": ["resnet", "clip"], "rows_per_seed": 53326,
        "replacement": "Within each real source's eleven R-T slots in original manifest order, alternate cached JPEG severity 1 (quality 70; six slots) and severity 2 (quality 50; five slots). Keep R-C and R-S, all AI rows, labels, source identities, slots and row order unchanged.",
        "fixed_before_new_fits": ["JPEG qualities", "six/five repetition counts", "original slot order", "source assignments", "fit hyperparameters", "validation operating point", "test cohort", "shared bootstrap resamples"],
        "unchanged_fit_parameters": protocol["logistic_regression_parameters"],
        "unchanged_operating_point": protocol["threshold_selection"], "unchanged_bootstrap": protocol["bootstrap"],
        "selection": "A prespecified narrow JPEG augmentation baseline requested after completion of the clean-repeat comparison; no JPEG-only head or test result was inspected before this amendment.",
        "limitation": "JPEG-only has two distinct transformed vectors per real source rather than eleven diverse edits. Equal slots and class weighting do not equalize augmentation information or diversity; this cannot isolate the unique benefit of exact transformation matching.",
        "helper_source_sha256": {p: digest(ROOT / p) for p in ["scripts/phase3_analyze_controls.py", "scripts/prepare_conference.py", "scripts/evaluate_encoding_controls.py", "src/aiwstudy/attacks.py"]}}
    amendment_path = OUT / "jpeg_protocol_amendment.json"
    if amendment_path.exists():
        saved = json.loads(amendment_path.read_text())
        for key in ["parent_protocol_sha256", "preserved_primary_complete_sha256", "script_sha256", "replacement", "helper_source_sha256"]:
            if saved[key] != amendment[key]:
                raise ValueError("Frozen JPEG-only amendment changed")
        amendment = saved
    else:
        write_json(amendment_path, amendment)
    print("JPEG-only amendment frozen", amendment["frozen_at_utc"], flush=True)
    sources = pd.read_csv(OUT / "bootstrap_source_union.csv", keep_default_na=False)
    with np.load(OUT / "bootstrap_source_multiplicities.npz") as saved:
        if not np.array_equal(saved["source_id"], sources.source_id.to_numpy(dtype=str)):
            raise ValueError("Changed primary source resamples")
        draws = saved["multiplicities"].copy()
    source_map = {s: i for i, s in enumerate(sources.source_id)}
    jpeg_rows = frame.loc[frame.group.eq("R-T") & frame.attack.eq("jpeg")].reset_index()
    jpeg_lookup = jpeg_rows.set_index(["source_id", "severity"])["index"]
    if jpeg_lookup.index.duplicated().any():
        raise ValueError("Ambiguous JPEG vector mapping")
    records, new_fits, comparisons, mean_boot = [], [], [], {}
    for name in ["resnet", "clip"]:
        x = load_features(name, parts)
        for seed, split in split_arrays.items():
            dest = OUT / str(seed)
            training = (split == "train") & eligible
            val = (split == "val") & eligible
            test = split == "test"
            indices = np.flatnonzero(training)
            tr = frame.loc[training]
            replacement_mask = tr.group.eq("R-T").to_numpy()
            targets = tr.loc[tr.group.eq("R-T"), ["source_id"]].copy()
            targets["severity"] = 1 + (targets.groupby("source_id", sort=False).cumcount().to_numpy() % 2)
            targets = pd.MultiIndex.from_frame(targets[["source_id", "severity"]])
            indices[replacement_mask] = jpeg_lookup.loc[targets].to_numpy(dtype=int)
            if not np.array_equal(frame.source_id.iloc[indices].to_numpy(), tr.source_id.to_numpy()):
                raise ValueError("JPEG-only source identity changed")
            labels = frame.label.to_numpy(dtype=int)
            if not np.array_equal(labels[indices], tr.label.to_numpy()):
                raise ValueError("JPEG-only row labels changed")
            path = dest / f"{name}_jpeg_only.npz"
            print("Fit JPEG-only", seed, name, len(indices), flush=True)
            info = fit_head(x[indices], labels[indices], path)
            new_fits.append({"seed": seed, "backbone": name, "variant": "jpeg_only", **info})
            vscore = linear_scores(x[val], path)
            selected = choose_fpr5_threshold(labels[val], vscore)
            write_json(dest / f"{name}_jpeg_only_threshold.json", selected)
            tscore = linear_scores(x[test], path)
            testframe = pd.read_csv(dest / f"{name}_test_predictions.csv", keep_default_na=False)
            if not np.array_equal(testframe.image_id.to_numpy(), frame.loc[test, "image_id"].to_numpy()):
                raise ValueError("Primary test cohort changed")
            testframe["jpeg_only_logit"] = tscore
            testframe.to_csv(dest / f"{name}_jpeg_only_test_predictions.csv", index=False)
            codes = testframe.source_id.map(source_map).to_numpy(dtype=int)
            full_threshold = json.loads((dest / f"{name}_full_threshold.json").read_text())["logit_threshold"]
            full_boot = bootstrap_metrics(labels[test], testframe.full_logit.to_numpy(), full_threshold, codes, draws)
            new_boot = bootstrap_metrics(labels[test], tscore, selected["logit_threshold"], codes, draws)
            difference = full_boot - new_boot
            mean_boot[(name, seed)] = difference
            for subset, sub in [("overall", testframe), ("coco", testframe.loc[testframe.label.eq(1) | testframe.collection.eq("coco")]),
                ("openimages", testframe.loc[testframe.label.eq(1) | testframe.collection.eq("openimages")])]:
                values = metric_values(sub.label.to_numpy(), sub.jpeg_only_logit.to_numpy(), selected["logit_threshold"])
                records.append({"seed": seed, "backbone": name, "variant": "jpeg_only", "real_source": subset,
                    "train_rows": len(indices), "train_sources": int(tr.source_id.nunique()), "test_rows": len(sub),
                    "test_sources": int(sub.source_id.nunique()), **selected, **values})
            new_values = records[-3]
            full_point = metric_values(labels[test], testframe.full_logit.to_numpy(), full_threshold)
            for j, measure in enumerate(MEASURES):
                lo, hi = np.percentile(difference[:, j], [2.5, 97.5])
                comparisons.append({"seed": str(seed), "backbone": name, "contrast": "full_minus_jpeg_only", "metric": measure,
                    "estimate": full_point[measure] - new_values[measure], "ci95_low": float(lo), "ci95_high": float(hi)})
            print("Measured JPEG-only", seed, name, new_values["auc"], new_values["tpr_at_validation_fpr5"], flush=True)
        del x
    result = pd.concat([pd.read_csv(OUT / "primary_archive/per_seed_metrics.csv"), pd.DataFrame(records)], ignore_index=True)
    result.to_csv(OUT / "per_seed_metrics.csv", index=False)
    summary = result.groupby(["backbone", "variant", "real_source"])[MEASURES].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(OUT / "mean_std_metrics.csv", index=False)
    for name in ["resnet", "clip"]:
        difference = np.mean([mean_boot[(name, seed)] for seed in SEEDS], axis=0)
        for j, measure in enumerate(MEASURES):
            point = np.mean([r["estimate"] for r in comparisons if r["backbone"] == name and r["metric"] == measure])
            lo, hi = np.percentile(difference[:, j], [2.5, 97.5])
            comparisons.append({"seed": "five_seed_mean", "backbone": name, "contrast": "full_minus_jpeg_only", "metric": measure,
                "estimate": float(point), "ci95_low": float(lo), "ci95_high": float(hi)})
    contrast = pd.concat([pd.read_csv(OUT / "primary_archive/paired_source_bootstrap.csv"), pd.DataFrame(comparisons)], ignore_index=True)
    contrast.to_csv(OUT / "paired_source_bootstrap.csv", index=False)
    convergence = json.loads((OUT / "primary_archive/convergence.json").read_text()) + new_fits
    write_json(OUT / "convergence.json", convergence)
    for category in ["input_sha256", "preserved_head_sha256"]:
        if any(digest(ROOT / p) != h for p, h in protocol[category].items()):
            raise ValueError("Earlier inputs or heads changed during JPEG follow-up")
    complete = {**primary, "completed_at_utc": utc(), "runtime_seconds": primary["runtime_seconds"] + time.perf_counter() - started,
        "jpeg_followup_runtime_seconds": time.perf_counter() - started, "fit_count": len(convergence),
        "all_converged": all(r["converged"] for r in convergence),
        "max_iterations_observed": max(max(r["n_iter"]) for r in convergence),
        "fit_seconds_total": sum(r["fit_seconds"] for r in convergence), "variants": VARIANTS + ["jpeg_only"],
        "amendment_frozen_before_JPEG_fits": amendment["frozen_at_utc"] < min(r["completed_at_utc"] for r in new_fits),
        "amendment_sha256": digest(amendment_path), "script_sha256": digest(Path(__file__)),
        "main_results": summary.loc[summary.real_source.eq("overall")].to_dict("records"),
        "mean_paired_differences": contrast.loc[contrast.seed.eq("five_seed_mean")].to_dict("records"),
        "caution": protocol["limitations"] + [amendment["limitation"]],
        "initial_three_arm_results_archived": "primary_archive/", "initial_script_snapshot_sha256": digest(OUT / "initial_script_snapshot.py")}
    complete["output_sha256"] = {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob("*") if p.is_file() and p != OUT / "complete.json"}
    write_json(OUT / "complete.json", complete)
    print(summary.loc[summary.real_source.eq("overall")].to_string(index=False), flush=True)
    print(contrast.loc[contrast.seed.eq("five_seed_mean") & contrast.contrast.eq("full_minus_jpeg_only")].to_string(index=False), flush=True)
    print("JPEG-only follow-up seconds", complete["jpeg_followup_runtime_seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--jpeg-only", action="store_true", help="Run the separately frozen JPEG-only amendment after the three-arm primary run")
    args = parser.parse_args()
    if args.samples < 100:
        raise ValueError("At least 100 source-cluster bootstrap replicates required")
    if args.jpeg_only:
        with threadpool_limits(limits=4):
            run_jpeg_only(args.samples)
    elif args.prepare_only:
        frame, _, eligible, split = load_design()
        protocol = prepare_protocol(frame, eligible, split, args.samples)
        print(json.dumps({"protocol": str(OUT / "protocol.json"), "frozen_at_utc": protocol["frozen_at_utc"], "counts": protocol["counts"]}, indent=2))
    else:
        with threadpool_limits(limits=4):
            run(args.samples)


if __name__ == "__main__":
    main()
