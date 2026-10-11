"""Cached unmarked-feature factorial diagnostic of training-process matching.

All four cells keep source composition, slots and optimization fixed. Two
different process families occur per origin per cell. The protocol precedes
new fits; existing heads and analyses are never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase4_size_controlled import (MEASURES, SEEDS, bootstrap_metrics,
    choose_fpr5_threshold, fit_head, frozen_inputs, linear_scores, load_design,
    load_features, metric_values, preserved_heads)
from evaluate_encoding_controls import digest

OUT = ROOT / "outputs/phase5/matching_factorial"
SETS = {"A": [("jpeg", 1), ("blur", 1)], "B": [("noise", 1), ("resize", 1)]}
CELLS = ["AA", "AB", "BA", "BB"]
BOOT_SEED = 20261013


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def design():
    whole, parts, _, whole_splits = load_design()
    unmarked_ids = set(whole.loc[whole.group.eq("S-U-A"), "source_id"])
    keys = pd.MultiIndex.from_frame(whole[["attack", "severity"]])
    keep = (whole.label.eq(0) | whole.source_id.isin(unmarked_ids)) & (
        whole.group.isin(["R-C", "S-U"]) | (
            whole.group.isin(["R-T", "S-U-A"]) & keys.isin(SETS["A"] + SETS["B"])))
    whole_indices = np.flatnonzero(keep.to_numpy())
    frame = whole.loc[keep].copy().reset_index(drop=True)
    splits = {seed: values[whole_indices] for seed, values in whole_splits.items()}
    if len(frame) != 28580 or frame.source_id.nunique() != 5716:
        raise ValueError("Cached factorial pool changed")
    if not frame.groupby("source_id").size().eq(5).all():
        raise ValueError("Each source requires clean plus all four unmarked edits")
    if not set(frame.group) <= {"R-C", "S-U", "R-T", "S-U-A"}:
        raise ValueError("Marked features entered unmarked factorial")
    lookup = frame.reset_index().set_index(["source_id", "attack", "severity"])["index"]
    if lookup.index.duplicated().any():
        raise ValueError("Ambiguous frozen source/condition feature")
    for seed, values in splits.items():
        test = frame.loc[values == "test"]
        if len(test) != 9215 or test.source_id.nunique() != 1843:
            raise ValueError("Existing full five-seed test sources were not preserved")
    return whole, parts, whole_indices, frame, splits, lookup


def training_indices(frame, splits, lookup, seed, cell):
    sources = frame.loc[splits[seed] == "train"].groupby("source_id")["label"].first().sort_index()
    indices = []
    structural_ids = []
    for sid, label in sources.items():
        edits = SETS[cell[1] if int(label) == 1 else cell[0]]
        for slot, key in enumerate([("identity", 0), *edits]):
            indices.append(int(lookup.loc[(sid, *key)]))
            structural_ids.append((sid, slot, int(label)))
    return np.asarray(indices, dtype=int), structural_ids


def prepare(frame, splits, lookup, samples):
    OUT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "outputs/phase5/paired_content/selection_complete.json").exists():
        raise RuntimeError("Root scheduled factorial fitting only after paired-content selection is locked")
    counts = []
    for seed, values in splits.items():
        indices, structural = training_indices(frame, splits, lookup, seed, "AA")
        for cell in CELLS[1:]:
            alternative, alternative_structural = training_indices(frame, splits, lookup, seed, cell)
            if structural != alternative_structural or not np.array_equal(frame.label.iloc[indices], frame.label.iloc[alternative]):
                raise ValueError("Factorial source/slot/label composition differs")
        train = frame.iloc[indices]
        counts.append({"seed": seed, "train_rows": len(indices), "train_sources": int(train.source_id.nunique()),
            "class_counts": {str(k): int(v) for k, v in train.label.value_counts().items()},
            "validation_rows": int((values == "val").sum()), "validation_sources": int(frame.loc[values == "val", "source_id"].nunique()),
            "test_rows": 9215, "test_sources": 1843})
    value = {"frozen_at_utc": utc(), "status": "Retrospective restricted fitting-pool diagnostic, fixed before all new fits",
        "seeds": SEEDS, "backbones": ["resnet", "clip"], "cells": CELLS,
        "cell_order": "First letter selects real-source edit set; second selects unmarked AI-source edit set",
        "augmentation_sets": {"A": "JPEG70 and Gaussian blur3", "B": "Gaussian noise sigma6 and resize0.8"},
        "augmentation_keys": {name: [list(key) for key in keys] for name, keys in SETS.items()},
        "training": "Each source contributes one clean feature and one feature for each of the two specified distinct process families. No marked AI or sham-marked real feature is used.",
        "pool": {"sources": 5716, "real": 2920, "AI_with_unmarked_edit_caches": 2796,
            "non_midjourney_AI": 1834, "midjourney_test_only": 962,
            "collection_sources": frame.groupby("collection").source_id.nunique().astype(int).to_dict(),
            "restriction": "Fitting/validation AI sources without all four cached unmarked edits are excluded. Every existing five-seed test source remains available and is retained. No new extraction is performed."},
        "unchanged_across_cells": ["training source identities", "structural source/slot ordering", "labels", "row count", "class weights", "frozen backbones", "solver and optimization budget", "validation sources/conditions", "test sources/conditions"],
        "primary_endpoint": "Five-seed mean overall origin ROC-AUC contrast 0.5*(AA+BB-AB-BA) on clean plus the equal-frequency union of four unmarked edits",
        "primary_contrast": "Mean matched cells AA/BB minus mean mismatched cells AB/BA; factor levels and their marginal frequency are balanced",
        "secondary_endpoints": "Fixed-validation-threshold TPR and actual test FPR; descriptive testROC TPR5; identity/per-condition and per-real-collection AUC",
        "validation_test": "Each held-out source contributes clean and all four unmarked edits, regardless of the cell used for fitting. The same five original source assignments are reused. All 962 Midjourney sources stay test only.",
        "threshold": "Lowest finite logit threshold attaining validation FPR<=0.05 with whole ties, frozen for test; no test threshold tuning",
        "counts": counts,
        "logistic_regression_parameters": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0).get_params(),
        "bootstrap": {"replicates": samples, "seed": BOOT_SEED,
            "unit": "Unique source in the union of all five test cohorts, retaining all derivatives and split membership",
            "stratification": "Collection; identical multiplicities for all four cells, both backbones and five split means",
            "conditioning": "Fixed fitted heads, validation thresholds and split/source membership; no training uncertainty. Overlapping tests and shared Midjourney holdout are not independent repetitions.",
            "interval": "Percentile95% paired source-cluster interval for the prespecified five-seed interaction contrast"},
        "limitations": ["Condition counts and family counts are equal, but distortion magnitude and information content are not equalized.",
            "The balanced factorial estimates process-matching interaction for these four fixed transforms, frozen features and this restricted fitting pool; it does not establish that matching uniquely explains the broad edit-suite gain.",
            "All source pools and prior results were already examined. This is retrospective and separate from the prospective caption-paired test.",
            "Removing marked training examples changes the original training scope; factorial effects must not be presented as the original full-versus-removal effect."],
        "input_sha256": frozen_inputs(), "preserved_head_sha256": preserved_heads(),
        "helper_script_sha256": digest(ROOT / "scripts/phase4_size_controlled.py"),
        "script_sha256": digest(Path(__file__)), "device": "CPU", "thread_limit": 4,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]}}
    target = OUT / "protocol.json"
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
        for key in ["counts", "input_sha256", "preserved_head_sha256", "helper_script_sha256", "script_sha256", "bootstrap", "augmentation_keys"]:
            if prior[key] != value[key]:
                raise ValueError(f"Frozen factorial protocol changed: {key}")
        return prior
    write_json(target, value)
    print("Factorial protocol frozen", value["frozen_at_utc"], flush=True)
    return value


def bootstrap_plan(frame, splits, samples):
    any_test = np.any(np.stack([values == "test" for values in splits.values()]), axis=0)
    union = frame.loc[any_test, ["source_id", "collection"]].drop_duplicates().sort_values(["collection", "source_id"]).reset_index(drop=True)
    if union.source_id.duplicated().any():
        raise ValueError("Bootstrap source maps to multiple collections")
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.zeros((samples, len(union)), dtype=np.int16)
    for kind in sorted(union.collection.unique()):
        indices = np.flatnonzero(union.collection.eq(kind))
        draws[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=samples).astype(np.int16)
    path = OUT / "bootstrap_source_multiplicities.npz"
    if path.exists():
        with np.load(path) as saved:
            if not np.array_equal(saved["source_id"], union.source_id.to_numpy()) or not np.array_equal(saved["multiplicities"], draws):
                raise ValueError("Frozen factorial resamples changed")
    else:
        np.savez(path, source_id=union.source_id.to_numpy(dtype=str), multiplicities=draws)
        union.to_csv(OUT / "bootstrap_source_union.csv", index=False)
    return union, {sid: index for index, sid in enumerate(union.source_id)}, draws


def run(samples, prepare_only=False):
    started = time.perf_counter()
    whole, parts, whole_indices, frame, splits, lookup = design()
    protocol = prepare(frame, splits, lookup, samples)
    if prepare_only:
        return
    union, source_map, draws = bootstrap_plan(frame, splits, samples)
    records, fit_records, boot_values = [], [], {}
    labels = frame.label.to_numpy(dtype=int)
    for backbone in ["resnet", "clip"]:
        whole_x = load_features(backbone, parts)
        x = whole_x[whole_indices].copy()
        del whole_x
        print("Loaded unmarked factorial", backbone, x.shape, flush=True)
        for seed, values in splits.items():
            dest = OUT / str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            val, test = values == "val", values == "test"
            prediction = frame.loc[test, ["image_id", "source_id", "label", "collection", "attack", "severity"]].copy()
            codes = prediction.source_id.map(source_map).to_numpy(dtype=int)
            for cell in CELLS:
                indices, _ = training_indices(frame, splits, lookup, seed, cell)
                path = dest / f"{backbone}_{cell}.npz"
                info = fit_head(x[indices], labels[indices], path)
                fit_records.append({"seed": seed, "backbone": backbone, "cell": cell, **info})
                chosen = choose_fpr5_threshold(labels[val], linear_scores(x[val], path))
                write_json(dest / f"{backbone}_{cell}_threshold.json", chosen)
                scores = linear_scores(x[test], path)
                prediction[cell + "_logit"] = scores
                boot_values[(backbone, seed, cell)] = bootstrap_metrics(labels[test], scores, chosen["logit_threshold"], codes, draws)
                for scope, selected in [("overall", prediction),
                    ("identity", prediction.loc[prediction.attack.eq("identity")]),
                    *[(name, prediction.loc[prediction.attack.eq(name)]) for name in ["jpeg", "blur", "noise", "resize"]],
                    ("coco", prediction.loc[prediction.label.eq(1) | prediction.collection.eq("coco")]),
                    ("openimages", prediction.loc[prediction.label.eq(1) | prediction.collection.eq("openimages")])]:
                    metric = metric_values(selected.label.to_numpy(), selected[cell + "_logit"].to_numpy(), chosen["logit_threshold"])
                    records.append({"seed": seed, "backbone": backbone, "cell": cell, "scope": scope,
                        "train_rows": len(indices), "train_sources": int(frame.source_id.iloc[indices].nunique()),
                        "test_rows": len(selected), "test_sources": int(selected.source_id.nunique()), **chosen, **metric})
                pd.DataFrame(records).to_csv(OUT / "per_seed_metrics.partial.csv", index=False)
                write_json(OUT / "convergence.partial.json", fit_records)
                print("Measured", backbone, seed, cell, records[-8]["auc"], info["n_iter"], flush=True)
            prediction.to_csv(dest / f"{backbone}_test_predictions.csv", index=False)
        del x
    result = pd.DataFrame(records)
    result.to_csv(OUT / "per_seed_metrics.csv", index=False)
    summary = result.groupby(["backbone", "cell", "scope"])[MEASURES].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary.to_csv(OUT / "mean_std_metrics.csv", index=False)
    comparisons = []
    overall = result.loc[result.scope.eq("overall")].set_index(["backbone", "seed", "cell"])
    for backbone in ["resnet", "clip"]:
        seed_boot, seed_points = [], []
        for seed in SEEDS:
            values = .5 * (boot_values[(backbone, seed, "AA")] + boot_values[(backbone, seed, "BB")]
                             - boot_values[(backbone, seed, "AB")] - boot_values[(backbone, seed, "BA")])
            point = .5 * (overall.loc[(backbone, seed, "AA"), MEASURES].to_numpy(dtype=float)
                            + overall.loc[(backbone, seed, "BB"), MEASURES].to_numpy(dtype=float)
                            - overall.loc[(backbone, seed, "AB"), MEASURES].to_numpy(dtype=float)
                            - overall.loc[(backbone, seed, "BA"), MEASURES].to_numpy(dtype=float))
            seed_boot.append(values)
            seed_points.append(point)
            for index, metric in enumerate(MEASURES):
                low, high = np.percentile(values[:, index], [2.5, 97.5])
                comparisons.append({"seed": str(seed), "backbone": backbone, "contrast": "mean_matched_minus_mean_mismatched",
                    "metric": metric, "estimate": float(point[index]), "ci95_low": float(low), "ci95_high": float(high)})
        average_boot, average_point = np.mean(seed_boot, axis=0), np.mean(seed_points, axis=0)
        for index, metric in enumerate(MEASURES):
            low, high = np.percentile(average_boot[:, index], [2.5, 97.5])
            comparisons.append({"seed": "five_seed_mean", "backbone": backbone, "contrast": "mean_matched_minus_mean_mismatched",
                "metric": metric, "estimate": float(average_point[index]), "ci95_low": float(low), "ci95_high": float(high)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "paired_source_bootstrap.csv", index=False)
    write_json(OUT / "convergence.json", fit_records)
    for category in ["input_sha256", "preserved_head_sha256"]:
        if any(digest(ROOT / path) != expected for path, expected in protocol[category].items()):
            raise ValueError("Existing input or model head changed during factorial diagnostic")
    complete = {"completed_at_utc": utc(), "runtime_seconds": time.perf_counter() - started,
        "fit_count": len(fit_records), "all_converged": all(row["converged"] for row in fit_records),
        "max_iterations": max(max(row["n_iter"]) for row in fit_records),
        "total_fit_seconds": sum(row["fit_seconds"] for row in fit_records),
        "protocol_frozen_before_new_fits": protocol["frozen_at_utc"] < min(row["completed_at_utc"] for row in fit_records),
        "original_heads_and_inputs_unchanged": True,
        "pool_sources": 5716, "source_union": len(union), "test_sources_per_seed": 1843,
        "test_rows_per_seed": 9215, "backbones": ["resnet", "clip"], "seeds": SEEDS, "cells": CELLS,
        "counts": protocol["counts"], "bootstrap": protocol["bootstrap"],
        "main_results": summary.loc[summary.scope.eq("overall")].to_dict("records"),
        "mean_paired_differences": comparison.loc[comparison.seed.eq("five_seed_mean")].to_dict("records"),
        "limitations": protocol["limitations"], "protocol_sha256": digest(OUT / "protocol.json"),
        "script_sha256": digest(Path(__file__)),
        "output_sha256": {path.relative_to(OUT).as_posix(): digest(path) for path in OUT.rglob("*") if path.is_file() and path.name != "complete.json"}}
    write_json(OUT / "complete.json", complete)
    print(json.dumps({"runtime_seconds": complete["runtime_seconds"], "all_converged": complete["all_converged"],
                      "main_results": complete["main_results"], "mean_paired_differences": complete["mean_paired_differences"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.bootstrap_samples, args.prepare_only)


if __name__ == "__main__":
    main()
