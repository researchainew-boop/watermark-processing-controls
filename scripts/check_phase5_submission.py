"""Read-only integrity and numerical checks for the completed Phase5 evidence.

No fitting, image generation, learned inference, or threshold selection occurs.
The experiment gate is callable by the local release packager. Manuscript/PDF
checks are a separate optional gate and never manufacture a visual-review claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "outputs/phase5"
SEEDS = list(range(20260920, 20260925))
POLICIES = ["per_image_core13", "conformal_core13", "conformal_all16"]
SYSTEMS = ["probe", "fusion", "gated"]
_hashes = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv(path):
    # Payloads are bit strings: numeric inference would erase a leading zero.
    # Reconstruct float64 scores exactly so an archived threshold tie is kept.
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip",
                       dtype={"payload": str, "embedded_payload": str, "query_payload": str})


def digest(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hashes:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        _hashes[key] = value.hexdigest()
    return _hashes[key]


def hashes(base, mapping):
    for name, value in mapping.items():
        require(digest(base / name) == value, f"Changed locked artifact: {name}")


def close(actual, expected, context, tolerance=2e-11):
    require(math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=tolerance),
            f"Numerical mismatch: {context}; {actual} versus {expected}")


def stamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Naive provenance timestamp")
    return parsed


def chronological(*values):
    dates = [stamp(value) for value in values]
    require(all(a <= b for a, b in zip(dates, dates[1:])), "Experiment chronology is inconsistent")


def interval(low, high, bounded=True):
    require(math.isfinite(float(low)) and math.isfinite(float(high)) and float(low) <= float(high),
            "Invalid uncertainty interval")
    if bounded:
        require(0 <= float(low) <= float(high) <= 1, "Rate interval outside [0,1]")


def fpr_cut(values):
    values = np.asarray(values, dtype=np.float64)
    unique, counts = np.unique(values, return_counts=True)
    accepted = np.cumsum(counts[::-1])[::-1]
    eligible = np.flatnonzero(accepted / len(values) <= .05)
    return float(unique[eligible[0]]) if len(eligible) else float(np.nextafter(values.max(), np.inf))


def head_score(vectors, path):
    with np.load(path, allow_pickle=False) as model:
        return vectors.astype(np.float64) @ model["coefficients"].reshape(-1) + float(model["intercept"].reshape(-1)[0])


def score_systems(vectors, active, probe, fusion):
    p = head_score(vectors, probe)
    f = head_score(np.column_stack([expit(p), active]), fusion)
    gate = np.where(np.asarray(active) >= 13 / 16, f, p)
    require(np.array_equal(gate[np.asarray(active) < 13 / 16], p[np.asarray(active) < 13 / 16]),
            "Gate fallback is not the fixed probe logit")
    return {"probe": p, "fusion": f, "gated": gate}


def accept(scores, policy):
    if policy["threshold_logit"] is None:
        require(policy["infinite_threshold"], "Null finite threshold")
        return np.zeros(len(scores), dtype=bool)
    require(policy["comparison"] in [">", ">="], "Unspecified tie rule")
    return scores > policy["threshold_logit"] if policy["comparison"] == ">" else scores >= policy["threshold_logit"]


def group_masks(frame, loo=False):
    return {"overall": np.ones(len(frame), dtype=bool),
        "coco": frame.collection.eq("coco").to_numpy(),
        "openimages": frame.collection.eq("openimages").to_numpy(),
        "seen_generators": frame.label.eq(1).to_numpy() & (frame.collection.eq("dalle-3").to_numpy() if loo else ~frame.collection.eq("midjourney").to_numpy()),
        "heldout_stable_diffusion": frame.collection.eq("stable-diffusion").to_numpy(),
        "heldout_midjourney": frame.collection.eq("midjourney").to_numpy(),
        "marked_AI": frame.label.eq(1).to_numpy() & frame.group.isin(["S-W", "S-A"]).to_numpy(),
        "unmarked_AI": frame.label.eq(1).to_numpy() & frame.group.isin(["S-U", "S-U-A"]).to_numpy()}


def check_calibration():
    from phase5_calibration import base_frame, complete_features, HELD
    work = P5 / "calibration"
    protocol, roles = read(work / "protocol.json"), csv(work / "source_roles.csv")
    hashes(ROOT, protocol["input_sha256"])
    require(digest(work / "source_roles.csv") == protocol["source_roles_sha256"], "Calibration roles changed")
    require(protocol["retrospective"] and not protocol["new400_is_untouched"] and not protocol["test_threshold_tuning"], "Calibration reuse not acknowledged")
    assignment = csv(ROOT / "outputs/phase3/controls/split_seeds/assignments.csv")
    for seed in SEEDS:
        part = roles.loc[roles.seed.eq(seed)]
        require(len(part) == 6823 and not part.source_id.duplicated().any(), "Calibration source partition mismatch")
        saved = assignment.loc[assignment.seed.eq(seed)].set_index("source_id").split
        require(part.loc[part.role.isin(["train", "test"])].apply(lambda r: saved[r.source_id] == r.role, axis=1).all(), "Original source split changed")
        val = part.loc[part.role.isin(["fusion_fit", "calibration"])]
        require(val.source_id.map(saved).eq("val").all(), "Non-validation source used in fit/calibration")
        for collection, group in val.groupby("collection"):
            ids = sorted(group.source_id, key=lambda s: hashlib.sha256(f"phase5|{seed}|validation-half|{collection}|{s}".encode()).hexdigest())
            require(set(group.loc[group.role.eq("fusion_fit"), "source_id"]) == set(ids[:len(ids)//2]), "Validation half differs from fixed hash rule")
        counts = part.loc[part.role.eq("calibration") & part.label.eq(0)].collection.value_counts().to_dict()
        require(counts == {"coco": 144, "openimages": 75}, "Negative calibration scope changed")
        require(part.loc[part.collection.eq("midjourney"), "role"].eq("test").all(), "Midjourney fit leakage")
    extra = read(work / "missing_features.json")
    inference = read(work / "clip_calibration_extra_inference.json")
    require(extra["rows"] == 792 and extra["sources"] == 264, "Missing CLIP scope mismatch")
    require(digest(work / "calibration_extra_manifest.csv") == extra["manifest_sha256"] == inference["manifest_sha256"], "Extra CLIP manifest mismatch")
    require(digest(work / "clip_calibration_extra_embeddings.npz") == inference["embedding_sha256"], "Extra CLIP cache changed")
    require(min(inference["anchor_cosine"]) > .99999 and max(inference["anchor_relative_l2"]) < .005, "CLIP anchors failed")
    chronological(protocol["locked_at_utc"], extra["locked_at_utc"], inference["started_at_utc"], inference["completed_at_utc"])
    frame, frames, paths = base_frame(extra=True)
    fresh_path = ROOT / "outputs/phase4/fresh_replication"
    fresh = csv(fresh_path / "manifest.csv")
    fresh["group"] = np.where(fresh.label.eq(0), "R-C", np.where(fresh.mark.eq("marked"), "S-A", "S-U-A"))
    regimes = []
    for directory, loo in [(work, False), (work / "sd_family_excluded", True)]:
        done, heads, thresholds = (read(directory / name) for name in ["complete.json", "heads_locked.json", "thresholds_locked.json"])
        hashes(directory, done["output_sha256"])
        hashes(ROOT, done["new_heads_sha256"])
        require(done["fit_count"] == (4 if loo else 10) and done["calibration_policy_count"] == (18 if loo else 90), "Calibration fit/policy count")
        require(done["earlier_heads_unchanged"] and not done["test_threshold_tuning"], "Earlier calibration head/threshold invariant")
        started = read(directory / "analysis_started.json")
        require(started["thresholds_locked_sha256"] == digest(directory / "thresholds_locked.json") == read(directory / "summary.json")["thresholds_locked_sha256"], "Threshold lock changed after scoring")
        require(started["heads_locked_sha256"] == digest(directory / "heads_locked.json"), "Head lock changed after scoring")
        chronological(heads["locked_at_utc"], thresholds["locked_at_utc"], started["test_scoring_started_at_utc"], done["completed_at_utc"])
        local_protocol = read(directory / "protocol.json")
        for fit in heads["fits"]:
            require(fit["converged"] and max(fit["n_iter"]) < 3000, "Unconverged calibration head")
            chronological(local_protocol["locked_at_utc"], fit["fit_started_at_utc"], fit["fit_completed_at_utc"], heads["locked_at_utc"])
            if loo:
                require(not fit["SD_or_Midjourney_sources_used_for_fit"] and not set(fit["collection_counts"]) & {"stable-diffusion", "midjourney"}, "SD exclusion failed")
            else:
                require(fit["sources"] == 439 and fit["rows"] == 5707 and not fit["calibration_or_test_sources_used_for_fit"], "Fusion reused calibration sources")
        if loo:
            local_roles = csv(directory / "source_roles.csv")
            require(local_roles.loc[local_roles.role.isin(["train", "fusion_fit", "calibration"]), "collection"].isin(["coco", "openimages", "dalle-3"]).all(), "Held AI family entered supervised/calibration roles")
        regimes.append((directory, loo, thresholds, csv(directory / "per_seed_metrics.csv"), csv(directory / "mean_std_ci.csv")))
    # Recompute all negative cuts and every saved test point from fixed cached
    # features. This never selects thresholds: the independently calculated cut
    # must equal the existing lock before any test point is inspected.
    for backbone in ["resnet", "clip"]:
        x = complete_features(backbone, frames, paths)
        with np.load(fresh_path / f"{backbone}_embeddings.npz", allow_pickle=True) as z:
            require(str(z["manifest_sha256"]) == digest(fresh_path / "manifest.csv") and np.array_equal(z["image_id"].astype(str), fresh.image_id.to_numpy()), "Fresh follow-up cache mismatch")
            xf = z["embedding"].astype(np.float32)
        for directory, loo, thresholds, metrics, means in regimes:
            for seed in ([SEEDS[0]] if loo else SEEDS):
                part = roles.loc[roles.seed.eq(seed)]
                cal_ids = set(part.loc[part.role.eq("calibration"), "source_id"])
                if loo:
                    cal_ids -= set(part.loc[part.collection.isin(["stable-diffusion", "midjourney"]), "source_id"])
                cal_mask = frame.source_id.isin(cal_ids).to_numpy()
                cal = frame.loc[cal_mask].reset_index(drop=True)
                probe = directory / f"{backbone}_probe.npz" if loo else ROOT / f"outputs/phase3/controls/split_seeds/{seed}/{backbone}_full.npz"
                fusion = directory / f"{backbone}_fusion.npz" if loo else directory / str(seed) / f"{backbone}_fusion.npz"
                model = thresholds["models"][backbone if loo else f"{seed}_{backbone}"]
                require(digest(probe) == model["probe_sha256"] and digest(fusion) == model["fusion_sha256"], "Calibrated fitted head mismatch")
                scores = score_systems(x[cal_mask], cal.active_score.to_numpy(), probe, fusion)
                core = np.array([(a, int(s)) not in HELD for a, s in zip(cal.attack, cal.severity)])
                for system, score in scores.items():
                    policies = model["systems"][system]
                    require(set(policies) == set(POLICIES), "Missing prespecified policy")
                    neg = cal.label.eq(0).to_numpy() & core
                    close(policies["per_image_core13"]["threshold_logit"], fpr_cut(score[neg]), "Independent per-copy threshold")
                    close(policies["per_image_core13"]["calibration_copy_fpr"], (score[neg] >= policies["per_image_core13"]["threshold_logit"]).mean(), "Calibration copy FPR")
                    require(policies["per_image_core13"]["calibration_copy_fpr"] <= .05 and policies["per_image_core13"]["comparison"] == ">=", "Copy threshold ties/target")
                    for family in ["core13", "all16"]:
                        policy = policies["conformal_" + family]
                        cuts = []
                        for collection, n, k in [("coco", 144, 138), ("openimages", 75, 73)]:
                            pick = cal.label.eq(0).to_numpy() & cal.collection.eq(collection).to_numpy() & (core if family == "core13" else True)
                            maxima = cal.loc[pick, ["source_id"]].assign(score=score[pick]).groupby("source_id").score.agg(["max", "count"])
                            require(len(maxima) == n and maxima["count"].eq(13 if family == "core13" else 16).all(), "Incomplete negative source bundle")
                            q = float(np.sort(maxima["max"].to_numpy())[k-1])
                            record = policy["collection_cuts"][collection]
                            require(record["calibration_real_sources"] == n and record["k_one_indexed"] == k and k == math.ceil((n+1)*.95), "Finite quantile index")
                            close(record["threshold_logit"], q, "Conformal source-max quantile")
                            close(record["finite_source_marginal_bound"], (n+1-k)/(n+1), "Marginal rank bound")
                            close(record["calibration_source_any_fpr"], (maxima["max"] > q).mean(), "Strict ties source calibration")
                            cuts.append(q)
                        close(policy["threshold_logit"], max(cuts), "Global maximum collection cut")
                        require(policy["comparison"] == ">" and policy["globally_deployable"], "Conformal deployment/tie rule")
                test_mask = frame.source_id.isin(set(part.loc[part.role.eq("test"), "source_id"])).to_numpy()
                batches = [("original", frame.loc[test_mask].reset_index(drop=True), x[test_mask]), ("fresh400_followup", fresh, xf)]
                for dataset, batch, vectors in batches:
                    measured = metrics.loc[metrics.seed.eq(seed) & metrics.backbone.eq(backbone) & metrics.dataset.eq(dataset)]
                    score_map = score_systems(vectors, batch.active_score.to_numpy(), probe, fusion)
                    masks = group_masks(batch, loo)
                    held = np.array([(a, int(s)) in HELD for a, s in zip(batch.attack, batch.severity)])
                    for value in measured.to_dict("records"):
                        scope = ~held if value["processing_scope"] == "core13" else np.ones(len(batch), dtype=bool)
                        sign = 1 if value["quantity"] == "copy_TPR" else 0
                        chosen = scope & masks[value["group"]] & batch.label.eq(sign).to_numpy()
                        prediction = accept(score_map[value["system"]], model["systems"][value["system"]][value["policy"]])
                        events = batch.loc[chosen, ["source_id"]].assign(event=prediction[chosen])
                        rate = events.groupby("source_id").event.max().mean() if value["quantity"] == "source_any_FPR" else events.event.mean()
                        close(value["rate"], rate, "Calibration follow-up test rate")
                        require(value["source_count"] == events.source_id.nunique() and value["copy_count"] == len(events), "Source/copy rate denominator")
                        interval(value["ci95_low"], value["ci95_high"])
        del x, xf
    for directory, loo, thresholds, records, means in regimes:
        keys = ["backbone", "system", "policy", "dataset", "processing_scope", "group", "quantity"]
        indexed = means.set_index(keys)
        for key, part in records.groupby(keys):
            row = indexed.loc[key]
            require(len(part) == (1 if loo else 5), "Missing split rate")
            close(row["mean"], part.rate.mean(), "Five-split rate mean")
            if not loo:
                close(row["std"], part.rate.std(ddof=1), "Five-split rate standard deviation")
            interval(row["mean_ci95_low"], row["mean_ci95_high"])
        for value in csv(directory / "paired_policy_differences.csv").to_dict("records"):
            criteria = {k: value[k] for k in keys if k != "policy"}
            target = indexed.loc[tuple(criteria.get(k, value["policy_minus_baseline"]) for k in keys)]
            baseline = indexed.loc[tuple(criteria.get(k, "per_image_core13") for k in keys)]
            close(value["difference"], target["mean"] - baseline["mean"], "Paired threshold-policy mean contrast")
            interval(value["ci95_low"], value["ci95_high"], bounded=False)
        for value in csv(directory / "source_event_intervals.csv").to_dict("records"):
            n, count = int(value["source_count"]), int(value["source_any_positive_count"])
            close(value["rate"], count/n, "Source event integer count")
            close(value["exact_binomial_ci95_low"], beta.ppf(.025, count, n-count+1) if count else 0., "Exact source-event lower endpoint")
            close(value["exact_binomial_ci95_high"], beta.ppf(.975, count+1, n-count) if count<n else 1., "Exact source-event upper endpoint")
    return {"independent_fusion_fits": 10, "SD_exclusion_fits": 4, "threshold_policies": 108,
        "calibration_extra_CLIP_rows": 792, "source_event_interval_rows": 648,
        "source_disjoint_roles_and_exact_negative_cuts": True, "test_point_rates_recomputed": True,
        "new400_retrospective_followup": True}


def check_matching_factorial():
    from phase5_matching_factorial import design, training_indices, CELLS
    from phase4_size_controlled import MEASURES, metric_values, bootstrap_metrics, load_features
    directory = P5 / "matching_factorial"
    protocol, done, fits = (read(directory / name) for name in ["protocol.json", "complete.json", "convergence.json"])
    hashes(ROOT, protocol["input_sha256"])
    hashes(ROOT, protocol["preserved_head_sha256"])
    hashes(directory, done["output_sha256"])
    require(done["fit_count"] == len(fits) == 40 and done["all_converged"] and done["original_heads_and_inputs_unchanged"], "Factorial completion/convergence")
    require(done["protocol_sha256"] == digest(directory / "protocol.json"), "Factorial protocol hash")
    _, parts, whole_indices, frame, splits, lookup = design()
    require(frame.source_id.nunique() == 5716 and frame.groupby("source_id").size().eq(5).all(), "Restricted factorial pool mismatch")
    require(set(frame.group) <= {"R-C", "S-U", "R-T", "S-U-A"}, "Marked features entered factorial")
    for seed in SEEDS:
        structure = training_indices(frame, splits, lookup, seed, "AA")[1]
        for cell in CELLS:
            _, other = training_indices(frame, splits, lookup, seed, cell)
            require(structure == other, "Factorial source/slot/class composition differs")
    for fit in fits:
        require(fit["converged"] and not fit["warnings"] and max(fit["n_iter"]) < fit["iteration_cap"], "Factorial fit did not converge")
        start = stamp(fit["completed_at_utc"]) - timedelta(seconds=fit["fit_seconds"])
        require(start > stamp(protocol["frozen_at_utc"]), "Factorial fit began before protocol")
        head = directory / str(fit["seed"]) / f"{fit['backbone']}_{fit['cell']}.npz"
        require(digest(head) == fit["model_sha256"], "Factorial head changed")
        for label, count in fit["class_counts"].items():
            close(fit["balanced_class_weights"][label], fit["rows"]/(2*count), "Factorial class weight")
    records, means = csv(directory / "per_seed_metrics.csv"), csv(directory / "mean_std_metrics.csv")
    require(len(records) == 320 and len(means) == 64, "Factorial table row count")
    for key, part in records.groupby(["backbone", "cell", "scope"]):
        row = means.set_index(["backbone", "cell", "scope"]).loc[key]
        for metric in MEASURES:
            close(row[metric+"_mean"], part[metric].mean(), "Factorial mean")
            close(row[metric+"_std"], part[metric].std(ddof=1), "Factorial split standard deviation")
    union = csv(directory / "bootstrap_source_union.csv")
    with np.load(directory / "bootstrap_source_multiplicities.npz", allow_pickle=False) as z:
        require(np.array_equal(z["source_id"].astype(str), union.source_id.to_numpy()), "Factorial bootstrap source identity")
        draws = z["multiplicities"]
    require(draws.shape == (2000, 4238), "Factorial source bootstrap shape")
    source_map = pd.Series(union.index, index=union.source_id)
    boot = {}
    for backbone in ["resnet", "clip"]:
        whole_x = load_features(backbone, parts)
        x = whole_x[whole_indices]
        del whole_x
        for seed in SEEDS:
            prediction = csv(directory / str(seed) / f"{backbone}_test_predictions.csv")
            require(len(prediction) == 9215 and prediction.source_id.nunique() == 1843, "Factorial full test source retention")
            require(prediction.groupby("source_id").size().eq(5).all(), "Factorial test source copies")
            for cell in CELLS:
                cut = read(directory / str(seed) / f"{backbone}_{cell}_threshold.json")
                require(cut["validation_fpr"] <= .05, "Factorial validation FPR target")
                val = splits[seed] == "val"
                val_score = head_score(x[val], directory / str(seed) / f"{backbone}_{cell}.npz")
                val_label = frame.loc[val, "label"].to_numpy()
                # The frozen factorial uses complete validation ROC candidate
                # boundaries. Positive scores between two negative boundaries
                # can lower the cut without changing its empirical FPR; this is
                # different from the Phase5 negative-only calibration policy.
                vfpr, vtpr, candidates = roc_curve(val_label, val_score, drop_intermediate=False)
                eligible = np.flatnonzero((vfpr <= .05) & np.isfinite(candidates))
                expected_cut = candidates[eligible[-1]] if len(eligible) else np.nextafter(val_score.max(), np.inf)
                close(cut["logit_threshold"], expected_cut, "Factorial threshold computed on complete validation ROC")
                close(cut["validation_fpr"], (val_score[val_label == 0] >= cut["logit_threshold"]).mean(), "Factorial fixed validation FPR")
                close(cut["validation_tpr"], (val_score[val_label == 1] >= cut["logit_threshold"]).mean(), "Factorial fixed validation TPR")
                require(cut["validation_real_rows"] == int((val_label == 0).sum())
                        and cut["validation_AI_rows"] == int((val_label == 1).sum()), "Factorial validation denominator")
                for value in records.loc[records.seed.eq(seed) & records.backbone.eq(backbone) & records.cell.eq(cell)].to_dict("records"):
                    scope = value["scope"]
                    selected = prediction if scope == "overall" else prediction.loc[prediction.attack.eq(scope)] if scope in ["identity", "jpeg", "blur", "noise", "resize"] else prediction.loc[prediction.label.eq(1) | prediction.collection.eq(scope)]
                    numerical = metric_values(selected.label.to_numpy(), selected[cell+"_logit"].to_numpy(), cut["logit_threshold"])
                    for metric in MEASURES:
                        close(value[metric], numerical[metric], "Factorial test metric")
                    close(value["logit_threshold"], cut["logit_threshold"], "Factorial test threshold unchanged")
                boot[(backbone, seed, cell)] = bootstrap_metrics(prediction.label.to_numpy(), prediction[cell+"_logit"].to_numpy(), cut["logit_threshold"], prediction.source_id.map(source_map).to_numpy(), draws)
        del x
    comparisons = csv(directory / "paired_source_bootstrap.csv")
    require(len(comparisons) == 48, "Factorial contrast interval count")
    overall = records.loc[records.scope.eq("overall")].set_index(["backbone", "seed", "cell"])
    for value in comparisons.to_dict("records"):
        seeds = SEEDS if value["seed"] == "five_seed_mean" else [int(value["seed"])]
        j = MEASURES.index(value["metric"])
        distribution = np.mean([.5*(boot[(value["backbone"], seed, "AA")][:,j] + boot[(value["backbone"], seed, "BB")][:,j] - boot[(value["backbone"], seed, "AB")][:,j] - boot[(value["backbone"], seed, "BA")][:,j]) for seed in seeds], axis=0)
        point = np.mean([.5*(overall.loc[(value["backbone"], seed, "AA"), value["metric"]] + overall.loc[(value["backbone"], seed, "BB"), value["metric"]] - overall.loc[(value["backbone"], seed, "AB"), value["metric"]] - overall.loc[(value["backbone"], seed, "BA"), value["metric"]]) for seed in seeds])
        close(value["estimate"], point, "Prespecified factorial interaction")
        close(value["ci95_low"], np.percentile(distribution, 2.5), "Factorial paired-source lower interval")
        close(value["ci95_high"], np.percentile(distribution, 97.5), "Factorial paired-source upper interval")
    return {"factorial_fits": 40, "factorial_metric_rows": 320, "factorial_paired_intervals": 48,
        "equal_source_slots_and_class_weights": True, "restricted_unmarked_pool_explicit": True}


def check_paired_content():
    """Recompute the frozen caption-pair scores and conditional pair intervals."""
    from phase4_fresh_replication import weighted_auc_draws, rate_draws
    directory = P5 / "paired_content"
    require((directory / "complete.json").exists(), "Paired-content generation/inference/analysis remains unfinished")
    done = read(directory / "complete.json")
    hashes(directory, done["output_sha256"])
    require(done["pairs"] == 200 and done["sources"] == 400 and done["copies"] == 1200,
            "Paired test was shortened or expanded")
    require(not done["test_refitting"] and not done["test_threshold_tuning"], "Paired test tuning")
    protocol, runtime, selection, assets, generation = (read(directory / name) for name in
        ["protocol.json", "runtime_protocol.json", "selection_complete.json", "generation_assets_lock.json", "generation_complete.json"])
    hashes(ROOT, runtime["input_sha256"])
    require(not runtime["generation_or_detector_scores_observed"] and not runtime["test_threshold_tuning"],
            "Paired runtime was not locked prospectively")
    require(runtime["generation_assets_sha256"] == digest(directory / "generation_assets_lock.json"), "Paired asset lock changed")
    require(selection["pairs"] == 200 and selection["all_pairs_split"] == "test"
            and selection["prior_native_id_intersections"] == 0 and selection["exact_or_ahash_overlap_screen_passed"],
            "Paired source screening/split incomplete")
    require(selection["protocol_sha256"] == assets["protocol_sha256"] == digest(directory / "protocol.json"), "Paired protocol changed")
    require(selection["source_pairs_sha256"] == digest(directory / "source_pairs.csv"), "Paired selection changed")
    require(selection["exclusion_manifest_sha256"] == digest(directory / "exclusion_manifest.csv"), "Paired exclusion manifest changed")
    require(assets["adapter_revision"] == "cf2fced511dbe7e26c8d1d397e728fbab875db4b"
            and assets["adapter_sha256"] == "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4", "Paired LCM adapter identity")
    require(protocol["generator"]["num_inference_steps"] == 8 and protocol["generator"]["guidance_scale"] == 1
            and protocol["generator"]["size"] == [512, 512] and not protocol["mark_embedding"], "Paired generation/mark protocol")
    require(not generation["candidate_resampling"] and not generation["scores_observed_during_generation"]
            and generation["pairs"] == 200, "Paired generation selection")
    start = read(directory / "generation_started.json")
    chronological(protocol["frozen_at_utc"], assets["locked_at_utc"], selection["completed_at_utc"],
                  runtime["locked_at_utc"], start["started_at_utc"], generation["completed_at_utc"])
    require(start["runtime_protocol_sha256"] == generation["runtime_protocol_sha256"] == digest(directory / "runtime_protocol.json"),
            "Paired generation runtime lock mismatch")
    pairs = pd.read_csv(directory / "source_pairs.csv", keep_default_na=False, dtype={"public_coco_image_id": str})
    generated = csv(directory / "generated_sources.csv")
    require(len(pairs) == len(generated) == pairs.pair_id.nunique() == generated.pair_id.nunique() == 200,
            "Paired generation identity/duplicate mismatch")
    require(set(pairs.pair_id) == set(generated.pair_id) and pairs.split.eq("test").all(), "Missing/non-test generated pair")
    require(generation["generated_sources_sha256"] == digest(directory / "generated_sources.csv"), "Paired generated-source lock changed")
    generated = generated.set_index("pair_id", verify_integrity=True)
    exclusion = csv(directory / "exclusion_manifest.csv")
    prior_coco = {str(value).zfill(12) for value in exclusion.loc[exclusion.collection.eq("coco"), "public_image_id"] if value != ""}
    require(not set(pairs.public_coco_image_id) & prior_coco, "Paired photograph native ID reused")
    prior_pixels = set(exclusion.raw_pixel_sha256) | set(exclusion.standardized_pixel_sha256)
    prior_ahashes = [int(value) for value in exclusion.ahash if value != ""]
    selected_ahashes = []
    captions_path = ROOT / ".tools/extension-assets/annotations_trainval2017.zip"
    require(digest(captions_path) == selection["caption_archive_sha256"], "Paired caption archive changed")
    with zipfile.ZipFile(captions_path) as archive:
        caption_bytes = archive.read("annotations/captions_val2017.json")
    require(hashlib.sha256(caption_bytes).hexdigest() == selection["captions_val2017_json_sha256"], "Paired caption annotation member changed")
    minimum_captions = {}
    for record in json.loads(caption_bytes)["annotations"]:
        key = str(int(record["image_id"])).zfill(12)
        if key not in minimum_captions or record["id"] < minimum_captions[key]["id"]:
            minimum_captions[key] = record
    for pair in pairs.to_dict("records"):
        expected_seed = int(hashlib.sha256(f"paired-content|20261010|coco|{pair['public_coco_image_id']}".encode()).hexdigest()[:16], 16) % 2**31
        require(int(pair["generation_seed"]) == expected_seed and pair["prompt"] == "Photograph: " + pair["caption"], "Pair prompt/seed selection changed")
        caption = minimum_captions[pair["public_coco_image_id"]]
        require(pair["caption_annotation_id"] == caption["id"] and pair["caption"] == caption["caption"], "Pair minimum-caption rule changed")
        require(pair["raw_pixel_sha256"] not in prior_pixels and pair["standardized_pixel_sha256"] not in prior_pixels,
                "Paired photograph pixel fingerprint reused")
        new_hash = int(pair["prejpeg_ahash"])
        require(all((new_hash ^ previous).bit_count() > 2 for previous in prior_ahashes + selected_ahashes),
                "Paired prior/within-sample aHash overlap")
        selected_ahashes.append(new_hash)
        row = generated.loc[pair["pair_id"]]
        require(row.ai_source_id == pair["ai_source_id"] and int(row.generation_seed) == expected_seed
                and row.runtime_mode in ["resident", "offload"], "Paired generated source/seed/runtime mismatch")
        require(digest(Path(pair["real_path"])) == pair["standardized_file_sha256"], "Changed paired photograph")
        require(digest(Path(row.ai_native_path)) == row.ai_native_file_sha256
                and digest(Path(row.ai_path)) == row.ai_file_sha256, "Changed generated pair image")
    public = read(directory / "public_pair_index_audit.json")
    require(public["all_checks_passed"] and all(public["checks"].values()) and public["counts"]["pairs"] == 200, "Public pair index audit incomplete")
    hashes(directory, public["files_sha256"])
    require(public["source_pairs_sha256"] == digest(directory / "source_pairs.csv"), "Public pair index stale")
    frame, prediction, metrics = (csv(directory / name) for name in ["manifest.csv", "predictions.csv", "metrics.csv"])
    derivation = read(directory / "derivation_complete.json")
    require(derivation["manifest_sha256"] == digest(directory / "manifest.csv") and not derivation["detector_inference_performed"], "Paired derived manifest lock")
    require(len(frame) == frame.image_id.nunique() == 1200 and frame.source_id.nunique() == 400
            and frame.pair_id.nunique() == 200 and frame.mark.eq("unmarked").all(), "Paired manifest/source count/mark status")
    require(frame.groupby("source_id").size().eq(3).all()
            and frame.groupby(["pair_id", "label"]).size().eq(3).all()
            and set(frame.attack) == {"identity", "jpeg", "lowres"}, "Incomplete caption pair processing bundle")
    require(set(frame.loc[frame.label.eq(0), "source_id"]) == set(pairs.real_source_id)
            and set(frame.loc[frame.label.eq(1), "source_id"]) == set(pairs.ai_source_id), "Pair membership in inference manifest changed")
    source_pair = {source: row.pair_id for row in pairs.itertuples() for source in [row.real_source_id, row.ai_source_id]}
    require(frame.source_id.map(source_pair).eq(frame.pair_id).all(), "Caption-pair linkage in inference manifest changed")
    require(np.array_equal(prediction.image_id, frame.image_id), "Paired prediction row order changed")
    for row in frame.to_dict("records"):
        require(digest(Path(row["path"])) == row["output_file_sha256"], "Changed paired derived pixels")
    pair_ids = sorted(frame.pair_id.unique())
    idx = frame.pair_id.map({key: i for i, key in enumerate(pair_ids)}).to_numpy()
    rng = np.random.default_rng(runtime["bootstrap_seed"])
    counts = np.stack([np.bincount(rng.integers(200, size=200), minlength=200)
                       for _ in range(runtime["bootstrap_replicates"])]).astype(np.float32)
    require(counts.shape == (2000, 200), "Paired bootstrap design changed")
    require(len(metrics) == 144 and not metrics.duplicated(["regime", "backbone", "system", "condition", "policy"]).any(),
            "Paired metric coverage/duplicate rows")
    for backbone in ["resnet", "clip"]:
        inference = read(directory / f"{backbone}_inference.json")
        require(inference["images"] == 1200 and max(inference["anchor_relative_l2"]) < .005, "Paired comparability anchors failed")
        require(inference["manifest_sha256"] == digest(directory / "manifest.csv")
                and inference["embedding_sha256"] == digest(directory / f"{backbone}_embeddings.npz"), "Paired embedding lock changed")
        chronological(generation["completed_at_utc"], derivation["completed_at_utc"], inference["started_at_utc"],
                      inference["completed_at_utc"], done["completed_at_utc"])
        with np.load(directory / f"{backbone}_embeddings.npz", allow_pickle=True) as z:
            require(str(z["manifest_sha256"]) == digest(directory / "manifest.csv")
                    and np.array_equal(z["image_id"].astype(str), frame.image_id.to_numpy()), "Paired cache row identity")
            x = z["embedding"].astype(np.float32)
            require(x.shape[0] == 1200 and np.isfinite(x).all(), "Paired invalid feature cache")
        for regime in ["joint", "SD_excluded"]:
            probe, fusion = [ROOT / p for p in runtime["heads"][f"{regime}_{backbone}"]]
            policy_path = P5 / "calibration" / ("thresholds_locked.json" if regime == "joint" else "sd_family_excluded/thresholds_locked.json")
            threshold_lock = read(policy_path)
            entry = threshold_lock["models"][f"20260920_{backbone}" if regime == "joint" else backbone]
            require(digest(probe) == entry["probe_sha256"] and digest(fusion) == entry["fusion_sha256"], "Paired fitted heads changed")
            chronological(threshold_lock["locked_at_utc"], runtime["locked_at_utc"], start["started_at_utc"])
            scores = score_systems(x, frame.active_score.to_numpy(), probe, fusion)
            for system, score in scores.items():
                require(np.allclose(prediction[f"{regime}_{backbone}_{system}_logit"], score, rtol=0, atol=2e-11), "Paired stored prediction mismatch")
                for condition in ["identity", "jpeg", "lowres", "overall"]:
                    mask = np.ones(len(frame), dtype=bool) if condition == "overall" else frame.attack.eq(condition).to_numpy()
                    label, s, pi = frame.label.to_numpy()[mask], score[mask], idx[mask]
                    auc_draws = weighted_auc_draws(label, s, pi, counts)
                    selected = metrics.loc[metrics.regime.eq(regime) & metrics.backbone.eq(backbone) & metrics.system.eq(system) & metrics.condition.eq(condition)]
                    require(set(selected.policy) == set(POLICIES), "Paired policy coverage changed")
                    for row in selected.to_dict("records"):
                        policy = entry["systems"][system][row["policy"]]
                        require(row["comparison"] == policy["comparison"], "Paired comparison/tie rule changed")
                        if policy["threshold_logit"] is None:
                            require(row["threshold_logit"] == "", "Paired infinite threshold changed")
                        else:
                            close(row["threshold_logit"], policy["threshold_logit"], "Paired frozen threshold")
                        close(row["AUC"], roc_auc_score(label, s), "Caption-pair AUC")
                        close(row["AUC_ci95_low"], np.percentile(auc_draws, 2.5), "Pair AUC lower interval")
                        close(row["AUC_ci95_high"], np.percentile(auc_draws, 97.5), "Pair AUC upper interval")
                        accepted = accept(s, policy)
                        for sign, name in [(1, "TPR"), (0, "FPR")]:
                            draws = rate_draws(label, accepted, pi, counts, sign)
                            close(row[name], accepted[label == sign].mean(), "Paired fixed-threshold rate")
                            close(row[name+"_ci95_low"], np.percentile(draws, 2.5), "Paired rate lower interval")
                            close(row[name+"_ci95_high"], np.percentile(draws, 97.5), "Paired rate upper interval")
                        source_event = np.bincount(pi[label == 0], weights=accepted[label == 0], minlength=200) > 0
                        # NumPy 2 preserves float32 during percentile
                        # interpolation; the inference runtime's NumPy 1.26
                        # uses float64 interpolation. Keep the archived draws
                        # unchanged and reproduce its float64 quantile rule.
                        draws = (counts @ source_event / 200).astype(np.float64)
                        close(row["source_any_FPR"], source_event.mean(), "Pair source-any FPR")
                        close(row["source_any_FPR_ci95_low"], np.percentile(draws, 2.5), "Pair source-any lower interval")
                        close(row["source_any_FPR_ci95_high"], np.percentile(draws, 97.5), "Pair source-any upper interval")
                        require(row["pairs"] == 200 and row["copies"] == len(label), "Paired rate denominator")
    summary = read(directory / "summary.json")
    require(summary["pairs"] == 200 and summary["origin_sources"] == 400 and summary["copies"] == 1200
            and not summary["test_refitting"] and not summary["test_threshold_tuning"] and not summary["candidate_resampling"],
            "Paired summary population/tuning status")
    require(len(summary["all_metrics"]) == len(metrics) and len(summary["primary"]) == 4, "Paired summary coverage")
    indexed = metrics.set_index(["regime", "backbone", "system", "condition", "policy"])
    for row in summary["all_metrics"]:
        key = tuple(row[k] for k in ["regime", "backbone", "system", "condition", "policy"])
        saved = indexed.loc[key]
        for name in ["AUC", "TPR", "FPR", "source_any_FPR"]:
            close(row[name], saved[name], "Paired JSON/CSV rate consistency")
    from audit_phase5_generated_overlap import audit as recompute_overlap
    overlap = read(directory / "postgeneration_overlap_audit.json")
    hashes(directory, overlap["input_sha256"])
    require(overlap["audit_completed"] and overlap["post_hoc"] and overlap["all_locked_test_items_retained"]
            and not overlap["filtering_resampling_model_inference_or_score_changes"], "Generated overlap audit scope/status")
    rebuilt_overlap = recompute_overlap()
    require(overlap["comparisons"] == rebuilt_overlap["comparisons"]
            and overlap["generated_fingerprint_inventory_sha256"] == rebuilt_overlap["generated_fingerprint_inventory_sha256"],
            "Generated post hoc pixel/aHash audit mismatch")
    require(overlap["no_detected_overlap_under_these_pixel_and_ahash_rules"]
            == all(value["either_overlap_pairs"] == 0 for value in overlap["comparisons"].values()), "Generated overlap finding misreported")
    # Findings never select or remove a locked test item. In particular, aHash
    # matches are reported even when no decoded pixel fingerprints match.
    from analyze_phase5_overlap_sensitivity import analyze as recompute_sensitivity
    sensitivity = read(directory / "postgeneration_overlap_sensitivity.json")
    hashes(ROOT, sensitivity["input_sha256"])
    require(sensitivity["analysis_complete"] and sensitivity["post_hoc_exploratory_sensitivity"]
            and sensitivity["all_primary_results_unchanged"] and not sensitivity["new_inference_refitting_resampling_or_threshold_tuning"],
            "Post hoc sensitivity scope changed")
    require(sensitivity["primary_pairs_retained_in_main_results"] == 200
            and sensitivity["sensitivity_pairs"] == 200 - sensitivity["flagged_generated_pairs"], "Sensitivity subset/primary denominators")
    rebuilt_sensitivity = recompute_sensitivity()
    require(sensitivity["measurements"] == rebuilt_sensitivity["measurements"]
            and sensitivity["flagged_pair_inventory_sha256"] == rebuilt_sensitivity["flagged_pair_inventory_sha256"]
            and sensitivity["retained_pair_inventory_sha256"] == rebuilt_sensitivity["retained_pair_inventory_sha256"]
            and sensitivity["matched_prior_original_role_counts"] == rebuilt_sensitivity["matched_prior_original_role_counts"],
            "Post hoc fixed-score subset measurements/roles mismatch")
    return {"paired_test_pairs": 200, "paired_test_sources": 400, "paired_test_copies": 1200,
            "paired_metric_rows": 144, "pair_cluster_intervals_recomputed": True,
            "paired_heads_and_thresholds_locked_before_generation": True,
            "one_SD15_family_pipeline_only": True,
            "postgeneration_overlap_audit_recomputed": True,
            "generated_prior_ahash_matching_images": overlap["comparisons"]["prior_acquisition_records"]["generated_images_with_ahash_HD_le2_overlap"],
            "generated_prior_ahash_overlap_pairs": overlap["comparisons"]["prior_acquisition_records"]["ahash_HD_le2_overlap_pairs"],
            "generated_exact_pixel_overlap_pairs": sum(value["exact_pixel_overlap_pairs"] for value in overlap["comparisons"].values()),
            "post_hoc_findings_do_not_filter_locked_test": True,
            "paired_post_hoc_sensitivity_pairs": sensitivity["sensitivity_pairs"],
            "paired_post_hoc_sensitivity_metric_rows": len(sensitivity["measurements"]),
            "paired_post_hoc_fixed_score_sensitivity_recomputed": True}


def check_fair_watermarks():
    """Check native archive flags and independently rebuild equal-target rates.

    No decoder is rerun. Raw continuous margins, their hard information-bit
    decisions, and saved official ECC flags remain distinct pieces of evidence.
    """
    directory = P5 / "fair_watermarks"
    require((directory / "summary.json").exists(), "Fair-watermark encoding/calibration/test decoding remains unfinished")
    protocol, amendment, selection, summary = (read(directory / name) for name in
        ["protocol.json", "protocol_amendment.json", "selection_complete.json", "summary.json"])
    input_paths = {"original_sources": "outputs/conference_clean/source_audit.csv",
        "openimages": "outputs/phase3/controls/openimages_sources.csv",
        "existing_AI_and_COCO": "outputs/phase3/watermarks/selected_sources.csv",
        "existing_pairs": "outputs/phase3/watermarks/selected_pairs.csv"}
    for key, path in input_paths.items():
        require(digest(ROOT / path) == protocol["input_sha256"][key], "Fair-mark locked source input changed")
    sources = csv(directory / "sources.csv")
    require(len(sources) == sources.source_id.nunique() == 1004, "Fair-mark source identity/count")
    expected_counts = {("calibration", "coco"): 200, ("calibration", "openimages"): 200,
        ("test", "coco"): 200, ("test", "openimages"): 200,
        ("test", "stable-diffusion"): 68, ("test", "dalle-3"): 68, ("test", "midjourney"): 68}
    require(sources.groupby(["role", "collection"]).size().to_dict() == expected_counts, "Fair-mark collection counts")
    require(selection["source_manifest_sha256"] == summary["source_manifest_sha256"] == digest(directory / "sources.csv"), "Fair-mark source lock changed")
    original, oi, previous_sources = (csv(ROOT / input_paths[key]) for key in ["original_sources", "openimages", "existing_AI_and_COCO"])
    def fixed_order(frame, arm):
        order = frame.source_id.map(lambda value: hashlib.sha256(f"20261015|{arm}|{value}".encode()).hexdigest())
        return frame.assign(_order=order).sort_values(["_order", "source_id"])
    retained = original.loc[original.retained.astype(str).str.lower().eq("true")]
    coco_cal = fixed_order(retained.loc[retained.label.eq(0) & retained.split.isin(["val", "validation"])], "cal_coco").head(200)
    coco_test = fixed_order(previous_sources.loc[previous_sources.label.eq(0)], "test_coco").head(200)
    oi_order = fixed_order(oi, "openimages")
    for role, collection, wanted in [("calibration", "coco", coco_cal), ("test", "coco", coco_test),
                                     ("calibration", "openimages", oi_order.iloc[:200]), ("test", "openimages", oi_order.iloc[200:400])]:
        require(set(sources.loc[sources.role.eq(role) & sources.collection.eq(collection), "source_id"]) == set(wanted.source_id),
                "Fair-mark source selection differs from frozen ordering")
    require(set(sources.loc[sources.label.eq(1), "source_id"]) == set(previous_sources.loc[previous_sources.label.eq(1), "source_id"]),
            "Fair-mark prior AI test sources changed")
    require(selection["ID_or_pixel_or_ahash_pairs"] == 0 and not selection["new_independent_source_dataset"], "Fair-mark source reuse/duplicate report")
    require(not sources.native_pixel_sha256.duplicated().any(), "Fair-mark exact source pixel overlap")
    ahashes = [int(value) for value in sources.native_ahash]
    require(all((ahashes[i] ^ ahashes[j]).bit_count() > 2 for i in range(len(ahashes)) for j in range(i)),
            "Fair-mark source aHash screen failed")
    require(selection["protocol_sha256"] == summary["protocol_sha256"] == amendment["original_protocol_sha256"]
            == digest(directory / "protocol.json"), "Fair-mark original protocol changed")
    require(summary["amendment_sha256"] == digest(directory / "protocol_amendment.json")
            and amendment["seed_manifest_sha256"] == digest(directory / "rounding_seeds.csv"), "Fair-mark rounding amendment changed")
    require(amendment["decoding_has_not_started"] and amendment["quantization"] == summary["quantization"] == "stochastic_rounding_v1",
            "Fair-mark quantization amendment/chronology")
    require(summary["shared_source_payload_uniforms_for_methods"] and summary["residual_rescaling_outside_official_configuration"]
            and summary["source_reuse"] and summary["calibration_threshold_uncertainty_not_bootstrapped"]
            and not summary["same_observed_test_FPR_guaranteed"], "Fair-mark interpretation changed")
    require(summary["sources"] == 1004 and summary["calibration_photographs"] == summary["test_photographs"] == 400
            and summary["test_AI"] == 204 and summary["test_images"] == 7312 and summary["bootstrap_draws"] == 2000,
            "Fair-mark summary denominator/design")
    require(summary["score_definitions"] == protocol["score"], "Fair-mark score-definition consistency")
    payloads = protocol["payloads"]
    require(len(payloads) == 3 and all(len(p) == 16 for p in payloads), "Fair-mark shared information bits")
    rounding = pd.read_csv(directory / "rounding_seeds.csv", keep_default_na=False,
                           dtype={"seed_uint64": str, "payload": str})
    require(len(rounding) == 612 and not rounding.duplicated(["source_id", "payload"]).any(), "Fair-mark rounding seed coverage")
    for row in rounding.to_dict("records"):
        expected = int(hashlib.sha256(f"20261015|stochastic_rounding_v1|{row['source_id']}|{row['payload']}".encode()).hexdigest()[:16], 16)
        require(int(row["seed_uint64"]) == expected and row["payload"] in payloads, "Fair-mark rounding seed selection changed")
    cpu, encoding, derivation, thresholds = (read(directory / name) for name in
        ["cpu_preparation_complete.json", "encoding_complete.json", "derivation_complete.json", "thresholds.json"])
    require(cpu["source_count"] == 1004 and cpu["covers"] == 1616 and not cpu["decoder_run"], "Fair-mark CPU preparation coverage")
    for name, key in [("base_sources.csv", "base_manifest_sha256"), ("cpu_covers.csv", "covers_sha256"), ("sources.csv", "sources_sha256")]:
        require(digest(directory / name) == cpu[key], "Fair-mark CPU preparation lock changed")
    require(encoding["covers"] == 2228 and encoding["cover_manifest_sha256"] == digest(directory / "covers.csv")
            and not encoding["decoder_run"], "Fair-mark encoding lock/coverage")
    require(derivation["images"] == summary["images"] == 8912 and derivation["counts_per_role"] == {"test": 7312, "calibration": 1600},
            "Fair-mark derived image coverage")
    require(derivation["manifest_sha256"] == thresholds["manifest_sha256"] == digest(directory / "manifest.csv"), "Fair-mark derivative/threshold manifest lock")
    chronological(protocol["frozen_at_utc"], selection["completed_at_utc"], amendment["locked_at_utc"], cpu["completed_at_utc"],
                  encoding["started_at_utc"], encoding["completed_at_utc"], derivation["completed_at_utc"])
    manifest, covers = csv(directory / "manifest.csv"), csv(directory / "covers.csv")
    require(len(manifest) == manifest.image_id.nunique() == 8912 and len(covers) == covers.cover_id.nunique() == 2228,
            "Fair-mark cover/image duplicates")
    require(manifest.groupby("cover_id").size().eq(4).all() and set(manifest.condition) == set(protocol["conditions"]),
            "Fair-mark processing bundles incomplete")
    require(manifest.groupby(["source_id", "watermark_method", "payload"]).size().eq(4).all(), "Fair-mark payload bundle duplication")
    source_info = sources.set_index("source_id")
    for column in ["role", "collection", "label"]:
        require(manifest.source_id.map(source_info[column]).eq(manifest[column]).all(), "Fair-mark derivative source role changed")
    marked_covers = covers.loc[covers.watermark_method.isin(["dwt_dct", "trustmark"])]
    require(len(marked_covers) == 1224 and (marked_covers.embedding_PSNR_dB - 40).abs().le(.1 + 1e-10).all(), "Fair-mark matched distortion target")
    require(marked_covers.groupby(["source_id", "watermark_method"]).size().eq(3).all(), "Fair-mark three-payload embedding coverage")
    for row in manifest.to_dict("records"):
        require(digest(Path(row["path"])) == row["output_file_sha256"], "Changed fair-mark derivative pixels")
    methods = ["dwt_dct", "trustmark"]
    scores_by_role = {}
    for role, count in [("calibration", 1600), ("test", 7312)]:
        inference = read(directory / f"{role}_inference_complete.json")
        require(inference["role"] == role and inference["image_count"] == count and not inference["additional_passes"], "Fair-mark repeated/incomplete decoding")
        require(inference["manifest_sha256"] == digest(directory / "manifest.csv")
                and inference["protocol_sha256"] == digest(directory / "protocol.json")
                and inference["amendment_sha256"] == digest(directory / "protocol_amendment.json"), "Fair-mark decoder lock changed")
        require(inference["raw_sha256"] == digest(directory / f"{role}_raw.npz"), "Fair-mark raw decoder archive changed")
        require(inference["checkpoints"] == encoding["official_checkpoint_provenance"]["checkpoints"], "Fair-mark encoder/decoder checkpoints differ")
        require(len(inference["anchors"]) > 0 and all(r["matches_public_API"] and r["matches_archived_bits_ECC"] for r in inference["anchors"]),
                "Fair-mark archived decoder/API anchors failed")
        chronological(derivation["completed_at_utc"], inference["started_at_utc"], inference["completed_at_utc"], summary["completed_at_utc"])
        if role == "test":
            require(inference["threshold_sha256_before_test"] == digest(directory / "thresholds.json"), "Fair-mark thresholds changed after test start")
            chronological(thresholds["locked_at_utc"], inference["started_at_utc"])
        else:
            chronological(inference["completed_at_utc"], thresholds["locked_at_utc"])
        scores = csv(directory / f"{role}_scores.csv")
        require(len(scores) == count * 6 and not scores.duplicated(["image_id", "method", "query_payload"]).any(), "Fair-mark score/query coverage")
        with np.load(directory / f"{role}_raw.npz", allow_pickle=False) as raw:
            ids = raw["image_id"].astype(str)
            subset = manifest.loc[manifest.role.eq(role)].sort_values("image_id").reset_index(drop=True)
            require(np.array_equal(ids, subset.image_id.to_numpy()), "Fair-mark raw decoder row identity")
            require(raw["dwt_raw"].shape == (count, 16) and raw["trust_raw"].shape == (count, 100)
                    and np.isfinite(raw["dwt_raw"]).all() and np.isfinite(raw["trust_raw"]).all(), "Fair-mark raw decoder shape/finiteness")
            lookup = scores.set_index(["image_id", "method", "query_payload"])
            for method in methods:
                margins = raw["dwt_raw"] / 12 if method == "dwt_dct" else raw["trust_raw"][:, :16]
                hard = raw["dwt_raw"] >= 0 if method == "dwt_dct" else raw["trust_raw"][:, :16] > 0
                for payload in payloads:
                    expected = np.array(list(payload), dtype=np.int8)
                    reconstructed = np.mean(margins.astype(np.float64) * (2 * expected - 1), axis=1)
                    agreement = np.sum(hard == expected, axis=1)
                    native = agreement >= 13 if method == "dwt_dct" else raw["trust_ECC_valid"] & (raw["trust_schema"] == 1) & (raw["trust_message"] == payload + "0" * 45)
                    selected = lookup.loc[[(iid, method, payload) for iid in ids]]
                    require(np.allclose(selected.score, reconstructed, rtol=0, atol=2e-11)
                            and np.array_equal(selected.information_bits_agree.to_numpy(), agreement)
                            and np.array_equal(selected.native_verifies.to_numpy().astype(bool), native), "Fair-mark margin/bit/native-rule computation mismatch")
                    for field in ["source_id", "role", "collection", "label", "watermark_method", "condition"]:
                        require(np.array_equal(selected[field].to_numpy(), subset[field].to_numpy()), "Fair-mark score source/condition metadata mismatch")
                    require(np.array_equal(selected.embedded_payload.to_numpy(), subset.payload.to_numpy()), "Fair-mark embedded-payload metadata mismatch")
        scores_by_role[role] = scores
    require(thresholds["whole_ties"] and thresholds["test_decoding_has_not_started"]
            and thresholds["nominal_FPR"] == .05 and len(thresholds["records"]) == 4, "Fair-mark four whole-tie target thresholds")
    require(thresholds["calibration_scores_sha256"] == digest(directory / "calibration_scores.csv")
            and thresholds["calibration_raw_sha256"] == digest(directory / "calibration_raw.npz"), "Fair-mark calibration archive lock")
    require(summary["thresholds_sha256"] == digest(directory / "thresholds.json")
            and summary["test_raw_sha256"] == digest(directory / "test_raw.npz"), "Fair-mark summary input lock")
    require(summary["thresholds"] == thresholds["records"], "Fair-mark JSON threshold consistency")
    cuts = {(row["method"], row["query_rule"]): row for row in thresholds["records"]}
    for method in methods:
        part = scores_by_role["calibration"].loc[lambda x: x.method.eq(method)]
        require(part.source_id.nunique() == 400 and part.label.eq(0).all(), "Fair-mark calibration origin/source population")
        for query in ["default", "any3"]:
            values = part.loc[part.query_payload.eq(payloads[0]), "score"].to_numpy() if query == "default" else part.groupby("image_id").score.max().to_numpy()
            record = cuts[(method, query)]
            close(record["threshold"], fpr_cut(values), "Fair-mark independent whole-tie calibration cut")
            accepted = values >= record["threshold"]
            close(record["empirical_FPR"], accepted.mean(), "Fair-mark calibration FPR")
            require(record["accepted_images"] == int(accepted.sum()) and record["calibration_images"] == len(values) == 1600
                    and record["empirical_FPR"] <= .05, "Fair-mark calibration acceptance count")
    measurements, boot = {}, {}
    with np.load(directory / "bootstrap_rates.npz", allow_pickle=False) as saved_boot:
        require(len(summary["measurements"]) == len(saved_boot.files) == 440, "Fair-mark rate coverage")
        for number, row in enumerate(summary["measurements"], 1):
            method, query = row["method"], row["query_rule"]
            part = scores_by_role["test"].loc[lambda x: x.method.eq(method)]
            if query == "default":
                part = part.loc[part.query_payload.eq(payloads[0])].copy()
            else:
                keys = ["image_id", "source_id", "collection", "label", "watermark_method", "embedded_payload", "condition"]
                part = part.groupby(keys, as_index=False).agg(score=("score", "max"), native_verifies=("native_verifies", "max"))
            if row["condition"] != "overall":
                part = part.loc[part.condition.eq(row["condition"])]
            if row["population"] == "photographs":
                part = part.loc[part.label.eq(0) & part.watermark_method.eq("unmarked")]
            elif row["population"] == "unmarked_AI":
                part = part.loc[part.label.eq(1) & part.watermark_method.eq("unmarked")]
            else:
                require(row["population"] == "marked_AI", "Fair-mark unexpected rate population")
                part = part.loc[part.label.eq(1) & part.watermark_method.eq(method)
                                & (part.embedded_payload.eq(payloads[0]) if query == "default" else True)]
            if row["collection"] != "all":
                part = part.loc[part.collection.eq(row["collection"])]
            threshold = cuts[(method, query)]["threshold"]
            event = part.score.to_numpy() >= threshold if row["criterion"] == "calibrated_margin" else part.native_verifies.to_numpy().astype(bool)
            if row["criterion"] == "calibrated_margin":
                close(row["threshold"], threshold, "Fair-mark frozen decision threshold")
            else:
                require(row["criterion"] == "native_rule" and row["threshold"] is None, "Fair-mark secondary native-rule labeling")
            events = part.assign(event=event.astype(int))
            grouped = events.groupby(["source_id", "collection"]).event.agg(["mean", "min", "max", "size"])
            require(row["images"] == len(events) and row["sources"] == len(grouped)
                    and row["accepted_images"] == int(event.sum()), "Fair-mark rate denominator/count")
            close(row["per_image_rate"], event.mean(), "Fair-mark per-image rate")
            close(row["source_mean_rate"], grouped["mean"].mean(), "Fair-mark equal-source mean rate")
            rng = np.random.default_rng(20261016)
            draws = np.zeros(2000)
            collections = grouped.index.get_level_values("collection").to_numpy()
            for collection in sorted(set(collections)):
                values = grouped["mean"].to_numpy()[collections == collection]
                draws += values[rng.integers(0, len(values), (2000, len(values)))].sum(axis=1)
            draws /= len(grouped)
            require(np.allclose(saved_boot[f"record_{number:04d}"], draws, rtol=0, atol=2e-11), "Fair-mark source bootstrap archive mismatch")
            for actual, expected in zip(row["source_bootstrap_ci95"], np.quantile(draws, [.025, .975])):
                close(actual, expected, "Fair-mark source-cluster rate interval")
            for event_name, column in [("ever", "max"), ("all", "min")]:
                n, count = len(grouped), int(grouped[column].sum())
                require(row[f"source_{event_name}_count"] == count, "Fair-mark source event count")
                close(row[f"source_{event_name}_rate"], count/n, "Fair-mark source event rate")
                expected = [beta.ppf(.025, count, n-count+1) if count else 0., beta.ppf(.975, count+1, n-count) if count<n else 1.]
                for actual, endpoint in zip(row[f"source_{event_name}_exact_ci95"], expected):
                    close(actual, endpoint, "Fair-mark exact source event interval")
            key = tuple(row[field] for field in ["method", "query_rule", "condition", "population", "collection", "criterion"])
            require(key not in measurements, "Fair-mark duplicate rate record")
            measurements[key], boot[key] = row, draws
    require(len(summary["paired_method_differences"]) == 30, "Fair-mark paired method contrast coverage")
    for row in summary["paired_method_differences"]:
        tail = (row["query_rule"], row["condition"], row["population"], "all", "calibrated_margin")
        a, b = measurements[("dwt_dct",) + tail], measurements[("trustmark",) + tail]
        close(row["TrustMark_minus_DWT_source_mean_rate"], b["source_mean_rate"] - a["source_mean_rate"], "Fair-mark paired method point difference")
        distribution = boot[("trustmark",) + tail] - boot[("dwt_dct",) + tail]
        for actual, expected in zip(row["paired_source_ci95"], np.quantile(distribution, [.025, .975])):
            close(actual, expected, "Fair-mark paired method interval")
        require(row["conditional_on_frozen_calibration"], "Fair-mark conditional uncertainty labeling")
    require(len(summary["quality"]) == 8, "Fair-mark quality summary coverage")
    for row in summary["quality"]:
        part = manifest.loc[manifest.watermark_method.eq(row["method"]) & manifest.condition.eq(row["condition"])]
        require(row["sources"] == part.source_id.nunique() == 204 and row["images"] == len(part) == 612, "Fair-mark quality denominator")
        for column in ["embedding_PSNR_dB", "embedding_SSIM", "base_encoder_PSNR_dB", "residual_gain", "PSNR_vs_original_dB", "SSIM_vs_original", "PSNR_vs_attack_input_dB", "SSIM_vs_attack_input"]:
            for operation in ["mean", "min", "max"]:
                close(row[column][operation], getattr(part[column], operation)(), "Fair-mark quality aggregate")
    rates = csv(directory / "rates.csv")
    contrasts = csv(directory / "paired_method_differences.csv")
    require(len(rates) == 440 and len(contrasts) == 30, "Fair-mark CSV rate/contrast coverage")
    for row in rates.to_dict("records"):
        key = tuple(row[field] for field in ["method", "query_rule", "condition", "population", "collection", "criterion"])
        saved = measurements[key]
        for field in ["per_image_rate", "source_mean_rate", "source_ever_rate", "source_all_rate"]:
            close(row[field], saved[field], "Fair-mark JSON/CSV rate consistency")
    contrasts = contrasts.set_index(["query_rule", "condition", "population"])
    for row in summary["paired_method_differences"]:
        saved = contrasts.loc[(row["query_rule"], row["condition"], row["population"])]
        close(row["TrustMark_minus_DWT_source_mean_rate"], saved["TrustMark_minus_DWT_source_mean_rate"],
              "Fair-mark JSON/CSV method contrast consistency")
    return {"fair_watermark_sources": 1004, "fair_watermark_images": 8912, "fair_watermark_raw_decoder_rows": 17824,
            "fair_watermark_score_rows": 53472, "fair_watermark_rate_records": 440,
            "fair_watermark_margin_cuts_and_source_intervals_recomputed": True,
            "same_nominal_calibration_FPR_not_equal_test_FPR": True,
            "quality_aggregates_recomputed_from_manifest": True,
            "official_ECC_flags_verified_as_archived_not_decoded_again": True,
            "pooled_binomial_endpoints_are_formula_checks_not_exact_mixture_coverage": True,
            "within_collection_binomial_coverage_requires_iid_source_events": True}


def check_phase5():
    """Return evidence integrity report; raise if any required stage is unfinished."""
    required = [P5 / "calibration/complete.json", P5 / "calibration/sd_family_excluded/complete.json",
                P5 / "matching_factorial/complete.json", P5 / "paired_content/complete.json", P5 / "fair_watermarks/summary.json"]
    missing = [p.relative_to(ROOT).as_posix() for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Phase5 is unfinished: " + ", ".join(missing))
    from check_conference_submission import check_phase4
    report = {**check_phase4(), **check_calibration(), **check_matching_factorial(),
              **check_paired_content(), **check_fair_watermarks()}
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["all_required_Phase5_evidence_complete"] = True
    return report


def check_manuscript(pages=5, references=24, visual_reviewed=False):
    from pypdf import PdfReader
    from check_conference_submission import resource_fonts
    tex_path, pdf_path = ROOT / "manuscript/main.tex", ROOT / "manuscript/main.pdf"
    tex = tex_path.read_text(encoding="utf-8")
    require(r"\IEEEauthorblockN{Anonymous Authors}" in tex and "C:" not in tex and "local_user" not in tex.lower(), "Manuscript anonymization")
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    citations = list(dict.fromkeys(k.strip() for values in re.findall(r"\\cite\{([^}]+)\}", tex) for k in values.split(",")))
    require(len(bibliography) == references and bibliography == citations, "Reference count/order/citation coverage")
    abstract = tex.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    require(len(re.findall(r"\d+(?:\.\d+)?", abstract)) <= 5, "Abstract numerical overload")
    require("OpenAI Codex assisted" in tex, "Authorship assistance disclosure missing")
    log = (ROOT / "manuscript/main.log").read_text(encoding="utf-8")
    require("Overfull" not in log and not re.search(r"(Citation|Reference).*undefined", log), "Unresolved LaTeX/overflow diagnostics")
    require(pdf_path.stat().st_mtime >= tex_path.stat().st_mtime, "Saved PDF is older than current source")
    reader = PdfReader(pdf_path)
    require(len(reader.pages) == pages and not reader.is_encrypted, "PDF length/encryption")
    fonts = set()
    for page in reader.pages:
        require(tuple(map(float, page.mediabox)) == (0., 0., 612., 792.), "PDF paper size")
        for font in resource_fonts(page["/Resources"]):
            if "/DescendantFonts" in font:
                font = font["/DescendantFonts"][0].get_object()
            descriptor = font.get("/FontDescriptor")
            require(font.get("/Subtype") == "/Type3" or (descriptor is not None and any(key in descriptor.get_object() for key in ["/FontFile", "/FontFile2", "/FontFile3"])), "PDF font not embedded")
            fonts.add(str(font.get("/BaseFont", "")))
    require(any("NimbusRom" in name or "Times" in name for name in fonts), "Conference body font")
    return {"pdf_pages": pages, "references": references, "tables": tex.count(r"\begin{table}") + tex.count(r"\begin{table*}"),
        "figures": tex.count(r"\begin{figure}") + tex.count(r"\begin{figure*}"),
        "fonts_embedded": True, "visual_review_confirmed_by_operator": bool(visual_reviewed),
        "manuscript_sha256": digest(tex_path), "PDF_sha256": digest(pdf_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=["all", "calibration", "matching", "paired", "watermarks"], default="all")
    ap.add_argument("--manuscript", action="store_true")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--references", type=int, default=24)
    ap.add_argument("--visual-reviewed", action="store_true")
    args = ap.parse_args()
    with threadpool_limits(limits=2):
        run = {"all": check_phase5, "calibration": check_calibration, "matching": check_matching_factorial,
               "paired": check_paired_content, "watermarks": check_fair_watermarks}[args.component]
        report = run()
        if args.manuscript:
            report.update(check_manuscript(args.pages, args.references, args.visual_reviewed))
    report["scope"] = args.component
    target = P5 / ("submission_check.json" if args.component == "all" else f"{args.component}_check.json")
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
