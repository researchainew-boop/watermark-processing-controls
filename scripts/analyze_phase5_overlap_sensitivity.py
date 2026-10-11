"""Post hoc identity-only sensitivity using saved, unchanged paired scores.

All 200 pairs remain the primary test. This diagnostic excludes the complete
pairs of generated images flagged by the separate coarse aHash audit, without
generation, inference, refitting, threshold changes, or private data export.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from audit_phase5_generated_overlap import ROOT, OUT, ahash, digest, require, records
from phase4_fresh_replication import weighted_auc_draws, rate_draws

BOOTSTRAP_SEED = 20261021
POLICIES = ["per_image_core13", "conformal_core13", "conformal_all16"]


def analyze():
    started = datetime.now(timezone.utc).isoformat()
    paths = {"overlap_audit": OUT / "postgeneration_overlap_audit.json",
        "predictions": OUT / "predictions.csv", "source_pairs": OUT / "source_pairs.csv",
        "generated_sources": OUT / "generated_sources.csv", "exclusion_manifest": OUT / "exclusion_manifest.csv",
        "runtime_protocol": OUT / "runtime_protocol.json", "summary": OUT / "summary.json", "complete": OUT / "complete.json",
        "joint_thresholds": ROOT / "outputs/phase5/calibration/thresholds_locked.json",
        "SD_excluded_thresholds": ROOT / "outputs/phase5/calibration/sd_family_excluded/thresholds_locked.json",
        "original_source_roles": ROOT / "outputs/conference_clean/source_audit.csv"}
    bound = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths.values()}
    overlap = json.loads(paths["overlap_audit"].read_text(encoding="utf-8"))
    require(overlap["audit_completed"] and overlap["all_locked_test_items_retained"], "Completed read-only overlap audit required")
    for name, value in overlap["input_sha256"].items():
        require(digest(OUT / name) == value, "Changed overlap audit input: " + name)
    runtime = json.loads(paths["runtime_protocol"].read_text(encoding="utf-8"))
    for name, value in runtime["input_sha256"].items():
        require(digest(ROOT / name) == value, "Changed frozen runtime/head input: " + name)
    for model_paths in runtime["heads"].values():
        for name in model_paths:
            bound[name] = digest(ROOT / name)
    prior = records("exclusion_manifest.csv")
    prior_hashes = [int(row["ahash"]) for row in prior]
    generated = records("generated_sources.csv")
    flagged_pairs, matched_references = set(), set()
    match_pairs = 0
    for row in generated:
        require(digest(row["ai_native_path"]) == row["ai_native_file_sha256"], "Changed generated image before sensitivity")
        native = cv2.imread(row["ai_native_path"], cv2.IMREAD_COLOR)
        require(native is not None and native.shape == (512, 512, 3), "Generated image decode/shape error")
        fingerprint = ahash(cv2.resize(native, (256, 256), interpolation=cv2.INTER_AREA))
        for number, reference in enumerate(prior_hashes):
            if (fingerprint ^ reference).bit_count() <= 2:
                flagged_pairs.add(row["pair_id"])
                matched_references.add(number)
                match_pairs += 1
    observed = overlap["comparisons"]["prior_acquisition_records"]
    require(len(flagged_pairs) == observed["generated_images_with_ahash_HD_le2_overlap"]
            and match_pairs == observed["ahash_HD_le2_overlap_pairs"], "Sensitivity aHash flags differ from aggregate audit")
    frame = pd.read_csv(paths["predictions"], keep_default_na=False)
    require(frame.pair_id.nunique() == 200 and len(frame) == 1200, "Primary paired population changed")
    kept_pairs = sorted(set(frame.pair_id) - flagged_pairs)
    selected = frame.loc[frame.attack.eq("identity") & frame.pair_id.isin(kept_pairs)].copy()
    n = len(kept_pairs)
    require(n == 200 - len(flagged_pairs) and 1 < n and len(selected) == 2*n
            and selected.groupby(["pair_id", "label"]).size().eq(1).all(), "Sensitivity must retain complete photograph/AI pairs")
    selected = selected.reset_index(drop=True)
    pair_index = selected.pair_id.map({pair: i for i, pair in enumerate(kept_pairs)}).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.stack([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(2000)]).astype(np.float32)
    label = selected.label.to_numpy()
    primary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    full = {(row["regime"], row["backbone"], row["policy"]): row for row in primary["all_metrics"]
            if row["system"] == "probe" and row["condition"] == "identity"}
    rates = []
    for regime in ["joint", "SD_excluded"]:
        locked = json.loads(paths[f"{regime}_thresholds"].read_text(encoding="utf-8"))
        for backbone in ["resnet", "clip"]:
            score = selected[f"{regime}_{backbone}_probe_logit"].to_numpy(dtype=np.float64)
            auc_draws = weighted_auc_draws(label, score, pair_index, draws)
            auc = float(roc_auc_score(label, score))
            model = locked["models"][f"20260920_{backbone}" if regime == "joint" else backbone]
            for policy_name in POLICIES:
                policy = model["systems"]["probe"][policy_name]
                threshold = policy["threshold_logit"]
                accepted = np.zeros(len(score), dtype=bool) if threshold is None else (
                    score > threshold if policy["comparison"] == ">" else score >= threshold)
                tpr_draws = rate_draws(label, accepted, pair_index, draws, 1)
                fpr_draws = rate_draws(label, accepted, pair_index, draws, 0)
                tpr, fpr = float(accepted[label == 1].mean()), float(accepted[label == 0].mean())
                original = full[regime, backbone, policy_name]
                rates.append({"regime": regime, "backbone": backbone, "system": "probe", "condition": "identity",
                    "policy": policy_name, "threshold_logit": threshold, "comparison": policy["comparison"],
                    "pairs": n, "AI_images": n, "photographs": n, "AUC": auc,
                    "AUC_pair_bootstrap_ci95": np.quantile(auc_draws, [.025, .975]).tolist(),
                    "TPR": tpr, "TPR_pair_bootstrap_ci95": np.quantile(tpr_draws, [.025, .975]).tolist(),
                    "FPR": fpr, "FPR_pair_bootstrap_ci95": np.quantile(fpr_draws, [.025, .975]).tolist(),
                    "accepted_AI": int(accepted[label == 1].sum()), "accepted_photographs": int(accepted[label == 0].sum()),
                    "primary_200_pair_AUC": original["AUC"], "primary_200_pair_TPR": original["TPR"], "primary_200_pair_FPR": original["FPR"],
                    "sensitivity_minus_primary_AUC": auc - original["AUC"],
                    "sensitivity_minus_primary_TPR": tpr - original["TPR"], "sensitivity_minus_primary_FPR": fpr - original["FPR"],
                    "zero_event_bootstrap_is_not_zero_population_risk": bool(not accepted[label == 0].any())})
    with paths["original_source_roles"].open(encoding="utf-8-sig", newline="") as stream:
        original_roles = {row["source_id"]: row for row in csv.DictReader(stream)}
    roles, collections = Counter(), Counter()
    for number in matched_references:
        row = prior[number]
        original = original_roles.get(row["source_id"])
        role = original["split"] if original else "other_prior_sample_or_collection"
        if role in ["val", "validation"]:
            role = "validation"
        require(role in ["train", "validation", "test", "other_prior_sample_or_collection"], "Unrecognized original reference role; do not infer it")
        roles[role] += 1
        collections[row["collection"]] += 1
    for name, value in bound.items():
        require(digest(ROOT / name) == value, "Frozen artifact changed during read-only sensitivity: " + name)
    return {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "started_at_utc": started,
        "analysis_complete": True, "post_hoc_exploratory_sensitivity": True,
        "primary_pairs_retained_in_main_results": 200, "flagged_generated_pairs": len(flagged_pairs),
        "sensitivity_pairs": n, "sensitivity_identity_images": 2*n,
        "selection": "For this supplementary diagnostic only, omit complete caption/photo pairs whose generated image has archived-prior aHash HD<=2; retain all primary200 test items unchanged",
        "criterion_used_for_flags": "Coarse preJPEG256 8x8 mean aHash only; no detector score or label-based flag selection; zero exact pixel matches reported separately",
        "flagged_pair_inventory_sha256": hashlib.sha256(json.dumps(sorted(flagged_pairs), separators=(",", ":")).encode()).hexdigest(),
        "retained_pair_inventory_sha256": hashlib.sha256(json.dumps(kept_pairs, separators=(",", ":")).encode()).hexdigest(),
        "matched_prior_records": len(matched_references), "prior_ahash_match_pairs": match_pairs,
        "matched_prior_original_role_counts": dict(sorted(roles.items())),
        "matched_prior_collection_counts": dict(sorted(collections.items())),
        "role_lookup_basis": "Existing original-source-audit split values only; other samples/collections are not assigned a guessed original split",
        "bootstrap": {"replicates": 2000, "seed": BOOTSTRAP_SEED, "unit": "Complete retained caption/photo pair",
            "interpretation": "Percentile95% intervals conditional on fixed heads, thresholds and post hoc retained subset; excludes fitting/calibration/flag-selection uncertainty"},
        "input_sha256": bound, "script_sha256": digest(__file__), "measurements": rates,
        "new_inference_refitting_resampling_or_threshold_tuning": False,
        "all_primary_results_unchanged": True,
        "limitations": ["This sensitivity was requested after observing the locked test and postgeneration audit; it is not a new untouched test or a prospective exclusion rule.",
            "Average-hash proximity is not evidence of duplicate scenes or semantic dependence.",
            "The omitted pairs include the corresponding photograph as well as the flagged generated image; denominators remain content-paired.",
            "AUC/rate differences from the full200 test are descriptive subset shifts, not model improvements or causal effects.",
            "Zero-event bootstrap intervals are degenerate and do not establish zero future risk."],
        "public_scope": "Aggregate measurements, counts and fingerprints only; no image bytes, private captions/prompts, source/pair IDs, local paths, embeddings or per-image scores"}


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        result = analyze()
    target = OUT / "postgeneration_overlap_sensitivity.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        require(old["input_sha256"] == result["input_sha256"] and old["measurements"] == result["measurements"],
                "Refusing to overwrite sensitivity of changed inputs or different measurements")
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"post_hoc": True, "primary_pairs": 200, "sensitivity_pairs": result["sensitivity_pairs"],
        "matched_prior_original_roles": result["matched_prior_original_role_counts"],
        "identity_auc": [{"regime": row["regime"], "backbone": row["backbone"], "AUC": row["AUC"],
                          "ci95": row["AUC_pair_bootstrap_ci95"]} for row in result["measurements"] if row["policy"] == "per_image_core13"]}, indent=2))
