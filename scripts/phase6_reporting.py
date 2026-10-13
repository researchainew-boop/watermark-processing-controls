"""Receipt-verified presentation of frozen Phase 6 evaluations.

No feature extraction, fitted-score computation, threshold selection or GPU
work is performed. Primary intervals are copied from the frozen evaluator.
Extra subgroup intervals reuse its 2,000 collection-stratified source-draw
logic, with derivatives/payloads/seeds kept together. They are descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
LEARNER = ROOT / "outputs/phase6/fusion_v2"
OUT = ROOT / "outputs/phase6/presentation"
ASSETS = ROOT / "manuscript/assets"
COMMON = ("identity", "jpeg50", "crop", "lowres32")
PRIMARY = "trustmark/resnet"
SEED = 20261011
BOOTSTRAPS = 2000
BASELINES = ("probe", "static_fusion", "hard_13_gate", "simple_confidence",
             "old_capped_positive", "old_reliability_mixture", "unconstrained_full_logistic")
ABLATIONS = ("primary", "no_null", "no_pair", "no_stability", "no_full_match")
CURVES = ("probe", "old_capped_positive", "hard_13_gate", "unconstrained_full_logistic", "primary")
LABELS = {"probe": "Probe", "static_fusion": "Static fusion", "hard_13_gate": "Hard 13/16 gate",
          "simple_confidence": "Simple confidence", "old_capped_positive": "Capped positive",
          "old_reliability_mixture": "Reliability mixture", "unconstrained_full_logistic": "Full logistic",
          "primary": "Paired correction", "no_null": "No null penalty", "no_pair": "No pair penalty",
          "no_stability": "No stability features", "no_full_match": "No full-decoding features"}
LABELS.update({"full_message_union":"Full-message union", "tuned_confidence":"Tuned confidence"})
COLORS = {"probe": "#303030", "old_capped_positive": "#d55e00", "hard_13_gate": "#0072b2", "unconstrained_full_logistic": "#009e73", "primary": "#6d3190"}
STYLES = {"probe": "-", "old_capped_positive": "--", "hard_13_gate": "-.", "unconstrained_full_logistic": ":", "primary": "-"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_records_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path,index=False)
    else:
        pd.DataFrame(columns=["evaluation","decoder","backbone","scope","scope_value","system","images","sources","AI_sources","real_sources"]).to_csv(path,index=False)


def verify_outputs(folder, receipt_name):
    receipt_path = folder / receipt_name
    if not receipt_path.exists():
        raise RuntimeError(f"Frozen completion receipt is required: {receipt_path}")
    receipt = read(receipt_path)
    for filename, expected in receipt.get("outputs_sha256", {}).items():
        if digest(folder / filename) != expected:
            raise ValueError(f"Frozen output checksum changed: {folder / filename}")
    return receipt


def locked_inputs(learner):
    fit = verify_outputs(learner, "fit_complete.json")
    protocol = read(learner / "protocol.json")
    if protocol["bootstrap_samples"] != BOOTSTRAPS or protocol["bootstrap_seed"] != SEED:
        raise ValueError("Reporting assumes the frozen 2,000-draw Phase 6 protocol")
    if tuple(protocol["primary_common_conditions"]) != COMMON or protocol["confirmatory_primary"]["system"] != PRIMARY:
        raise ValueError("Unexpected frozen primary family or system")
    return fit, protocol, read(learner / "selection.json")


def evaluation(learner, name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Evaluation names must be plain frozen evaluation directory names")
    folder = learner / "evaluations" / name
    receipt = verify_outputs(folder, "complete.json")
    if receipt["fit_complete_sha256"] != digest(learner / "fit_complete.json"):
        raise ValueError("Evaluation belongs to different fitted models")
    if receipt["calibration_complete_sha256"] != digest(learner / "calibration_complete.json"):
        raise ValueError("Evaluation belongs to different calibration")
    results = read(folder / "results.json")
    if results["bootstrap_samples"] != BOOTSTRAPS or not results["no_test_tuning"]:
        raise ValueError("Unexpected evaluator intervals or test tuning")
    frame = pd.read_csv(folder / "predictions.csv", keep_default_na=False,
        dtype={"image_id": str, "source_id": str, "embedded_payload": str,
               "label": np.int8, "decision": np.int8, "score": np.float64})
    required = {"image_id", "source_id", "collection", "label", "watermark_method", "embedded_payload",
                "condition", "detector", "backbone", "system", "score", "decision"}
    if not required.issubset(frame) or not np.isfinite(frame.score).all():
        raise ValueError("Incomplete or nonfinite saved prediction schema")
    if frame.duplicated(["detector", "backbone", "system", "image_id"]).any():
        raise ValueError("Repeated saved method/image prediction")
    if not set(frame.decision).issubset({0, 1}) or not set(frame.label).issubset({0, 1}):
        raise ValueError("Expected binary decisions and origin labels")
    if frame.source_id.nunique() != receipt["source_count"]:
        raise ValueError("Prediction source count differs from completion receipt")
    return folder, receipt, results, frame


def flatten_metric(row, record, metric):
    point = record["source_bootstrap"].get(metric)
    if point is None:
        row[metric] = None
        return
    row[metric] = point["estimate"]
    row[metric + "_ci_low"], row[metric + "_ci_high"] = point["ci95"]
    row[metric + "_sources"] = point["sources"]
    minus = point["minus_probe"]
    row[metric + "_minus_probe"] = minus["estimate"]
    row[metric + "_minus_probe_ci_low"], row[metric + "_minus_probe_ci_high"] = minus["ci95"]
    row[metric + "_minus_probe_one_sided_lower95"] = minus["one_sided_lower95"]


def aggregate(results, selection, evaluation_name, scope_name="primary_common"):
    rows, shares = [], []
    for key, scopes in results["models"].items():
        detector, backbone = key.split("/")
        scope = scopes[scope_name]
        strongest = selection[key]["strongest_baseline"]
        for system, record in scope["systems"].items():
            row = {"evaluation": evaluation_name, "scope": scope_name, "decoder": detector,
                "backbone": backbone, "system": system, "label": LABELS.get(system, system),
                "interpretation": scope["interpretation"], "images": scope["images"], "sources": scope["sources"],
                "tune_origin_baseline_eligible": selection[key]["origin_baseline_eligibility"].get(system),
                "tune_selected_strongest_baseline": system == strongest,
                "marked_image_auc_secondary": record["auc"]["marked"],
                "unmarked_image_auc_secondary": record["auc"]["unmarked"],
                "additional_secondary_baseline":False,"original_primary_comparator_unchanged":True,
                "selected_correction_bound":None,"image_AUC_status":"Saved frozen evaluation scores",
                "frozen_cut":None}
            for metric in ("marked_tpr", "unmarked_tpr", "real_image_fpr", "real_source_fpr"):
                flatten_metric(row, record, metric)
            row["unmarked_loss_one_sided_upper95_approximate"] = (
                -row["unmarked_tpr_minus_probe_one_sided_lower95"] if row.get("unmarked_tpr") is not None else None)
            rows.append(row)
            for share in record["source_bootstrap"]["marked_shares_secondary"]:
                shares.append({"evaluation": evaluation_name, "decoder": detector, "backbone": backbone, "system": system,
                    "marked_share": share["marked_share"], "source_balanced_tpr": share["tpr"],
                    "tpr_ci_low": share["ci95"][0], "tpr_ci_high": share["ci95"][1],
                    "gain_vs_probe": share["gain_vs_probe"], "gain_ci_low": share["gain_ci95"][0],
                    "gain_ci_high": share["gain_ci95"][1], "scope": "Descriptive fixed-cut counterfactual mixture"})
    return pd.DataFrame(rows), pd.DataFrame(shares)


def paired_primary_comparisons(results, selection, evaluation_name):
    rows=[]
    for key,scopes in results["models"].items():
        detector,backbone=key.split("/")
        strongest=selection[key]["strongest_baseline"]
        bootstrap=scopes["primary_common"]["systems"]["primary"]["source_bootstrap"]
        for metric in ("marked_tpr","unmarked_tpr","real_image_fpr","real_source_fpr"):
            if bootstrap.get(metric) is None: continue
            for reference,record in bootstrap[metric].get("comparisons",{}).items():
                rows.append({"evaluation":evaluation_name,"decoder":detector,"backbone":backbone,
                    "system":"primary","reference":reference,"metric":metric,"estimate":record["estimate"],
                    "ci_low":record["ci95"][0],"ci_high":record["ci95"][1],
                    "prespecified_primary_gain_comparison":key==PRIMARY and metric=="marked_tpr" and reference in ("probe",strongest),
                    "additional_secondary_comparison":False})
    return rows


def aligned_predictions(frame, key, conditions=None):
    detector, backbone = key.split("/")
    part = frame.loc[frame.detector.eq(detector) & frame.backbone.eq(backbone)].copy()
    if conditions is not None: part = part.loc[part.condition.isin(conditions)]
    identities = ["image_id", "source_id", "collection", "label", "watermark_method", "embedded_payload", "condition"]
    meta = part[identities].drop_duplicates().sort_values("image_id").reset_index(drop=True)
    if meta.image_id.duplicated().any(): raise ValueError("One image has inconsistent scientific metadata")
    names = sorted(part.system.unique())
    if "probe" not in names or "primary" not in names: raise ValueError("Probe and primary are required")
    scores, decisions = {}, {}
    for name, group in part.groupby("system"):
        group = group.set_index("image_id").reindex(meta.image_id)
        if group.score.isna().any() or group.decision.isna().any():
            raise ValueError("Every comparator must score exactly the same received images")
        scores[name], decisions[name] = group.score.to_numpy(float), group.decision.to_numpy(bool)
    meta["is_unmarked"] = meta.watermark_method.eq("unmarked")
    if not meta.loc[meta.label.eq(0), "is_unmarked"].all():
        raise ValueError("Photographic controls unexpectedly marked in origin reporting")
    sources = meta[["source_id", "collection", "label"]].drop_duplicates()
    if sources.source_id.duplicated().any(): raise ValueError("A source has inconsistent collection/label")
    return meta, scores, decisions


def subgroup_intervals(meta, decisions, seed):
    """Vectorized equivalent of the frozen evaluator's source_bootstrap.

    Source ordering, collection ordering and integer multiplicity draws match
    that implementation. Every system shares one draw matrix within a scope.
    """
    source_rows = meta[["source_id", "collection", "label"]].drop_duplicates().sort_values("source_id").reset_index(drop=True)
    source_index = {source: i for i, source in enumerate(source_rows.source_id)}
    codes = meta.source_id.map(source_index).to_numpy(int)
    names = sorted(decisions)
    values = np.column_stack([decisions[name] for name in names]).astype(float)
    baseline = names.index("probe")
    n = len(source_rows)
    ai = meta.label.eq(1).to_numpy()
    marked = ai & ~meta.is_unmarked.to_numpy()
    unmarked = ai & meta.is_unmarked.to_numpy()
    real = ~ai
    endpoints = {}
    for metric, mask in [("marked_tpr", marked), ("unmarked_tpr", unmarked), ("real_image_fpr", real)]:
        counts = np.bincount(codes[mask], minlength=n)
        sums = np.zeros((n, len(names)))
        np.add.at(sums, codes[mask], values[mask])
        result = np.full_like(sums, np.nan)
        valid = counts > 0
        result[valid] = sums[valid] / counts[valid, None]
        endpoints[metric] = result
    any_real = np.full((n, len(names)), np.nan)
    real_sources = np.unique(codes[real])
    any_real[real_sources] = 0.
    np.maximum.at(any_real, codes[real], values[real])
    endpoints["real_source_fpr"] = any_real
    rng = np.random.default_rng(seed)
    draws = np.zeros((BOOTSTRAPS, n), dtype=np.int32)
    for _, group in source_rows.groupby("collection", sort=True):
        stratum = group.index.to_numpy(int)
        picked = rng.integers(0, len(stratum), size=(BOOTSTRAPS, len(stratum)))
        for j in range(BOOTSTRAPS): draws[j, stratum] = np.bincount(picked[j], minlength=len(stratum))
    rows = {name: {"system": name} for name in names}
    for metric, endpoint in endpoints.items():
        valid = np.isfinite(endpoint[:, 0])
        if not valid.any(): continue
        w = draws[:, valid]
        boot = (w @ endpoint[valid]) / w.sum(axis=1)[:, None]
        points = endpoint[valid].mean(axis=0)
        delta = boot - boot[:, baseline, None]
        for column, name in enumerate(names):
            row = rows[name]
            row[metric], row[metric + "_sources"] = float(points[column]), int(valid.sum())
            row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(boot[:, column], [.025, .975])
            row[metric + "_minus_probe"] = float(points[column] - points[baseline])
            row[metric + "_minus_probe_ci_low"], row[metric + "_minus_probe_ci_high"] = np.quantile(delta[:, column], [.025, .975])
            row[metric + "_minus_probe_one_sided_lower95"] = float(np.quantile(delta[:, column], .05))
    return list(rows.values())


def secondary_tables(frame, evaluation_name, results, common_only=True):
    """Each family uses shared real controls; no multiplication of real N."""
    family_rows, condition_rows, payload_rows, noise_rows, photo_rows = [], [], [], [], []
    for di, detector in enumerate(("dwt_dct", "trustmark")):
        for bi, backbone in enumerate(("resnet", "clip")):
            key = f"{detector}/{backbone}"
            meta, _, decisions = aligned_predictions(frame, key)
            seed = SEED + 100 * di + bi
            common = meta.condition.isin(COMMON).to_numpy() if common_only else np.ones(len(meta), dtype=bool)
            families = sorted(meta.loc[meta.label.eq(1), "collection"].unique())
            scopes = []
            for family in families:
                scopes.append(("family", family, common & (meta.label.eq(0) | meta.collection.eq(family)).to_numpy()))
            for collection in sorted(meta.loc[meta.label.eq(0),"collection"].unique()):
                scopes.append(("photographic_collection",collection,common & (meta.label.eq(1) | meta.collection.eq(collection)).to_numpy()))
            for condition in sorted(meta.condition.unique()):
                scopes.append(("condition", condition, meta.condition.eq(condition).to_numpy()))
            for payload in sorted(meta.loc[meta.label.eq(1) & ~meta.is_unmarked, "embedded_payload"].unique()):
                mask = common & (meta.is_unmarked | meta.embedded_payload.eq(payload)).to_numpy()
                scopes.append(("payload", payload, mask))
            for condition in sorted(meta.loc[meta.condition.str.contains("noise|seed", case=False, regex=True), "condition"].unique()):
                scopes.append(("noise", condition, meta.condition.eq(condition).to_numpy()))
            for kind, value, mask in scopes:
                sub = meta.loc[mask].reset_index(drop=True)
                if sub.empty: continue
                subset_decisions = {name: values[mask] for name, values in decisions.items()}
                records = subgroup_intervals(sub, subset_decisions, seed)
                for row in records:
                    row.update({"evaluation": evaluation_name, "decoder": detector, "backbone": backbone,
                        "scope": kind, "scope_value": value, "images": len(sub), "sources": sub.source_id.nunique(),
                        "real_sources": sub.loc[sub.label.eq(0), "source_id"].nunique(),
                        "AI_sources": sub.loc[sub.label.eq(1), "source_id"].nunique(),
                        "interval_status": "Secondary descriptive, conditional on fitted models/fixed payloads/seeds",
                        "real_controls": "One photographic collection; all AI controls reused" if kind=="photographic_collection" else "Both photographic collections reused for each AI family; not additional independent real sources"})
                {"family": family_rows, "condition": condition_rows, "payload": payload_rows, "noise": noise_rows,"photographic_collection":photo_rows}[kind].extend(records)
                print("Secondary saved-score intervals", evaluation_name, key, kind, value, flush=True)
    return {"family_results": family_rows, "condition_results": condition_rows,
            "payload_results": payload_rows, "noise_results": noise_rows,"photographic_collection_results":photo_rows}


def additional_baselines(path,learner,selection,primary_results,evaluation_receipt,evaluation_name):
    """Import sealed additional aggregate results; no score reconstruction."""
    path=path.resolve()
    receipt=verify_outputs(path,"complete.json")
    cal_receipt=read(path/"calibration_complete.json"); cuts=read(path/"cuts_locked.json"); results=read(path/"results.json")
    script=ROOT/"scripts/phase6_additional_baselines.py"
    expected={"script_sha256":digest(script),"root_protocol_sha256":digest(ROOT/"outputs/phase6/frozen_protocol.json"),
        "fit_sha256":digest(learner/"fit_complete.json"),"development_features_sha256":digest(ROOT/"outputs/phase6/development/features.csv"),
        "calibration_features_sha256":digest(ROOT/"outputs/phase6/calibration/features.csv")}
    if cuts["inputs"]!=expected or cal_receipt["inputs"]!=expected:
        raise ValueError("Additional comparator calibration dependencies changed")
    if cal_receipt["cuts_sha256"]!=digest(path/"cuts_locked.json") or receipt["calibration_complete_sha256"]!=digest(path/"calibration_complete.json"):
        raise ValueError("Additional comparator cut receipt changed")
    if receipt["test_features_sha256"]!=evaluation_receipt["test_features_sha256"] or receipt["script_sha256"]!=digest(script) or receipt["primary_retuned"] is not False:
        raise ValueError("Additional comparators do not share frozen primary test features or were retuned")
    start=read(ROOT/"outputs/phase6/test/inference_started.json")
    if datetime.fromisoformat(cuts["locked_at_utc"])>=datetime.fromisoformat(start["utc"]):
        raise ValueError("Additional comparators were sealed after first final scores")
    if results["secondary_only"] is not True or results["original_primary_comparator_unchanged"] is not True or results["bootstrap_samples"]!=BOOTSTRAPS:
        raise ValueError("Additional results are not unchanged secondary comparators")
    rows,shares,comparisons=[],[],[]
    for key,systems in results["systems"].items():
        detector,backbone=key.split("/")
        original=primary_results["models"][key]["primary_common"]
        for reference in ("probe","primary"):
            for metric in ("marked_tpr","unmarked_tpr","real_image_fpr","real_source_fpr"):
                current=systems[reference]["source_bootstrap"][metric]
                saved=original["systems"][reference]["source_bootstrap"][metric]
                if abs(current["estimate"]-saved["estimate"])>1e-12 or not np.allclose(current["ci95"],saved["ci95"],rtol=0,atol=1e-12):
                    raise ValueError("Additional baseline reference values differ from primary saved outputs")
        for name in ("full_message_union","tuned_confidence"):
            record=systems[name]
            model=cuts["systems"][key][name]["model"]
            row={"evaluation":evaluation_name,"scope":"primary_common","decoder":detector,"backbone":backbone,
                "system":name,"label":LABELS[name],"interpretation":"Additional simple comparator sealed before first fresh scores; secondary",
                "images":original["images"],"sources":original["sources"],"tune_origin_baseline_eligible":None,
                "tune_selected_strongest_baseline":False,"additional_secondary_baseline":True,
                "original_primary_comparator_unchanged":True,"selected_correction_bound":model.get("bound"),
                "marked_image_auc_secondary":None,"unmarked_image_auc_secondary":None,
                "image_AUC_status":"Not supplied in additional aggregate output; no scores reconstructed",
                "frozen_cut":record["policy"]["threshold"]}
            for metric in ("marked_tpr","unmarked_tpr","real_image_fpr","real_source_fpr"):flatten_metric(row,record,metric)
            row["unmarked_loss_one_sided_upper95_approximate"]=-row["unmarked_tpr_minus_probe_one_sided_lower95"]
            rows.append(row)
            for share in record["source_bootstrap"]["marked_shares_secondary"]:
                shares.append({"evaluation":evaluation_name,"decoder":detector,"backbone":backbone,"system":name,
                    "marked_share":share["marked_share"],"source_balanced_tpr":share["tpr"],"tpr_ci_low":share["ci95"][0],"tpr_ci_high":share["ci95"][1],
                    "gain_vs_probe":share["gain_vs_probe"],"gain_ci_low":share["gain_ci95"][0],"gain_ci_high":share["gain_ci95"][1],
                    "scope":"Additional secondary fixed-cut counterfactual mixture"})
            for metric in ("marked_tpr","unmarked_tpr","real_image_fpr","real_source_fpr"):
                comparison=systems["primary"]["source_bootstrap"][metric]["comparisons"][name]
                comparisons.append({"evaluation":evaluation_name,"decoder":detector,"backbone":backbone,"system":"primary","reference":name,
                    "metric":metric,"estimate":comparison["estimate"],"ci_low":comparison["ci95"][0],"ci_high":comparison["ci95"][1],
                    "prespecified_primary_gain_comparison":False,"additional_secondary_comparison":True})
    provenance={"secondary_only":True,"original_primary_comparator_unchanged":True,
        "folder":path.relative_to(ROOT).as_posix(),"complete_receipt_sha256":digest(path/"complete.json"),
        "calibration_receipt_sha256":digest(path/"calibration_complete.json"),"cuts_sha256":digest(path/"cuts_locked.json"),
        "results_sha256":digest(path/"results.json"),"locked_at_utc":cuts["locked_at_utc"],"first_test_scores_started_utc":start["utc"],
        "score_reconstruction_or_inference":False,"registered_rules":cuts["registered_rules"],
        "interpretation":"Comparators influence empirical novelty and utility interpretation without replacing the original confirmatory endpoint/comparator",
        "decision_boundary_ties":"Not available from aggregate-only additional results; original saved-score tie counts are reported separately"}
    return pd.DataFrame(rows),pd.DataFrame(shares),comparisons,provenance


def verify_source_registration(learner, folder, receipt, frame):
    """Check source/feature receipts, without importing inference backbones."""
    protocol_path=ROOT/"outputs/phase6/frozen_protocol.json"
    source_lock_path=ROOT/"outputs/phase6/test_sources_locked.json"
    inventory_path=ROOT/"outputs/phase6/data_inventory/primary_source_manifest.csv"
    root_protocol=read(protocol_path); source_lock=read(source_lock_path)
    if source_lock["protocol_sha256"]!=digest(protocol_path) or source_lock["source_manifest_sha256"]!=digest(inventory_path):
        raise ValueError("Root method/source lock changed")
    if ROOT/root_protocol["learner_output"]!=learner.resolve():
        raise ValueError("Reporting learner differs from root method lock")
    if receipt["script_sha256"]!=root_protocol["input_sha256"]["scripts/phase6_fusion.py"]:
        raise ValueError("Saved evaluation implementation differs from root lock")
    feature_receipt_path=ROOT/"outputs/phase6/test/features_complete.json"
    if folder.name=="fresh_test":
        feature_receipt=read(feature_receipt_path)
        if feature_receipt["features_sha256"]!=receipt["test_features_sha256"] or feature_receipt["protocol_sha256_before_scores"]!=digest(protocol_path):
            raise ValueError("Fresh evaluation differs from the locked test-feature receipt")
    inventory=pd.read_csv(inventory_path,keep_default_na=False,dtype={"source_id":str})
    registered=inventory.loc[inventory.role.eq("test"),["source_id","collection","label"]].drop_duplicates()
    actual=frame[["source_id","collection","label"]].drop_duplicates()
    known=registered.set_index("source_id")
    if not set(actual.source_id).issubset(set(registered.source_id)):
        raise ValueError("Evaluation includes unregistered final sources")
    for row in actual.to_dict("records"):
        match=known.loc[row["source_id"]]
        if row["collection"]!=match.collection or int(row["label"])!=int(match.label):
            raise ValueError("Saved source collection/origin differs from source registration")
    source_receipt=read(ROOT/"outputs/phase6/data_inventory/selection_complete.json")
    if source_receipt["manifest_sha256"]!=digest(inventory_path): raise ValueError("Source-selection receipt changed")
    return {"registered_test_sources":len(registered),"evaluated_sources":len(actual),
        "root_protocol_sha256":digest(protocol_path),"source_lock_sha256":digest(source_lock_path),
        "source_inventory_sha256":digest(inventory_path),"all_registered_test_sources_present":set(actual.source_id)==set(registered.source_id)}


def attack_analysis(path,learner,selection,output):
    """Explicit secondary adapter for the separate attack-policy reporter."""
    path=path.resolve()
    receipt=verify_outputs(path,"complete.json")
    results=read(path/"results.json")
    cuts=read(path/"secondary_cuts_locked.json")
    cut_receipt=read(path/"calibration_complete.json")
    root_protocol=ROOT/"outputs/phase6/frozen_protocol.json"
    attack_script=ROOT/"scripts/phase6_attack_reporting.py"
    expected_inputs={"script_sha256":digest(attack_script),"root_protocol_sha256":digest(root_protocol),
        "model_fit_sha256":digest(learner/"fit_complete.json"),
        "calibration_features_sha256":digest(ROOT/"outputs/phase6/calibration_attacks/features.csv")}
    if cuts["inputs"]!=expected_inputs or cut_receipt["inputs"]!=expected_inputs:
        raise ValueError("Secondary attack cut dependencies changed")
    cut_sha=digest(path/"secondary_cuts_locked.json")
    if cut_receipt["secondary_cuts_sha256"]!=cut_sha or receipt["secondary_cuts_sha256"]!=cut_sha or results["secondary_cuts_sha256"]!=cut_sha:
        raise ValueError("Secondary attack cut receipt mismatch")
    if receipt["script_sha256"]!=digest(attack_script) or receipt["test_features_sha256"]!=digest(ROOT/"outputs/phase6/test_attacks/features.csv"):
        raise ValueError("Secondary attack evaluation dependencies changed")
    if results.get("secondary_only") is not True or results["bootstrap_samples"]!=BOOTSTRAPS or receipt.get("primary_model_or_threshold_retuned") is not False:
        raise ValueError("Attack report is not an unchanged secondary assessment")
    frame=pd.read_csv(path/"predictions.csv",keep_default_na=False,
        dtype={"image_id":str,"source_id":str,"embedded_payload":str,"label":np.int8,"decision":np.int8,"score":np.float64})
    policies=("core_cuts_diagnostic","attack_family_source_cuts")
    if set(frame.policy)!=set(policies) or frame.duplicated(["policy","detector","backbone","system","image_id"]).any():
        raise ValueError("Attack prediction policy coverage or uniqueness mismatch")
    if not np.isfinite(frame.score).all(): raise ValueError("Nonfinite attack scores")
    table_rows,share_rows=[],[]
    secondary={name:[] for name in ("family_results","condition_results","payload_results","noise_results","photographic_collection_results")}
    for policy in policies:
        table,shares=aggregate(results,selection,"attack_analysis",policy)
        table["secondary_only"]=True; shares["secondary_only"]=True; shares["policy"]=policy
        table_rows.append(table); share_rows.append(shares)
        selected=frame.loc[frame.policy.eq(policy)]
        for key,scopes in results["models"].items():
            meta,scores,decisions=aligned_predictions(selected,key)
            scope=scopes[policy]
            if len(meta)!=scope["images"] or meta.source_id.nunique()!=scope["sources"]:
                raise ValueError("Secondary attack source/image count differs from report")
            matching_policies=(read(learner/"calibration.json")["origin_policies"] if policy=="core_cuts_diagnostic" else cuts["policies"])
            for system,values in scores.items():
                threshold=matching_policies[key][system]["threshold"]
                threshold=threshold if threshold is not None else np.inf
                if not np.array_equal(values>threshold,decisions[system]):
                    raise ValueError("Secondary decisions differ from the labeled frozen cut family")
        records=secondary_tables(selected,"attack_analysis",results,common_only=False)
        for kind,rows in records.items():
            for row in rows:
                row.update({"policy":policy,"secondary_only":True,
                    "origin_label_scope":"Input-source provenance; reconstructed photographic inputs contain generated pixels"})
            secondary[kind].extend(rows)
    pd.concat(table_rows,ignore_index=True).to_csv(output/"attack_systems_aggregate.csv",index=False)
    pd.concat(share_rows,ignore_index=True).to_csv(output/"attack_marked_shares.csv",index=False)
    for kind,rows in secondary.items(): write_records_csv(output/("attack_"+kind+".csv"),rows)
    shutil.copyfile(path/"quality.csv",output/"attack_quality_saved.csv")
    return {"secondary_only":True,"folder":path.relative_to(ROOT).as_posix(),"complete_receipt_sha256":digest(path/"complete.json"),
        "cut_receipt_sha256":digest(path/"calibration_complete.json"),"secondary_cuts_sha256":cut_sha,
        "policies":list(policies),"no_primary_cuts_changed":True,
        "interpretation":"Extra attack/noise family is never remapped to confirmatory fresh_test. Core cuts are out-of-family diagnostics; separate attack cuts cover only their fixed secondary family under assumptions.",
        "quality_interpretation":"Copied attack-input pixel PSNR/SSIM summaries; shared pixel budgets do not imply shared semantic distortion"}


def plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save_figure(fig, assets, name, title):
    assets.mkdir(parents=True, exist_ok=True)
    pdf, png = assets / (name + ".pdf"), assets / (name + ".png")
    fig.savefig(pdf, bbox_inches="tight", metadata={"Title": title, "Creator": "Phase 6 saved-output reporting", "CreationDate": None})
    fig.savefig(png, dpi=300, bbox_inches="tight")
    from PIL import Image
    with Image.open(png) as raster:
        width,height=raster.size
    return {"pdf": pdf.relative_to(ROOT).as_posix(), "pdf_sha256": digest(pdf),
            "png": png.relative_to(ROOT).as_posix(), "png_sha256": digest(png),
            "png_width":width,"png_height":height,"title":title,
            "reporting_script_sha256":digest(__file__)}


def figure_asset_records(artifacts):
    rows=[]
    for artifact in artifacts:
        for kind in ("pdf","png"):
            rows.append({"path":artifact[kind],"format":kind,"sha256":artifact[kind+"_sha256"],
                "title":artifact["title"],"reporting_script_sha256":artifact["reporting_script_sha256"],
                "width_pixels":artifact.get("png_width") if kind=="png" else None,
                "height_pixels":artifact.get("png_height") if kind=="png" else None})
    return rows


def pipeline_figure(protocol, assets):
    plt = plot_setup()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig = plt.figure(figsize=(7.1, 2.75))
    ax = fig.add_axes([.015,.04,.97,.92])
    ax.set_xlim(-.12, 10.12); ax.set_ylim(0, 4.4); ax.axis("off")
    def box(x, y, w, h, text, color="#f4f5f7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=color, edgecolor="#354052", linewidth=.8))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7.1, linespacing=1.2)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color="#354052", linewidth=.9))
    ax.text(0, 4.25, "Inference: only the received image, its JPEG90 view and three fixed queries", fontsize=8.4, weight="bold")
    box(.05, 2.9, 1.5, .95, "Received image\nFrozen passive\nlogit z")
    box(1.93, 2.9, 2.45, .95, "Decode image + JPEG90\n11 decoder features\nSame 3 registered queries")
    box(4.78, 2.9, 2.08, .95, "Bounded correction\n"+r"$c=B\max_q\sigma(\theta^\top x_{iq})$"+"\n"+r"$s=z+c,\;0\leq c\leq B$", "#f2edf7")
    box(7.26, 2.9, 2.6, .95, "Locked decision cut\nAccept received image if "+r"$s>t$"+"\n"+r"$t$ calibrated on source maxima"+"\nNo issuer authentication")
    for left, right in [(1.55,1.93),(4.38,4.78),(6.86,7.26)]: arrow(left, 3.37, right, 3.37)
    ax.text(0, 2.28, "Development and independent calibration", fontsize=8.4, weight="bold")
    box(.05, .95, 2.82, 1.03, "Same source / same edit\nBalanced cross entropy\n+ null and paired penalties")
    box(3.21, .95, 2.82, 1.03, "Tune source FPR and harm\nLock bound and coefficients\nConstraint check does not reselect")
    box(6.37, .95, 3.49, 1.03, "New photographic calibration\nMax over 4 conditions / all queries\nOrder-statistic cut per collection\nUse the larger photographic cut")
    arrow(2.87,1.46,3.21,1.46); arrow(6.03,1.46,6.37,1.46)
    arrow(5.2,1.98,5.2,2.86); arrow(8.4,1.98,8.4,2.86)
    ax.text(0,.30,"Training metadata never enters inference. Query search is identical across origins.\nNominal source-FPR control assumes exchangeability within the fixed collection and processing family.",fontsize=6.9,linespacing=1.25)
    artifact=save_figure(fig,assets,"phase6_pipeline","Actual paired fusion and independent source calibration")
    plt.close(fig)
    artifact.update({"measurements_used":False,"registered_query_count":len(protocol["registered_queries"]),"evaluated_common_conditions":list(COMMON),
        "internal_received_view":"JPEG90; not a clean reference or an additional claimed test condition"})
    return artifact


def measured_figures(frame, results, calibration, assets):
    plt=plot_setup()
    meta,scores,decisions=aligned_predictions(frame,PRIMARY,COMMON)
    primary=results["models"][PRIMARY]["primary_common"]
    if len(meta)!=primary["images"] or meta.source_id.nunique()!=primary["sources"]:
        raise ValueError("Primary figure source/image totals differ from frozen report")
    real=meta.label.eq(0).to_numpy(); ai=~real
    states={"Marked AI + photographs":ai&~meta.is_unmarked.to_numpy(),"Unmarked AI + photographs":ai&meta.is_unmarked.to_numpy()}
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.6),gridspec_kw={"width_ratios":[1,1,1.03]})
    metadata={"scope":"Four fixed common conditions; fresh TrustMark/ResNet, image ROCs separate from deployed source operating points",
        "images":len(meta),"sources":int(meta.source_id.nunique()),"conditions":list(COMMON),"ROC_states":{},"operating_points":[]}
    for ax,(state,positive) in zip(axes[:2],states.items()):
        mask=real|positive; y=meta.label.to_numpy()[mask]
        lower=1/(2*int(real.sum()))
        state_key="marked" if state.startswith("Marked") else "unmarked"
        details={"positive_images":int(positive.sum()),"real_images":int(real.sum()),
            "positive_sources":int(meta.loc[positive,"source_id"].nunique()),"real_sources":int(meta.loc[real,"source_id"].nunique()),
            "log_axis_left_edge":lower,"zero_FPR_display":"Zero-FPR segments shown at the left display edge; their actual FPR remains zero", "methods":{}}
        for system in CURVES:
            fpr,tpr,_=roc_curve(y,scores[system][mask],drop_intermediate=False)
            auc=float(roc_auc_score(y,scores[system][mask]))
            expected=primary["systems"][system]["auc"][state_key]
            if abs(auc-expected)>1e-12: raise ValueError("Figure AUC differs from saved evaluator")
            ax.plot(np.maximum(fpr,lower),tpr,color=COLORS[system],linestyle=STYLES[system],linewidth=1.35 if system!="primary" else 1.9)
            details["methods"][system]={"auc":auc,"maximum_TPR_at_zero_image_FPR":float(tpr[fpr==0].max())}
        ax.set_xscale("log"); ax.set_xlim(lower,1); ax.set_ylim(0,1.02)
        ax.set_xlabel("Image FPR (log scale)"); ax.set_ylabel("Image TPR")
        ax.set_title(state,fontsize=8.4)
        ax.grid(True,which="major",alpha=.18)
        ax.text(.02,.025,"FPR = 0 at left edge",transform=ax.transAxes,fontsize=6.3,color="#4a4a4a")
        metadata["ROC_states"][state_key]=details
    ax=axes[2]
    for system in CURVES:
        row=primary["systems"][system]
        decision=decisions[system]
        source_real=pd.DataFrame({"source":meta.source_id.to_numpy()[real],"accepted":decision[real]}).groupby("source").accepted.max()
        fpr=float(source_real.mean())
        if abs(fpr-row["real_source_fpr"])>1e-12: raise ValueError("Operating source FPR differs from saved evaluator")
        policies=calibration["origin_policies"][PRIMARY][system]
        cut=policies["threshold"] if policies["threshold"] is not None else np.inf
        if not np.array_equal(scores[system]>cut,decision): raise ValueError("Saved deployed decisions differ from frozen strict cuts")
        marked=row["source_bootstrap"]["marked_tpr"]["estimate"]; unmarked=row["source_bootstrap"]["unmarked_tpr"]["estimate"]
        ax.plot([fpr,fpr],[unmarked,marked],color=COLORS[system],linewidth=.7,alpha=.65)
        ax.scatter(fpr,marked,marker="o",s=28,color=COLORS[system],zorder=3)
        ax.scatter(fpr,unmarked,marker="^",s=28,facecolors="white",edgecolors=COLORS[system],linewidths=1,zorder=3)
        metadata["operating_points"].append({"system":system,"photographic_source_FPR_any_view":fpr,
            "source_balanced_marked_TPR":marked,"source_balanced_unmarked_TPR":unmarked,"threshold":policies["threshold"],
            "real_sources":len(source_real),"source_TPR_definition":"Mean acceptance across covers/views within an AI source, then across sources; not any-view AI TPR"})
    largest=max(r["photographic_source_FPR_any_view"] for r in metadata["operating_points"])
    ax.set_xlim(-.003,max(.06,largest*1.12)); ax.set_ylim(0,1.02)
    ax.axvline(.05,color="#999999",linewidth=.7,linestyle=":")
    ax.set_xlabel("Source FPR (any photo view)"); ax.set_ylabel("Source-balanced TPR")
    ax.set_title("Frozen operating points",fontsize=8.4); ax.grid(True,alpha=.18)
    ax.text(.035,.98,"● Marked     △ Unmarked\n5% nominal source target",transform=ax.transAxes,fontsize=6.5,va="top")
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=COLORS[n],linestyle=STYLES[n],label=LABELS[n]) for n in CURVES]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.02),ncol=5,frameon=False)
    fig.tight_layout(rect=[0,.085,1,1],w_pad=.6)
    metadata["ROC_artifact"]=save_figure(fig,assets,"phase6_roc_tm_resnet","Separate image ROCs and deployed source-FPR operating points")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.4,2.4))
    metadata["marked_share"]={}
    for system in CURVES:
        rows=primary["systems"][system]["source_bootstrap"]["marked_shares_secondary"]
        x=np.array([r["marked_share"] for r in rows]); y=np.array([r["gain_vs_probe"] for r in rows])
        lo=np.array([r["gain_ci95"][0] for r in rows]); hi=np.array([r["gain_ci95"][1] for r in rows])
        if not np.array_equal(x,np.array([0,.25,.5,.75,1.])): raise ValueError("Unexpected marked-share grid")
        ax.plot(x*100,y,label=LABELS[system],color=COLORS[system],linestyle=STYLES[system],marker="o",markersize=3,linewidth=1.4)
        if system!="probe": ax.fill_between(x*100,lo,hi,color=COLORS[system],alpha=.12,linewidth=0)
        metadata["marked_share"][system]=rows
    ax.axhline(0,color="#888888",linewidth=.65)
    ax.set_xlim(0,100); ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Marked AI share (%)"); ax.set_ylabel("Fixed-cut TPR gain vs probe")
    ax.legend(frameon=False,fontsize=6.5,loc="best"); ax.grid(True,alpha=.18)
    fig.tight_layout()
    metadata["marked_share_artifact"]=save_figure(fig,assets,"phase6_marked_share_tm_resnet","Fixed-cut paired marked-share TPR gains with source intervals")
    plt.close(fig)
    metadata["marked_share_interpretation"]="Descriptive reweighting of paired states at unchanged independently calibrated cuts; not measured real-world prevalence or a prevalence-dependent recalibration"
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner",type=Path,default=LEARNER)
    parser.add_argument("--evaluation",default="fresh_test")
    parser.add_argument("--extra-evaluation",action="append",default=[])
    parser.add_argument("--attack-analysis",type=Path,nargs="?",const=ROOT/"outputs/phase6/attack_analysis",
        help="Explicit receipt-verified secondary attack reporter folder; never treated as the primary fresh evaluation")
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--assets",type=Path,default=ASSETS)
    parser.add_argument("--pipeline-only",action="store_true",help="Create the measured-data-free actual method figure before final evaluation")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    fit,protocol,selection=locked_inputs(args.learner)
    pipeline=pipeline_figure(protocol,args.assets)
    if args.pipeline_only:
        write_json(args.output/"pipeline_metadata.json",{"fit_receipt_sha256":digest(args.learner/"fit_complete.json"),
            "protocol_sha256":digest(args.learner/"protocol.json"),"figure":pipeline,"fresh_scores_read":False})
        write_json(args.output/"figure_assets.json",{"measurement_figures_available":False,"assets":figure_asset_records([pipeline])})
        print("Created actual method pipeline; no fresh scores read",flush=True)
        return
    calibration_receipt=verify_outputs(args.learner,"calibration_complete.json")
    if calibration_receipt["fit_complete_sha256"]!=digest(args.learner/"fit_complete.json"):
        raise ValueError("Calibration belongs to different fitted models")
    calibration=read(args.learner/"calibration.json")
    folder,receipt,results,frame=evaluation(args.learner,args.evaluation)
    source_registration=verify_source_registration(args.learner,folder,receipt,frame)
    table,shares=aggregate(results,selection,args.evaluation)
    extra_table,extra_shares,extra_comparisons,extra_provenance=additional_baselines(
        ROOT/"outputs/phase6/additional_baselines",args.learner,selection,results,receipt,args.evaluation)
    table=pd.concat([table,extra_table],ignore_index=True)
    shares=pd.concat([shares,extra_shares],ignore_index=True)
    table.to_csv(args.output/"systems_aggregate.csv",index=False)
    primary_mask=table.decoder.eq("trustmark")&table.backbone.eq("resnet")
    order=list(BASELINES)+["full_message_union","tuned_confidence","primary"]
    primary=table.loc[primary_mask&table.system.isin(order)].copy()
    primary["order"]=primary.system.map({name:i for i,name in enumerate(order)})
    primary.sort_values("order").drop(columns="order").to_csv(args.output/"primary_comparison.csv",index=False)
    table.loc[table.system.isin(ABLATIONS)].to_csv(args.output/"all_system_ablations.csv",index=False)
    table.loc[primary_mask&table.system.isin(ABLATIONS)].to_csv(args.output/"primary_ablations.csv",index=False)
    shares.to_csv(args.output/"marked_shares.csv",index=False)
    write_records_csv(args.output/"paired_primary_comparisons.csv",paired_primary_comparisons(results,selection,args.evaluation)+extra_comparisons)
    metadata=measured_figures(frame,results,calibration,args.assets)
    secondary={name:[] for name in ("family_results","condition_results","payload_results","noise_results","photographic_collection_results")}
    all_receipts={args.evaluation:{"folder":folder.relative_to(ROOT).as_posix(),"receipt_sha256":digest(folder/"complete.json"),"receipt":receipt}}
    supplied=[(args.evaluation,frame,results)]
    for name in args.extra_evaluation:
        if name in all_receipts: raise ValueError("An evaluation must not be added twice")
        extra_folder,extra_receipt,extra_results,extra_frame=evaluation(args.learner,name)
        all_receipts[name]={"folder":extra_folder.relative_to(ROOT).as_posix(),"receipt_sha256":digest(extra_folder/"complete.json"),"receipt":extra_receipt}
        supplied.append((name,extra_frame,extra_results))
    for name,predictions,report in supplied:
        records=secondary_tables(predictions,name,report)
        for kind,rows in records.items(): secondary[kind].extend(rows)
    for kind,rows in secondary.items():
        write_records_csv(args.output/(kind+".csv"),rows)
    attack_receipts=attack_analysis(args.attack_analysis,args.learner,selection,args.output) if args.attack_analysis else None
    assessment=results["confirmatory_assessment"]
    write_json(args.output/"primary_assessment.json",assessment)
    write_json(args.output/"figure_metadata.json",metadata)
    write_json(args.output/"figure_assets.json",{"measurement_figures_available":True,
        "evaluation_receipt_sha256":digest(folder/"complete.json"),
        "assets":figure_asset_records([pipeline,metadata["ROC_artifact"],metadata["marked_share_artifact"]])})
    receipt_paths={"root_method_protocol":ROOT/"outputs/phase6/frozen_protocol.json",
        "root_source_lock":ROOT/"outputs/phase6/test_sources_locked.json",
        "sampling_protocol":ROOT/"outputs/phase6/data_inventory/sampling_protocol.json",
        "source_selection_receipt":ROOT/"outputs/phase6/data_inventory/selection_complete.json"}
    provenance={"reporting_script_sha256":digest(__file__),"fit_receipt_sha256":digest(args.learner/"fit_complete.json"),
        "report_stage":"Primary intermediate presentation" if not args.attack_analysis else "Primary plus separately labeled secondary attack presentation",
        "release_readiness_claim":False,
        "calibration_receipt_sha256":digest(args.learner/"calibration_complete.json"),"evaluation_receipts":all_receipts,
        "source_registration":source_registration,"secondary_attack_analysis":attack_receipts,
        "additional_secondary_baselines":extra_provenance,
        "source_protocol_receipts":{name:{"sha256":digest(path),"content":read(path)} for name,path in receipt_paths.items() if path.exists()},
        "primary_intervals":"Copied from frozen evaluator, never replaced by newly chosen analysis",
        "secondary_intervals":"2,000 deterministic collection-stratified source draws; derivatives/payloads/seeds remain together; descriptive",
        "score_or_threshold_refitting":False,"inference_performed":False,"pipeline":pipeline,
        "processing_caveat":"Primary common128 round trip then JPEG95; native-to256 processing is a separate diagnostic",
        "scope_caveats":["GenImage/photo collections are not caption paired", "Supervised-family holdout does not exclude backbone pretraining exposure",
            "Nominal source FPR is marginal under fixed-family within-collection exchangeability", "Margin evidence/native message recovery do not authenticate an issuer",
            "Unmarked and real false-positive changes are retained, including ineligible comparators", "Extra attacks/payloads/seeds do not enlarge the independent source count"],
        "independent_source_counts":metadata["sources"],"confirmatory_coverage_valid":assessment["coverage"]["valid"]}
    write_json(args.output/"reporting_provenance.json",provenance)
    csv_names=["systems_aggregate.csv","primary_comparison.csv","all_system_ablations.csv","primary_ablations.csv",
        "marked_shares.csv","paired_primary_comparisons.csv"]+[kind+".csv" for kind in secondary]
    if args.attack_analysis:
        csv_names += ["attack_systems_aggregate.csv","attack_marked_shares.csv","attack_quality_saved.csv"]+["attack_"+kind+".csv" for kind in secondary]
    write_json(args.output/"reporting_complete.json",{"inference_performed":False,"score_or_threshold_refitting":False,
        "primary_only_intermediate":not bool(args.attack_analysis),"release_readiness_claim":False,
        "reporting_script_sha256":digest(__file__),"evaluation_receipt_sha256":digest(folder/"complete.json"),
        "outputs_sha256":{name:digest(args.output/name) for name in csv_names},
        "secondary_row_counts":{kind:len(rows) for kind,rows in secondary.items()},
        "figure_metadata_sha256":digest(args.output/"figure_metadata.json"),
        "figure_assets_sha256":digest(args.output/"figure_assets.json"),
        "reporting_provenance_sha256":digest(args.output/"reporting_provenance.json")})
    print("Saved Phase 6 tables and scientific figures",args.output,flush=True)


if __name__=="__main__":
    main()
