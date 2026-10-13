"""Secondary attack-family calibration and diagnostics; never changes primary cuts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import phase6_fusion as f
from phase6_features import ROOT,OUT,digest,utc,write,check_final_lock
from phase6_runtime import attack_subset

DEST = OUT / "attack_analysis"
CONDITIONS = tuple([f"{n}_{v}" for n in ("waves_pgd8","waves_reconstruction_q3","sd15_ddim_s005") for v in ("raw","budget35")]
                   +[f"noise6_seed{s}" for s in (20261011,20261012,20261013)])

def inputs(role):
    directory = OUT / f"{role}_attacks"
    completed = json.loads((directory / "features_complete.json").read_text())
    if completed["features_sha256"] != digest(directory / "features.csv") or completed["manifest_sha256"] != digest(directory / "manifest.csv"):
        raise RuntimeError("Attack feature receipt mismatch")
    root=json.loads((OUT / "frozen_protocol.json").read_text())
    if completed["protocol_sha256_before_scores"] != digest(OUT / "frozen_protocol.json") or completed["driver_sha256"] != digest(ROOT / "scripts/phase6_features.py"):
        raise RuntimeError("Attack features use different frozen inference dependencies")
    frame,queries = f.read_features(directory / "features.csv",(role,),root["registered_payloads"])
    if set(frame.source_id) != attack_subset(role) or set(frame.condition) != set(CONDITIONS):
        raise RuntimeError("Secondary source subset or condition family changed")
    inventory=pd.read_csv(OUT / "data_inventory/primary_source_manifest.csv",keep_default_na=False)
    inventory=inventory.loc[inventory.role.eq(role)&inventory.source_id.isin(attack_subset(role))]
    known=inventory.set_index("source_id")[["collection","label"]].to_dict("index")
    manifest=pd.read_csv(directory / "manifest.csv",keep_default_na=False,dtype={"embedded_payload":str})
    expected_covers=set()
    for row in inventory.to_dict("records"):
        expected_covers.add((row["source_id"],"unmarked",""))
        if int(row["label"]):
            expected_covers.update((row["source_id"],method,payload) for method in f.METHODS for payload in queries)
    cover_keys=["source_id","watermark_method","embedded_payload"]
    actual_covers=set(map(tuple,manifest[cover_keys].drop_duplicates().to_numpy()))
    if actual_covers != expected_covers or manifest.image_id.duplicated().any():
        raise RuntimeError("Secondary cover coverage differs from fixed marking states")
    for key,block in manifest.groupby(cover_keys,dropna=False):
        if len(block) != len(CONDITIONS) or set(block.condition) != set(CONDITIONS):
            raise RuntimeError("Every secondary cover must include all nine conditions")
    for row in frame[["source_id","collection","label"]].drop_duplicates().to_dict("records"):
        if row["source_id"] not in known or row["collection"] != known[row["source_id"]]["collection"] or int(row["label"]) != int(known[row["source_id"]]["label"]):
            raise RuntimeError("Secondary source metadata differ from registration")
    pairs={(r["image_id"],method) for r in manifest.to_dict("records") for method in f.METHODS
           if r["watermark_method"] in (method,"unmarked")}
    if set(map(tuple,frame[["image_id","detector"]].drop_duplicates().to_numpy())) != pairs:
        raise RuntimeError("Secondary decoder coverage is incomplete")
    return frame,queries,directory

def calibration_dependencies(folder,directory):
    return {"script_sha256":digest(__file__),"root_protocol_sha256":digest(OUT / "frozen_protocol.json"),
            "model_fit_sha256":digest(folder / "fit_complete.json"),"calibration_features_sha256":digest(directory / "features.csv")}

def verify_cut_receipt(cuts,folder):
    dependencies=calibration_dependencies(folder,OUT / "calibration_attacks")
    if cuts["inputs"] != dependencies:raise RuntimeError("Secondary calibration dependencies changed")
    receipt=json.loads((DEST / "calibration_complete.json").read_text())
    if receipt["inputs"] != dependencies or receipt["secondary_cuts_sha256"] != digest(DEST / "secondary_cuts_locked.json"):
        raise RuntimeError("Secondary calibrated policy changed")

def learner():
    lock=json.loads((OUT / "frozen_protocol.json").read_text())
    folder=ROOT / lock["learner_output"]
    _,models,_=f.frozen(folder)
    f.verify_complete(folder,"calibration_complete.json",{"fit_complete_sha256":digest(folder / "fit_complete.json")})
    return folder,models,f.load_json(folder / "calibration.json")

def calibrate():
    check_final_lock("calibration")
    frame,queries,directory=inputs("calibration")
    if not frame.label.eq(0).all():raise RuntimeError("Secondary calibration is negative-input-source only")
    folder,models,primary=learner()
    DEST.mkdir(parents=True,exist_ok=True)
    path=DEST / "secondary_cuts_locked.json"
    dependencies=calibration_dependencies(folder,directory)
    if path.exists():
        saved=json.loads(path.read_text())
        verify_cut_receipt(saved,folder)
        return
    if (OUT / "test_attacks/inference_started.json").exists():raise RuntimeError("Freeze attack cuts before secondary test scores")
    policies={}
    for method in f.METHODS:
        for backbone in f.BACKBONES:
            key=f"{method}/{backbone}";data=f.view(frame,method,backbone)
            policies[key]={}
            for name,model in models[key].items():
                values=f.score(data,model)
                policy=f.conformal_cut(data["meta"],values)
                cut=policy["threshold"] if policy["threshold"] is not None else np.inf
                policy["calibration_rates"]=f.rates(data,values,cut,strict=True)
                policy["abstention_width"]=float(np.quantile(np.abs(values-cut),.2)) if np.isfinite(cut) else None
                policies[key][name]=policy
    write(path,f.finite_json({"locked_at_utc":utc(),"inputs":dependencies,"conditions":CONDITIONS,
        "sources":frame.source_id.nunique(),"policies":policies,"test_read":False,"secondary_only":True,
        "scope":"Input-source provenance after attacks, including diffusion-reconstructed photographs; this is not a camera-origin claim.",
        "no_primary_threshold_changes":True}))
    write(DEST / "calibration_complete.json",{"utc":utc(),"inputs":dependencies,
        "secondary_cuts_sha256":digest(path),"secondary_test_scores_observed":False})

def evaluate():
    check_final_lock("test")
    folder,models,primary=learner()
    cuts=json.loads((DEST / "secondary_cuts_locked.json").read_text())
    verify_cut_receipt(cuts,folder)
    frame,queries,directory=inputs("test")
    existing=DEST / "complete.json"
    if existing.exists():
        saved=json.loads(existing.read_text())
        if saved["script_sha256"] != digest(__file__) or saved["test_features_sha256"] != digest(directory / "features.csv") or saved["secondary_cuts_sha256"] != digest(DEST / "secondary_cuts_locked.json"):
            raise RuntimeError("Completed secondary analysis dependencies changed")
        for name,checksum in saved["outputs_sha256"].items():
            if digest(DEST / name) != checksum:raise RuntimeError("Secondary analysis output changed")
        print("Verified completed secondary analysis; not recomputed")
        return
    report={"evaluated_at_utc":utc(),"secondary_only":True,"bootstrap_samples":2000,
        "test_features_sha256":digest(directory / "features.csv"),"secondary_cuts_sha256":digest(DEST / "secondary_cuts_locked.json"),
        "photographic_reconstruction_labels":"Input-source provenance; reconstructed negatives contain generated pixels.",
        "primary_cuts_outside_calibrated_family":"Core thresholds applied to extra conditions are diagnostic and have no covered-family false-positive guarantee.","models":{}}
    rows=[]
    for di,method in enumerate(f.METHODS):
        for bi,backbone in enumerate(f.BACKBONES):
            key=f"{method}/{backbone}";data=f.view(frame,method,backbone)
            report["models"][key]={}
            for policy_name,policies in (("core_cuts_diagnostic",primary["origin_policies"][key]),("attack_family_source_cuts",cuts["policies"][key])):
                result,scores,decisions=f.evaluate_scope(data,models[key],policies,2000,f.SEED+100*di+bi)
                result.pop("primary_assessment",None)
                result["interpretation"]="Secondary attack stress test; no primary hypothesis selection or universal guarantee"
                report["models"][key][policy_name]=result
                for name in models[key]:
                    meta=data["meta"][["image_id","source_id","collection","label","watermark_method","embedded_payload","condition"]].copy()
                    meta["detector"]=method;meta["backbone"]=backbone;meta["system"]=name;meta["policy"]=policy_name
                    meta["score"]=scores[name];meta["decision"]=decisions[name].astype(int);rows.append(meta)
    report["verification_at_core_cuts"]=f.verification_report(frame,primary)
    write(DEST / "results.json",f.finite_json(report))
    pd.concat(rows,ignore_index=True).to_csv(DEST / "predictions.csv",index=False)
    manifest=pd.read_csv(directory / "manifest.csv",keep_default_na=False)
    manifest.groupby(["condition","watermark_method"])[["PSNR_dB","SSIM"]].agg(["mean","std","min","max"]).to_csv(DEST / "quality.csv")
    write(DEST / "complete.json",{"utc":utc(),"script_sha256":digest(__file__),"test_features_sha256":digest(directory / "features.csv"),
        "secondary_cuts_sha256":digest(DEST / "secondary_cuts_locked.json"),"outputs_sha256":{p:digest(DEST / p) for p in ("results.json","predictions.csv","quality.csv")},
        "primary_model_or_threshold_retuned":False})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("calibrate","evaluate"),required=True);a=p.parse_args()
    calibrate() if a.stage=="calibrate" else evaluate()
