"""Prospective secondary comparators sealed before the first final test score."""
from __future__ import annotations
import argparse,json
import numpy as np
from scipy.special import expit
import phase6_fusion as f
from phase6_features import ROOT,OUT,digest,write,utc,check_final_lock

DEST=OUT / "additional_baselines"

def score(data,model):
    if model["kind"]=="probe":return data["z"]
    if model["kind"]=="full_message_union":
        full=data["features"][:,:,f.FEATURES.index("full_match")].max(axis=1)>0
        return np.where(full,2.,expit(data["z"]))
    agreement=data["features"][:,:,f.FEATURES.index("agreement")]
    stability=data["features"][:,:,f.FEATURES.index("stability")]
    return data["z"]+model["bound"]*(np.maximum(2*agreement-1,0)*stability).max(axis=1)

def dependencies(folder):
    return {"script_sha256":digest(__file__),"root_protocol_sha256":digest(OUT / "frozen_protocol.json"),
            "fit_sha256":digest(folder / "fit_complete.json"),"development_features_sha256":digest(OUT / "development/features.csv"),
            "calibration_features_sha256":digest(OUT / "calibration/features.csv")}

def calibrate():
    check_final_lock("calibration")
    lock=json.loads((OUT / "frozen_protocol.json").read_text());folder=ROOT / lock["learner_output"]
    f.frozen(folder)
    path=DEST / "cuts_locked.json";receipt_path=DEST / "calibration_complete.json"
    dep=dependencies(folder)
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt["inputs"]!=dep or receipt["cuts_sha256"]!=digest(path):raise RuntimeError("Additional comparator lock changed")
        return
    if (OUT / "test/inference_started.json").exists():raise RuntimeError("Register secondary comparators before final test scores")
    dev,_=f.read_features(OUT / "development/features.csv",("fit","tune","constraint"),lock["registered_payloads"])
    cal,_=f.read_features(OUT / "calibration/features.csv",("calibration",),lock["registered_payloads"])
    systems={}
    for method in f.METHODS:
        for backbone in f.BACKBONES:
            key=f"{method}/{backbone}";tune=f.view(dev,method,backbone,"tune");data=f.view(cal,method,backbone)
            probe_cut=f.empirical_threshold(tune["meta"],tune["z"]);base=tune["z"]>=probe_cut
            candidates=[]
            for bound in f.CORRECTION_BOUNDS:
                model={"kind":"tuned_confidence","bound":bound}
                values=score(tune,model);cut=f.empirical_threshold(tune["meta"],values)
                rates=f.rates(tune,values,cut,base)
                feasible=bool(rates["unmarked_groups"] and min(r["tpr_minus_probe"] for r in rates["unmarked_groups"])>=-.01-1e-12 and rates["real_source_fpr"]<=.05+1e-12)
                candidates.append({"model":model,"feasible":feasible,"tuning_cut":cut,"tuning_rates":rates})
            eligible=[c for c in candidates if c["feasible"]]
            selected=max(eligible,key=lambda c:(c["tuning_rates"]["marked_tpr"],-c["model"]["bound"]))["model"] if eligible else {"kind":"probe"}
            systems[key]={}
            for name,model in (("full_message_union",{"kind":"full_message_union"}),("tuned_confidence",selected)):
                policy=f.conformal_cut(data["meta"],score(data,model))
                systems[key][name]={"model":model,"policy":policy,"tuning_candidates":candidates if name=="tuned_confidence" else []}
    write(path,f.finite_json({"locked_at_utc":utc(),"inputs":dep,"systems":systems,"test_read":False,"secondary_only":True,
        "reason":"Additional simple-comparator gap identified before any final test inference; original primary model, comparator and endpoints unchanged",
        "registered_rules":{"full_message_union":"score2 if any registered native full match, otherwise sigmoid(probe logit); independently calibrated source cut",
                            "tuned_confidence":"probe logit+B*max_query((2*agreement-1)_+*stability); B in same1,2,4,8,16 range, tuneonly with same harm/FPR constraints"}}))
    write(receipt_path,{"utc":utc(),"inputs":dep,"cuts_sha256":digest(path),"final_test_scores_observed":False})
    print("ADDITIONAL COMPARATORS FROZEN",flush=True)

def evaluate():
    check_final_lock("test")
    lock=json.loads((OUT / "frozen_protocol.json").read_text());folder=ROOT / lock["learner_output"]
    _,models,protocol=f.frozen(folder);cal=f.load_json(folder / "calibration.json")
    receipt=f.load_json(DEST / "calibration_complete.json");cuts=f.load_json(DEST / "cuts_locked.json")
    if receipt["inputs"]!=dependencies(folder) or receipt["cuts_sha256"]!=digest(DEST / "cuts_locked.json"):
        raise RuntimeError("Additional comparator dependencies changed")
    if f.load_json(OUT / "test/inference_started.json")["utc"]<=cuts["locked_at_utc"]:raise RuntimeError("Secondary comparators sealed after first scores")
    path=OUT / "test/features.csv";frame,_=f.read_features(path,("test",),protocol["registered_queries"])
    results={"secondary_only":True,"bootstrap_samples":2000,"original_primary_comparator_unchanged":True,"systems":{}}
    for di,method in enumerate(f.METHODS):
        for bi,backbone in enumerate(f.BACKBONES):
            key=f"{method}/{backbone}";data=f.view(frame,method,backbone)
            scores={n:f.score(data,models[key][n]) for n in ("probe","primary")}
            policies={n:cal["origin_policies"][key][n] for n in ("probe","primary")}
            for name,entry in cuts["systems"][key].items():scores[name]=score(data,entry["model"]);policies[name]=entry["policy"]
            decisions={n:v>(policies[n]["threshold"] if policies[n]["threshold"] is not None else np.inf) for n,v in scores.items()}
            boot=f.source_bootstrap(data,decisions,decisions["probe"],2000,f.SEED+100*di+bi)
            results["systems"][key]={n:{**f.rates(data,v,policies[n]["threshold"] if policies[n]["threshold"] is not None else np.inf,decisions["probe"],strict=True),
                "source_bootstrap":boot[n],"policy":policies[n]} for n,v in scores.items()}
    write(DEST / "results.json",f.finite_json(results))
    write(DEST / "complete.json",{"utc":utc(),"script_sha256":digest(__file__),"test_features_sha256":digest(path),
          "calibration_complete_sha256":digest(DEST / "calibration_complete.json"),"outputs_sha256":{"results.json":digest(DEST / "results.json")},"primary_retuned":False})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("calibrate","evaluate"),required=True);a=p.parse_args()
    calibrate() if a.stage=="calibrate" else evaluate()
