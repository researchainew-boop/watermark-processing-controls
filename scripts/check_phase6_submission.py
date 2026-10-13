"""Read-only Phase6 integrity, saved-decision arithmetic and paper checks."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import phase6_fusion as f
from phase6_features import ROOT,OUT,digest,check_final_lock,write

def require(condition,message):
    if not condition:raise AssertionError(message)

def read(path):return json.loads(path.read_text())

def verify_receipt(folder,filename):
    receipt=read(folder / filename)
    for name,checksum in receipt.get("outputs_sha256",{}).items():
        require(digest(folder / name)==checksum,f"Changed receipt output: {folder.name}/{name}")
    return receipt

def check_phase6(require_attacks=True,require_native=True):
    root_hash=check_final_lock("test")
    lock=read(OUT / "frozen_protocol.json")
    folder=ROOT / lock["learner_output"]
    f.frozen(folder)
    calibration=verify_receipt(folder,"calibration_complete.json")
    result_folder=folder / "evaluations/fresh_test"
    complete=verify_receipt(result_folder,"complete.json")
    report=read(result_folder / "results.json")
    feature=read(OUT / "test/features_complete.json")
    require(feature["protocol_sha256_before_scores"]==root_hash,"Test feature protocol mismatch")
    require(feature["features_sha256"]==digest(OUT / "test/features.csv"),"Test features changed")
    require(complete["test_features_sha256"]==feature["features_sha256"],"Evaluation feature mismatch")
    require(complete["bootstrap_samples"]==report["bootstrap_samples"]==2000,"Primary draws changed")
    require(report["confirmatory_assessment"]["coverage"]["valid"],"Incomplete confirmatory sample")
    source=read(OUT / "test_sources_locked.json")
    require(source["source_manifest_sha256"]==digest(OUT / "data_inventory/primary_source_manifest.csv"),"Source inventory changed")
    inventory=pd.read_csv(OUT / "data_inventory/primary_source_manifest.csv",keep_default_na=False)
    expected={("calibration",c):400 for c in ("coco","openimages")}
    expected.update({("test",c):200 for c in ("coco","openimages","ADM","BigGAN","VQDM")})
    require(inventory.groupby(["role","collection"]).size().to_dict()==expected,"Source quotas changed")
    require(inventory.source_id.nunique()==len(inventory)==1800,"Duplicate source identities")
    dev=pd.read_csv(folder / "development_sources.csv",keep_default_na=False)
    require(not set(dev.source_id)&set(inventory.source_id),"Fresh source/development overlap")
    timestamps=[lock["frozen_at_utc"],calibration["frozen_utc"],read(OUT / "test/inference_started.json")["utc"],complete["completed_utc"]]
    require(all(datetime.fromisoformat(a)<=datetime.fromisoformat(b) for a,b in zip(timestamps,timestamps[1:])),"Scoring/calibration chronology failure")
    guard=read(OUT / "confirmatory_command_v2_locked.json")
    require(guard["wrapper_sha256"]==digest(ROOT / "scripts/phase6_confirm.py") and guard["bootstrap_samples"]==2000,"Command guard changed")
    require(datetime.fromisoformat(guard["locked_at_utc"])<datetime.fromisoformat(timestamps[2]),"Command sealed after test scores")
    frame=pd.read_csv(result_folder / "predictions.csv",keep_default_na=False,dtype={"embedded_payload":str})
    primary=frame.loc[frame.detector.eq("trustmark")&frame.backbone.eq("resnet")]
    systems=report["models"]["trustmark/resnet"]["primary_common"]["systems"]
    for name,block in primary.groupby("system"):
        require(not block.image_id.duplicated().any(),"Duplicate saved decisions")
        masks={"marked_tpr":block.label.eq(1)&block.watermark_method.eq("trustmark"),
               "unmarked_tpr":block.label.eq(1)&block.watermark_method.eq("unmarked"),
               "real_image_fpr":block.label.eq(0)}
        for metric,mask in masks.items():
            value=block.loc[mask].groupby("source_id").decision.mean().mean()
            require(abs(value-systems[name][metric])<1e-12,f"Saved {name}/{metric} arithmetic mismatch")
        fp=block.loc[block.label.eq(0)].groupby("source_id").decision.max().mean()
        require(abs(fp-systems[name]["real_source_fpr"])<1e-12,"Source FPR arithmetic mismatch")
    base=primary.loc[primary.system.eq("probe")].set_index("image_id")
    new=primary.loc[primary.system.eq("primary")].set_index("image_id")
    unmarked=base.label.eq(1)&base.watermark_method.eq("unmarked")
    harms=(base.loc[unmarked,"decision"].astype(bool)&~new.loc[base.index[unmarked],"decision"].astype(bool))
    events=harms.groupby(base.loc[unmarked,"source_id"]).max()
    conclusion=report["confirmatory_assessment"]
    require(int(events.sum())==conclusion["independent_source_harm_assessment"]["source_any_harm_events"],"Any-view harm event mismatch")
    require(len(events)==600,"Wrong harm source denominator")
    if not events.any():
        bound=-np.expm1(np.log(.05)/600)
        require(abs(bound-conclusion["independent_source_harm_assessment"]["pooled_zero_event_upper95"])<1e-14,"Zero-harm risk bound mismatch")
    require(conclusion["strongest_baseline_selected_on_tune"]=="old_capped_positive","Primary comparator changed")
    selected=read(folder / "selection.json")["trustmark/resnet"]["strongest_baseline"]
    require(selected==conclusion["strongest_baseline_selected_on_tune"],"Result comparator differs from tune selection")
    statuses={"primary_complete":True,"core_calibration_sources":800,"test_sources":1000,"test_AI_sources":600,
              "source_disjoint_roles":True,"primary_models_and_cuts_frozen":True,"core_cut_chronology_verified":True,
              "primary_saved_decision_rates_recomputed":True,"primary_criteria_status":conclusion["candidate_status"],
              "primary_claim_succeeded":all(conclusion[k] for k in ("marked_gain_supported","gain_vs_tune_selected_baseline_supported","unmarked_noninferiority_supported","realized_source_fpr_within_nominal")),
              "acceptance_or_novelty_rating_guaranteed":False}
    if require_attacks:
        attack=verify_receipt(OUT / "attack_analysis","complete.json")
        require(attack["primary_model_or_threshold_retuned"] is False,"Attack report retuned primary")
        cuts=read(OUT / "attack_analysis/secondary_cuts_locked.json")
        start=read(OUT / "test_attacks/inference_started.json")
        require(datetime.fromisoformat(cuts["locked_at_utc"])<datetime.fromisoformat(start["utc"]),"Secondary cuts frozen after scores")
        require(digest(OUT / "test_attacks/features.csv")==attack["test_features_sha256"],"Attack feature mismatch")
        statuses["secondary_attacks_complete"]=True
    if require_native:
        native=OUT / "native_baseline"
        require((native / "metrics.json").exists(),"Native baseline metrics missing")
        own=read(native / "thresholds_locked.json");start=read(native / "test_started.json")
        require(datetime.fromisoformat(own["locked_at_utc"])<datetime.fromisoformat(start["started_at_utc"]),"Native baseline threshold chronology failure")
        require(read(native / "test_complete.json")["predictions_sha256"]==digest(native / "test_predictions.csv"),"Native score identity mismatch")
        statuses["published_native_detector_complete"]=True
    if require_attacks and require_native:
        extra=OUT / "additional_baselines"
        extra_complete=verify_receipt(extra,"complete.json")
        extra_cut=read(extra / "cuts_locked.json")
        extra_cal=read(extra / "calibration_complete.json")
        require(extra_complete["test_features_sha256"]==feature["features_sha256"],"Additional comparators use different test features")
        require(extra_cal["cuts_sha256"]==digest(extra / "cuts_locked.json"),"Additional comparator cuts changed")
        require(extra_cut["inputs"]["calibration_features_sha256"]==calibration["calibration_features_sha256"],"Additional comparators use different calibration negatives")
        require(datetime.fromisoformat(extra_cut["locked_at_utc"])<datetime.fromisoformat(timestamps[2]),"Additional comparators added after test scores")
        require(extra_complete["primary_retuned"] is False,"Additional comparators retuned primary")
        statuses["additional_simple_baselines_complete"]=True
        statuses["additional_comparators_sealed_before_test_scores"]=True
    statuses.update({"checked_at_utc":datetime.now(timezone.utc).isoformat(),"root_protocol_sha256":root_hash,
                    "fresh_evaluation_receipt_sha256":digest(result_folder / "complete.json"),"all_required_Phase6_evidence_complete":bool(require_attacks and require_native)})
    return statuses

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--core-only",action="store_true");p.add_argument("--manuscript",action="store_true")
    p.add_argument("--pages",type=int,default=5);p.add_argument("--references",type=int,default=20);p.add_argument("--visual-reviewed",action="store_true")
    p.add_argument("--claims-reviewed",action="store_true",help="Set only after reviewing the finished manuscript against saved outcomes, including failures")
    a=p.parse_args();report=check_phase6(not a.core_only,not a.core_only)
    if a.manuscript:
        from check_phase5_submission import check_manuscript
        report.update(check_manuscript(a.pages,a.references,a.visual_reviewed))
        report["truthful_completed_frozen_results"]=bool(a.claims_reviewed)
        report["reporting_provenance_sha256"]=digest(OUT / "presentation/reporting_provenance.json")
    write(OUT / ("submission_check.json" if a.manuscript else "evidence_check.json"),report)
    print(json.dumps(report,indent=2))
