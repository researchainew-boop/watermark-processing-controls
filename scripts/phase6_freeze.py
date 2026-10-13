"""Separate immutable method/policy and final-source receipts before scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from phase6_features import ROOT,OUT,digest,write,utc,check_final_lock

LEARNER = OUT / "fusion_v2"


def freeze_method():
    target=OUT/"frozen_protocol.json"
    if target.exists():
        check_final_lock("calibration")
        print("Existing method/policy lock verified")
        return
    if any((OUT/p/"inference_started.json").exists() for p in ["calibration","test"]):
        raise RuntimeError("Cannot freeze method after fresh scores")
    fit=json.loads((LEARNER/"fit_complete.json").read_text())
    if fit["script_sha256"]!=digest(ROOT/"scripts/phase6_fusion.py"):
        raise ValueError("Fitted method code changed")
    for name,sha in fit["outputs_sha256"].items():
        if digest(LEARNER/name)!=sha:raise ValueError("Fitted artifact changed")
    paths=[ROOT/p for p in ["docs/phase6_evaluation_plan.md","scripts/phase6_features.py","scripts/phase6_runtime.py",
        "scripts/phase6_freeze.py","scripts/phase6_fusion.py","scripts/phase6_attacks.py","scripts/phase6_native_baseline.py",
        "scripts/benchmark_watermarks.py","scripts/evaluate_encoding_controls.py","scripts/phase5_fair_watermarks.py",
        "src/aiwstudy/watermark.py","src/aiwstudy/attacks.py","src/aiwstudy/models.py",
        "outputs/phase6/development/preparation.json","outputs/phase6/development/features.csv",
        "outputs/phase6/data_inventory/sampling_protocol.json","outputs/phase6/data_inventory/calibration_source_manifest.csv",
        "outputs/phase6/native_baseline/plan.json","outputs/conference_clean/resnet_full.npz","outputs/conference_clean/clip_full.npz",
        ".tools/extension-assets/ViT-L-14.pt",".tools/extension-assets/univfd_fc_weights.pth",".tools/extension-assets/official_clip_model.py"]]
    paths += [LEARNER/n for n in ["fit_complete.json",*fit["outputs_sha256"]]]
    write(target,{"frozen_at_utc":utc(),"learner_output":LEARNER.relative_to(ROOT).as_posix(),
        "primary":"TrustMark+ResNet;marked TPR gains over probe and strongest tune-feasible origin-fusion baseline;unmarked loss<=.01",
        "source_FPR_target":.05,"registered_payloads":json.loads((LEARNER/"protocol.json").read_text())["registered_queries"],
        "calibration_sources":800,"test_AI_sources_per_family":200,"test_real_sources_per_collection":200,
        "families":["BigGAN","ADM","VQDM"],"primary_conditions":["identity","jpeg50","crop","lowres32"],
        "initial_hypothesis_and_models_locked_before_fresh_scores":True,"no_acceptance_or_novelty_rating_guarantee":True,
        "input_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in paths}})
    print("METHOD LOCKED",digest(target),flush=True)


def freeze_test_sources():
    check_final_lock("calibration")
    target=OUT/"test_sources_locked.json"
    path=OUT/"data_inventory/primary_source_manifest.csv"
    frame=pd.read_csv(path,keep_default_na=False)
    assert len(frame)==frame.source_id.nunique()==1800
    counts=frame.groupby(["role","collection"]).size().to_dict()
    expected={("calibration",c):400 for c in ["coco","openimages"]}
    expected.update({("test",c):200 for c in ["coco","openimages","BigGAN","ADM","VQDM"]})
    if counts!=expected:raise ValueError("Final source quotas changed")
    calibration=pd.read_csv(OUT/"data_inventory/calibration_source_manifest.csv",keep_default_na=False)
    selected=frame.loc[frame.role.eq('calibration')]
    for key in ["source_id","base_path","output_sha256"]:
        if dict(zip(selected.source_id,selected[key]))!=dict(zip(calibration.source_id,calibration[key])):
            raise ValueError("Final calibration records changed")
    if target.exists():
        receipt=json.loads(target.read_text())
        if receipt["source_manifest_sha256"]!=digest(path):raise ValueError("Final sources changed")
        return
    if (OUT/"test/inference_started.json").exists():raise ValueError("Cannot lock sources after final scoring")
    write(target,{"locked_at_utc":utc(),"protocol_sha256":digest(OUT/"frozen_protocol.json"),
        "source_manifest_sha256":digest(path),"source_count":len(frame),"test_scoring_started":False,
        "counts":[{"role":r,"collection":c,"n":int(n)} for (r,c),n in counts.items()],
        "selection":"previously frozen hash-order sampling/eligibility rules;no model scores used"})
    print("FINAL SOURCES LOCKED",digest(target),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['method','test-sources']);a=p.parse_args()
    freeze_method() if a.stage=='method' else freeze_test_sources()
