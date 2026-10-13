"""Bind the confirmatory command to the registered sample and 2000 draws."""
from __future__ import annotations
import argparse
import json
from types import SimpleNamespace
import pandas as pd
import phase6_fusion as fusion
from phase6_features import ROOT, OUT, digest, write, utc, check_final_lock

GUARD = OUT / "confirmatory_command_v2_locked.json"

def seal():
    protocol = check_final_lock("calibration")
    source = OUT / "test_sources_locked.json"
    receipt = json.loads(source.read_text())
    if receipt["protocol_sha256"] != protocol:
        raise RuntimeError("Source receipt differs from the method lock")
    inputs = {"root_protocol_sha256":protocol,"test_source_receipt_sha256":digest(source),
              "wrapper_sha256":digest(__file__),"learner_sha256":digest(fusion.__file__),
              "initial_guard_sha256":digest(OUT / "confirmatory_command.original.json"),
              "bootstrap_samples":2000,"evaluation_name":"fresh_test"}
    if GUARD.exists():
        saved = json.loads(GUARD.read_text())
        for key,value in inputs.items():
            if saved[key] != value:raise RuntimeError("Confirmatory command guard changed")
    else:
        if (OUT / "test/inference_started.json").exists():
            raise RuntimeError("Seal confirmatory command before fresh test scores")
        write(GUARD,{"locked_at_utc":utc(),**inputs,"changes_to_models_or_endpoints":False})
    return inputs

def evaluate():
    seal()
    check_final_lock("test")
    features = OUT / "test/features.csv"
    complete = json.loads((OUT / "test/features_complete.json").read_text())
    if complete["features_sha256"] != digest(features):raise RuntimeError("Completed features changed")
    if complete["manifest_sha256"] != digest(OUT / "test/manifest.csv") or complete["protocol_sha256_before_scores"] != digest(OUT / "frozen_protocol.json"):
        raise RuntimeError("Completed features are not bound to the registered inference run")
    if complete["driver_sha256"] != digest(ROOT / "scripts/phase6_features.py"):
        raise RuntimeError("Feature driver changed")
    frame = pd.read_csv(features,keep_default_na=False,dtype={"embedded_payload":str,"query_payload":str})
    inventory = pd.read_csv(OUT / "data_inventory/primary_source_manifest.csv",keep_default_na=False)
    expected = inventory.loc[inventory.role.eq("test")]
    if set(frame.source_id) != set(expected.source_id) or set(frame.role) != {"test"}:
        raise RuntimeError("Confirmatory feature sample differs from locked sources")
    if expected.loc[expected.label.eq(1)].groupby("collection").size().to_dict() != {"ADM":200,"BigGAN":200,"VQDM":200}:
        raise RuntimeError("Confirmatory generator quota changed")
    metadata = frame[["source_id","collection","label"]].drop_duplicates().sort_values("source_id").reset_index(drop=True)
    registered = expected[["source_id","collection","label"]].sort_values("source_id").reset_index(drop=True)
    if not metadata.equals(registered):raise RuntimeError("Confirmatory metadata differ from source registration")
    lock = json.loads((OUT / "frozen_protocol.json").read_text())
    fusion.evaluate(SimpleNamespace(output=ROOT/lock["learner_output"],features=features,
                    name="fresh_test",bootstrap_samples=2000,allow_unseen_conditions=False))

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("seal","evaluate"),required=True)
    a=p.parse_args();seal() if a.stage=="seal" else evaluate()
