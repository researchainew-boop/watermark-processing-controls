"""Document score-free dependency/guard repairs without replacing method choices."""
from __future__ import annotations
import json
from phase6_features import ROOT, OUT, digest, write, utc, checkpoint_defaults, check_final_lock

INITIAL = "a0bb581898c76ccdd8181ca0c182285a18da75af140adf536103e0fe399f49ca"

def main():
    target = OUT / "frozen_protocol.json"
    if digest(target) != INITIAL:
        check_final_lock("calibration")
        print("Amended dependency lock already verified")
        return
    if any((OUT / p / "inference_started.json").exists() for p in ("calibration", "test")):
        raise RuntimeError("Dependency repair must precede every fresh score")
    original = target.read_bytes()
    lock = json.loads(original)
    archive = OUT / "frozen_protocol.original.json"
    archive.write_bytes(original)
    old_sampling = OUT / "data_inventory/sampling_protocol.root_frozen_original.json"
    assert digest(old_sampling) == lock["input_sha256"]["outputs/phase6/data_inventory/sampling_protocol.json"]
    before = json.loads(old_sampling.read_text())
    after = json.loads((OUT / "data_inventory/sampling_protocol.json").read_text())
    after.pop("score_separation")
    assert before == after
    expected = {"resnet":"f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                "clip":"e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"}
    actual = {name:digest(p) for name,p in zip(("resnet","clip"),checkpoint_defaults())}
    assert actual == expected
    receipt = OUT / "passive_dependencies_locked.json"
    write(receipt,{"locked_at_utc":utc(),"original_protocol_sha256":INITIAL,"before_any_fresh_scores":True,
        "backbone_sha256":actual,"resnet_provenance":"TorchVision IMAGENET1K_V1 resnet18-f37072fd.pth",
        "clip_provenance":"timm/vit_base_patch32_clip_224.openai revision a6f597a30f7b82c51704746581f9a4e41421e878",
        "development_identity_evidence":"Official ResNet V1 identity and prior anchor equivalence; CLIP full SHA matches Phase3/4/5 provenance. Legacy development blocks do not retrospectively certify their full ResNet SHA."})
    changed = ["scripts/phase6_features.py","docs/phase6_evaluation_plan.md","outputs/phase6/data_inventory/sampling_protocol.json"]
    for path in changed:
        lock["input_sha256"][path] = digest(ROOT / path)
    for path in ["scripts/phase6_dependency_repair.py","outputs/phase6/passive_dependencies_locked.json",
                 "outputs/phase6/frozen_protocol.original.json","outputs/phase6/data_inventory/sampling_protocol.root_frozen_original.json"]:
        lock["input_sha256"][path] = digest(ROOT / path)
    lock["score_free_amendment"] = {"utc":utc(),"original_protocol_sha256":INITIAL,
        "changes":["Seal actual passive backbone dependencies at final-stage entry, including cache reuse",
                   "Check supplied prospective manifest roles, IDs and metadata against registered sources",
                   "State existing implementation's realized source FPR<=.05 success criterion before outcomes",
                   "Accept only added score_separation sampling explanation; no scientific sampling changes"],
        "fitted_parameters_changed":False,"candidate_selection_changed":False,"sampling_changed":False,
        "fresh_detector_or_watermark_scores_observed":False}
    write(target,lock)
    check_final_lock("calibration")
    print("AMENDED BEFORE FRESH SCORES",digest(target),flush=True)

if __name__ == "__main__":main()
