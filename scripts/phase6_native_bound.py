"""Supplemental source/threshold guard for the unchanged Phase6 UFD driver.

Only ``validate_lock`` is replaced. Preprocessing, model loading, inference,
calibration cuts, and paired statistics remain the original function objects.
The original root lock and original native driver are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import phase6_native_baseline as baseline

ROOT = baseline.ROOT
WRAPPER = Path(__file__).resolve()
GUARD = baseline.OUT / "guard_locked.json"
SOURCE_RECEIPT = ROOT / "outputs/phase6/test_sources_locked.json"
EXPECTED_ROOT_SHA = "1e7a6d425f7ee4356a2d010067c6058ad3d90497c4715512a8b298eb913302f8"
EXPECTED_SOURCE_RECEIPT_SHA = "665924683bc9fe74ae4bd2e0e992d070d45c97bbe7beda892785b4deb4954474"


def project_file(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("A guard dependency leaves the project directory")
    return path


def source_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_upstream():
    lock_path = baseline.LOCK
    if baseline.digest(lock_path) != EXPECTED_ROOT_SHA:
        raise ValueError("The authorized root protocol changed")
    if baseline.digest(SOURCE_RECEIPT) != EXPECTED_SOURCE_RECEIPT_SHA:
        raise ValueError("The separately frozen source receipt changed")
    lock = baseline.read(lock_path)
    binding = baseline.read(SOURCE_RECEIPT)
    if binding["protocol_sha256"] != baseline.digest(lock_path):
        raise ValueError("The source receipt belongs to another root protocol")
    if binding["source_manifest_sha256"] != baseline.digest(baseline.SOURCES):
        raise ValueError("The final source manifest changed")
    if binding["source_count"] != 1800 or binding["test_scoring_started"] is not False:
        raise ValueError("The source receipt was not fixed before test scoring")
    required = ["scripts/phase6_native_baseline.py", "outputs/phase6/native_baseline/plan.json", *baseline.ASSET_HASHES]
    for relative in required:
        if relative not in lock["input_sha256"]:
            raise ValueError(f"Original baseline input missing from root lock: {relative}")
    for relative, expected in lock["input_sha256"].items():
        if baseline.digest(project_file(relative)) != expected:
            raise ValueError(f"Frozen root input changed: {relative}")
    saved_plan = baseline.plan()
    for relative, checksum in saved_plan["model_asset_sha256"].items():
        if baseline.digest(project_file(relative)) != checksum:
            raise ValueError(f"Official detector asset changed: {relative}")
    learner = project_file(lock["learner_output"])
    expected_fit = lock["input_sha256"].get((learner/"fit_complete.json").relative_to(ROOT).as_posix())
    if expected_fit is None or baseline.digest(learner/"fit_complete.json") != expected_fit:
        raise ValueError("The root-selected learner fit is not locked")
    # Compare the exact immutable photographic calibration metadata with the
    # corresponding portion of the finalized combined inventory.
    full = source_rows(baseline.SOURCES)
    calibration_path = ROOT / "outputs/phase6/data_inventory/calibration_source_manifest.csv"
    calibration = source_rows(calibration_path)
    expected = {(r,c):n for r, groups in saved_plan["required_source_counts"].items() for c,n in groups.items()}
    counts = {}
    ids = set()
    for row in full:
        ids.add(row["source_id"])
        key = (row["role"],row["collection"])
        counts[key] = counts.get(key,0)+1
        label = int(row["label"])
        if label != int(row["collection"] not in ["coco","openimages"]):
            raise ValueError("Source provenance label changed")
        if row["role"] != row["split"]:
            raise ValueError("Source split differs from its fixed role")
    if len(full) != 1800 or len(ids) != 1800 or counts != expected:
        raise ValueError("Source quotas or uniqueness changed")
    registered = {r["source_id"]:r for r in full if r["role"] == "calibration"}
    if len(calibration) != 800 or len(registered) != 800 or {r["source_id"] for r in calibration} != set(registered):
        raise ValueError("Independent calibration identities changed")
    for row in calibration:
        if any(str(registered[row["source_id"]].get(k,"")) != str(v) for k,v in row.items()):
            raise ValueError("Independent calibration metadata changed")
    return lock, learner


def guard_inputs(lock, learner):
    inputs = dict(lock["input_sha256"])
    extra = [WRAPPER, baseline.LOCK, SOURCE_RECEIPT, baseline.SOURCES,
             Path(baseline.official.__file__), learner/"fit_complete.json"]
    for path in extra:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        inputs[relative] = baseline.digest(path)
    return inputs


def seal():
    lock, learner = verify_upstream()
    inputs = guard_inputs(lock,learner)
    if GUARD.exists():
        old = baseline.read(GUARD)
        if old["input_sha256"] != inputs:
            raise ValueError("Existing supplemental native guard differs")
        return old
    for name in ["calibration_started.json", "test_started.json", "calibration_predictions.csv", "test_predictions.csv"]:
        if (baseline.OUT/name).exists():
            raise RuntimeError("Cannot introduce a new native guard after native inference began")
    for subdir in ["calibration_blocks","test_blocks"]:
        if (baseline.OUT/subdir).exists() and any((baseline.OUT/subdir).glob("*.npz")):
            raise RuntimeError("Cannot introduce a new native guard after native score blocks exist")
    value = {"sealed_at_utc":baseline.utc(), "implementation_repair":
        "Replace only stale root/source/threshold integration guard; bind immutable separate source receipt and dynamic root learner_output",
        "root_protocol_sha256":baseline.digest(baseline.LOCK),
        "source_receipt_sha256":baseline.digest(SOURCE_RECEIPT),
        "source_manifest_sha256":baseline.digest(baseline.SOURCES),
        "learner_output":learner.relative_to(ROOT).as_posix(),
        "native_detector_scores_observed_before_seal":False,
        "original_native_driver_preserved":True,
        "detector_preprocessing_scoring_cut_and_statistics_functions_changed":False,
        "input_sha256":inputs}
    baseline.write(GUARD,value)
    return value


def verify_main_thresholds(learner, lock):
    cut = learner/"thresholds_locked.json"
    complete = learner/"calibration_complete.json"
    if not cut.exists() or not complete.exists():
        raise RuntimeError("The root-selected learner thresholds must be locked before native test scoring")
    main = baseline.read(cut)
    receipt = baseline.read(complete)
    if main["test_read"] is not False or receipt["final_test_read"] is not False:
        raise ValueError("The main learner was not calibrated before test inspection")
    if main["fit_complete_sha256"] != baseline.digest(learner/"fit_complete.json"):
        raise ValueError("Main thresholds reference another selected fit")
    if main["calibration_json_sha256"] != baseline.digest(learner/"calibration.json"):
        raise ValueError("Main calibration decisions changed")
    for name, checksum in receipt["outputs_sha256"].items():
        if baseline.digest(learner/name) != checksum:
            raise ValueError("Main calibration receipt output changed")
    if receipt["fit_complete_sha256"] != baseline.digest(learner/"fit_complete.json"):
        raise ValueError("Main calibration receipt references another fit")
    if receipt["script_sha256"] != lock["input_sha256"]["scripts/phase6_fusion.py"]:
        raise ValueError("Main calibration used another frozen implementation")
    features = ROOT/"outputs/phase6/calibration/features.csv"
    feature_receipt = baseline.read(features.with_name("features_complete.json"))
    if feature_receipt["protocol_sha256_before_scores"] != baseline.digest(baseline.LOCK):
        raise ValueError("Main calibration scores belong to another root protocol")
    if receipt["calibration_features_sha256"] != baseline.digest(features) or feature_receipt["features_sha256"] != baseline.digest(features):
        raise ValueError("Main calibration score identities changed")


def validate_lock(role=None):
    lock, learner = verify_upstream()
    if not GUARD.exists():
        raise RuntimeError("Seal the supplemental native guard before any native stage")
    saved = baseline.read(GUARD)
    if saved["input_sha256"] != guard_inputs(lock,learner):
        raise ValueError("Supplemental native guard dependencies changed")
    if role == "test":
        cuts = baseline.OUT/"thresholds_locked.json"
        if not cuts.exists():
            raise RuntimeError("Native detector thresholds must be locked before its test scoring")
        own = baseline.read(cuts)
        if own["root_protocol_sha256"] != baseline.digest(baseline.LOCK):
            raise ValueError("Native thresholds belong to another root protocol")
        if baseline.digest(baseline.OUT/"calibration_predictions.csv") != own["calibration_predictions_sha256"]:
            raise ValueError("Native calibration predictions changed")
        verify_main_thresholds(learner,lock)
    return baseline.digest(baseline.LOCK)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True)
    args,_ = parser.parse_known_args()
    if args.stage == "seal":
        seal()
        print("Guard SHA256",baseline.digest(GUARD),flush=True)
        return
    baseline.validate_lock = validate_lock
    baseline.main()


if __name__ == "__main__": main()
