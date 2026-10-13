"""Run fixed native UFD test only after primary GPU work and checks complete."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import phase6_native_baseline as native
import phase6_native_bound as guard

ROOT=Path(__file__).resolve().parents[1]


def main():
    complete=ROOT/"outputs/phase6/fusion_v2/evaluations/fresh_test/complete.json"
    evidence=ROOT/"outputs/phase6/evidence_check.json"
    print("Native continuation waiting for completed primary evaluation and arithmetic checks",flush=True)
    while True:
        if complete.exists() and evidence.exists():
            try:
                report=json.loads(evidence.read_text())
                if report.get("primary_complete") is True:break
            except (json.JSONDecodeError,OSError):
                pass  # A receipt may be in the middle of its atomic stage write.
        time.sleep(15)
    if report["root_protocol_sha256"]!=native.digest(native.LOCK):
        raise ValueError("Primary evidence belongs to a different root protocol")
    if report["fresh_evaluation_receipt_sha256"]!=native.digest(complete):
        raise ValueError("Primary completion receipt differs from its arithmetic checks")
    receipt=json.loads(complete.read_text())
    if receipt["source_count"]!=1000 or receipt["retuning"] is not False:
        raise ValueError("Incomplete or retuned primary test")
    for name,checksum in receipt["outputs_sha256"].items():
        if native.digest(complete.parent/name)!=checksum:
            raise ValueError("Primary checked output changed")
    guard.validate_lock("test")
    native.write(native.OUT/"continuation_trigger.json",{
        "triggered_at_utc":native.utc(),"controller_sha256":native.digest(Path(__file__)),
        "primary_evaluation_receipt_sha256":native.digest(complete),"primary_evidence_check_sha256":native.digest(evidence),
        "native_guard_sha256":native.digest(guard.GUARD),"native_thresholds_sha256":native.digest(native.OUT/"thresholds_locked.json"),
        "authorization":"Root authorized native test GPU only after primary evaluation and primary_complete arithmetic receipt",
        "native_test_or_score_selection_changed":False})
    print("Primary GPU work and checks complete; starting fixed native UFD test",flush=True)
    subprocess.run([sys.executable,"scripts/phase6_native_bound.py","--stage","test","--device","cuda","--batch-size","4"],cwd=ROOT,check=True)
    print("Native test GPU process exited; starting paired CPU analysis",flush=True)
    subprocess.run([sys.executable,"scripts/phase6_native_bound.py","--stage","analyze"],cwd=ROOT,check=True)
    print("Fixed native UFD test and analysis complete",flush=True)


if __name__=="__main__":main()
