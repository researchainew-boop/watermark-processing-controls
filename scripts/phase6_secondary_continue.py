"""Run fixed extra-attack stages after primary and native GPU work completes."""
from __future__ import annotations
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def main():
    while not (ROOT / "outputs/phase6/native_baseline/metrics.json").exists():time.sleep(2)
    complete=ROOT / "outputs/phase6/calibration_attacks/features_complete.json"
    if complete.exists():
        saved=json.loads(complete.read_text())
        manifest=ROOT / "outputs/phase6/calibration_attacks/manifest.csv"
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=saved['manifest_sha256']:
            raise RuntimeError('Completed calibration manifest changed; preserve receipts and repair exact bytes before resume')
    else:
        run("scripts/phase6_runtime.py","--stage","extra","--role","calibration")
    run("scripts/phase6_features.py","--stage","calibration","--manifest","outputs/phase6/calibration_attacks/manifest.csv",
        "--destination","outputs/phase6/calibration_attacks","--batch-size","12")
    run("scripts/phase6_attack_reporting.py","--stage","calibrate")
    derived=ROOT / "outputs/phase6/test_attacks/derivation_complete.json"
    if derived.exists():
        saved=json.loads(derived.read_text())
        manifest=ROOT / "outputs/phase6/test_attacks/manifest.csv"
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=saved['manifest_sha256']:
            raise RuntimeError('Completed test attack manifest changed')
    else:
        run("scripts/phase6_quarantine_uncommitted_attacks.py")
        run("scripts/phase6_runtime.py","--stage","extra","--role","test")
    run("scripts/phase6_features.py","--stage","test","--manifest","outputs/phase6/test_attacks/manifest.csv",
        "--destination","outputs/phase6/test_attacks","--batch-size","12")
    run("scripts/phase6_attack_reporting.py","--stage","evaluate")
    run("scripts/check_phase6_submission.py")

if __name__=="__main__":main()
