"""Resume fixed primary stages after the already-running encoding completes."""
from __future__ import annotations
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def main():
    complete=ROOT / "outputs/phase6/test/encoding_complete.json"
    while not complete.exists():time.sleep(2)
    receipt=json.loads(complete.read_text())
    if receipt["sources"]!=1000 or receipt["covers"]!=4600:raise RuntimeError("Incomplete fixed test covers")
    run("scripts/phase6_runtime.py","--stage","core","--role","test")
    run("scripts/phase6_features.py","--stage","test","--manifest","outputs/phase6/test/manifest.csv",
        "--destination","outputs/phase6/test","--batch-size","12")
    run("scripts/phase6_confirm.py","--stage","evaluate")
    run("scripts/check_phase6_submission.py","--core-only")

if __name__=="__main__":main()
