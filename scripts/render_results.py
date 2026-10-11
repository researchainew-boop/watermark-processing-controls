"""Render verified metrics.json into LaTeX rows without editing the manuscript."""

import argparse
import json
from pathlib import Path


def _row(name: str, values: dict) -> str:
    low, high = values["roc_auc_ci"]
    return (
        f"{name} & {values['roc_auc']:.3f} "
        f"[{low:.3f}, {high:.3f}] & "
        f"{values['balanced_accuracy']:.3f} & {values['f1']:.3f} \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="?", default=Path("outputs/metrics.json"))
    args = parser.parse_args()
    with args.metrics.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    print("% Binary systems")
    for name in ("active", "passive", "fusion"):
        print(_row(name.title(), report[name]))

    residual = report.get("residual_after_payload_death", {})
    if residual:
        print("% Residual after payload death")
        print(f"mean bit accuracy & {residual['mean_bit_accuracy']:.3f} \\\\")
        for name, values in residual.get("systems", {}).items():
            print(f"{name} residual AUC & {values['roc_auc']:.3f} \\\\")

    humanify = report.get("humanify")
    if humanify:
        print("% Humanify detector (step 3b)")
        print(_row("Humanify", humanify))
        print(
            "TPR on S-A & "
            f"{humanify.get('true_positive_on_stripped_ai', float('nan')):.3f} & "
            "FPR on R-T & "
            f"{humanify.get('false_positive_on_matched_real', float('nan')):.3f} \\\\"
        )

    shortcut = report.get("artifact_shortcut", {})
    if shortcut:
        print("% Artifact shortcut FPR(R-T)-FPR(R-C)")
        for name, values in shortcut.items():
            delta = values.get("delta_fpr_rt_minus_rc", float("nan"))
            spoof = values.get("spoof_accept_rate", float("nan"))
            print(f"{name} & {delta:.3f} & {spoof:.3f} \\\\")

    decision = report.get("decision", {})
    if decision:
        print("% WAVES")
        print(
            f"coverage & {decision['coverage']:.3f} & "
            f"decided acc & {decision['balanced_accuracy_decided']:.3f} \\\\"
        )
        art = decision.get("artifact_shortcut", {})
        print(
            f"WAVES shortcut & {art.get('delta_positive_rate_rt_minus_rc', float('nan')):.3f} & "
            f"spoof accept & {art.get('spoof_accept_rate', float('nan')):.3f} \\\\"
        )


if __name__ == "__main__":
    main()
