"""Exact source-level intervals from saved verification outputs; no inference.

The event is that every evaluated payload of one input source verifies under
one condition using that method's fixed rule. This differs from mean output
verification. It avoids interpreting an all-zero/all-one empirical bootstrap
interval as proof of a zero/one population rate.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/extension/watermarks"


def exact_interval(successes, sources):
    return [0.0 if successes == 0 else float(beta.ppf(.025, successes, sources - successes + 1)),
            1.0 if successes == sources else float(beta.ppf(.975, successes + 1, sources - successes))]


def main():
    path = OUT / "measurements.csv"
    frame = pd.read_csv(path, keep_default_na=False, dtype={"payload": str, "source_id": str, "cover_id": str})
    if frame.duplicated(["cover_id", "condition", "detector"]).any():
        raise ValueError("Duplicate saved detector/cover/condition rows")
    covers = pd.read_csv(OUT / "base_covers.csv", keep_default_na=False, dtype={"payload": str})
    local = pd.read_csv(OUT / "attack_manifest.csv", keep_default_na=False)
    regen = pd.read_csv(ROOT / "outputs/extension/diffusion/manifest.csv", keep_default_na=False)
    expected = set(zip(pd.concat([local, regen]).cover_id, pd.concat([local, regen]).condition))
    actual = set(zip(frame.cover_id, frame.condition))
    if actual != expected or len(frame) != 2 * len(expected):
        raise ValueError("Saved measurements must contain both decoders for every completed local and regeneration output")
    summaries = []
    for (condition, detector), block in frame.groupby(["condition", "detector"], sort=True):
        for origin in [0, 1]:
            relevant = block.loc[(block.input_origin_label == origin) &
                                  block.watermark_method.eq(detector if origin else "unmarked")]
            if relevant.empty:
                continue
            grouped = relevant.groupby("source_id")
            source_verified = grouped.verified.min().astype(int)
            counts = grouped.size()
            if not len(source_verified) == 24 or counts.nunique() != 1:
                raise ValueError("Expected 24 distinct input sources and equal fixed payload counts per condition/origin")
            payloads = int(counts.iloc[0])
            if payloads not in [1, 3]:
                raise ValueError("Expected one regeneration payload or three edit payloads per marked source; real controls use one")
            successes, n = int(source_verified.sum()), len(source_verified)
            # Confirm all sources and declared payloads come from the saved cover table.
            selected = covers.loc[covers.cover_id.isin(relevant.cover_id)]
            if set(selected.source_id) != set(relevant.source_id):
                raise ValueError("Measurement source identities differ from cover table")
            summaries.append({"condition": condition, "detector": detector, "generator": "all",
                "input_origin_label": origin, "n_sources": n, "payloads_evaluated_per_source": payloads,
                "all_evaluated_payloads_verified_sources": successes,
                "all_evaluated_payloads_verification_rate": successes / n,
                "ci95_exact_clopper_pearson": exact_interval(successes, n),
                "verification_rule": "agreement >=13/16" if detector == "dwt_dct" else "exact expected 61-bit data message plus valid schema/ECC",
                "scope": "Input-source provenance and this fixed condition; real unmarked controls check only the original expected string" if origin == 0 else "All fixed payloads evaluated for this source and condition verify"})
    result = {"measurements_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sampling_unit": "distinct input source", "generator": "all", "source_count_per_origin": 24,
        "interval_target": "Probability all evaluated payloads verify on one source under a fixed condition; differs from mean per-output verification rate",
        "interval_method": "Two-sided 95% exact Clopper-Pearson for the source-level binary event",
        "caution": "Conditional on the chosen payloads, seed, methods, and exploratory sample; collection-balanced source selection is not a population probability sample. Distinct sources are treated as independent with a common success probability for this descriptive interval.",
        "inference_or_threshold_tuning_performed": False, "measurements": summaries}
    target = OUT / "source_verification_intervals.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved", len(summaries), "source-level intervals to", target, flush=True)


if __name__ == "__main__":
    main()
