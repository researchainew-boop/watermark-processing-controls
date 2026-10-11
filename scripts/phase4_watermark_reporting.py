"""Describe fixed-rule watermark false matches from saved decoder outputs.

No inference, score fitting, or test-derived threshold selection occurs here.
The separately saved analysis plan must exist before outcomes are read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/phase3/watermarks"
OUT = ROOT / "outputs/phase4/watermark_reporting"
PAYLOADS = ("1011010010110100", "0100101101001011", "1100101000110101")
DEFAULT = PAYLOADS[0]
METHODS = ("dwt_dct", "trustmark")
CONDITIONS = ("identity", "jpeg_s1", "jpeg_s2", "blur_s1", "blur_s2", "noise_s1",
              "crop_s1", "lowres_s2", "regenerate_s2", "spectral_s1",
              "noise_sigma6_seed20261009", "noise_sigma6_seed20261010", "noise_sigma6_seed20261011")
TABLE_CONDITIONS = ("identity", "jpeg_s2", "crop_s1", "lowres_s2", "regenerate_s2", "spectral_s1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Empty aggregate")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def accepts(row: dict, payload: str) -> bool:
    if row["detector"] == "dwt_dct":
        return sum(a == b for a, b in zip(row["payload_bits"], payload)) >= 13
    return (row["ecc_valid"] == "1" and row["decoded_schema"] == "1"
            and row["decoded_message"] == payload + "0" * 45)


def exact_interval(k: int, n: int) -> tuple[float, float]:
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k)))


def bootstrap_interval(values: np.ndarray, identity: str) -> tuple[float, float]:
    # Independent reproducible streams for each descriptive aggregate.
    seed = int(hashlib.sha256(f"20261010|{identity}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [.025, .975]))


def aggregate_binary(values: list[bool], **metadata) -> dict:
    k, n = sum(values), len(values)
    low, high = exact_interval(k, n)
    return {**metadata, "n_sources": n, "matches": k, "rate": k/n,
            "ci95_lower": low, "ci95_upper": high,
            "interval": "exact_two_sided_source_binomial"}


def main() -> None:
    plan_path = OUT / "analysis_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError("Save the prospective reporting plan before reading outcomes")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["payloads"] == list(PAYLOADS)
    input_names = ("measurements.csv", "base_covers.csv", "attack_manifest.csv", "selection.json", "analysis_provenance.json")
    input_hashes = {f"outputs/phase3/watermarks/{name}": sha(INPUT / name) for name in input_names}
    covers = read_csv(INPUT / "base_covers.csv")
    attacks = read_csv(INPUT / "attack_manifest.csv")
    frame = read_csv(INPUT / "measurements.csv")
    assert len(covers) == 1632 and len(attacks) == 21216 and len(frame) == 42432
    cover_by_id = {r["cover_id"]: r for r in covers}
    attack_by_id = {(r["cover_id"], r["condition"]): r for r in attacks}
    assert len(cover_by_id) == 1632 and len(attack_by_id) == 21216
    assert set(r["condition"] for r in attacks) == set(CONDITIONS)
    assert Counter(r["watermark_method"] for r in covers) == {"unmarked": 408, "dwt_dct": 612, "trustmark": 612}
    measurements = defaultdict(dict)
    for row in frame:
        key = (row["cover_id"], row["condition"])
        assert row["detector"] in METHODS and row["detector"] not in measurements[key]
        measurements[key][row["detector"]] = row
        assert len(row["payload_bits"]) == 16 and set(row["payload_bits"]) <= {"0", "1"}
        cover, attack = cover_by_id[key[0]], attack_by_id[key]
        for col in ("source_id", "input_origin_label", "watermark_method", "payload"):
            assert row[col] == cover[col], (key, col)
        assert row["output_file_sha256"] == attack["output_file_sha256"]
        payload = row["payload"] or DEFAULT
        assert int(row["verified"]) == int(accepts(row, payload))
        agreement = sum(a == b for a, b in zip(row["payload_bits"], payload))/16
        assert abs(float(row["payload_bit_accuracy"]) - agreement) < 1e-12
        if row["detector"] == "trustmark" and row["ecc_valid"] == "1" and row["decoded_schema"] == "1":
            assert len(row["decoded_message"]) == 61 and set(row["decoded_message"]) <= {"0", "1"}
    assert set(measurements) == set(attack_by_id)
    assert all(set(v) == set(METHODS) for v in measurements.values())
    assert all(sum(k[0] == cid for k in attack_by_id) == 13 for cid in cover_by_id)
    controls = {"photograph": [r for r in covers if r["watermark_method"] == "unmarked" and r["input_origin_label"] == "0"],
                "unmarked_ai": [r for r in covers if r["watermark_method"] == "unmarked" and r["input_origin_label"] == "1"]}
    assert all(len(v) == len({r["source_id"] for r in v}) == 204 for v in controls.values())
    per_condition, overall, by_key = [], [], {}
    for control, inventory in controls.items():
        inventory = sorted(inventory, key=lambda r: r["source_id"])
        for method in METHODS:
            for query in (*PAYLOADS, "any_of_three"):
                matrix = np.array([[any(accepts(measurements[(cover["cover_id"], condition)][method], p)
                                             for p in (PAYLOADS if query == "any_of_three" else (query,)))
                                    for condition in CONDITIONS] for cover in inventory], dtype=np.uint8)
                for j, condition in enumerate(CONDITIONS):
                    result = aggregate_binary(matrix[:, j].tolist(), control=control, detector=method,
                                              query=query, condition=condition)
                    per_condition.append(result)
                    by_key[(control, method, query, condition)] = result
                source_means = matrix.mean(axis=1)
                low, high = bootstrap_interval(source_means, f"{control}|{method}|{query}|overall")
                result = {"control": control, "detector": method, "query": query,
                          "condition": "overall_equal_condition_mean", "n_sources": 204,
                          "n_condition_outputs": int(matrix.size), "matches": int(matrix.sum()),
                          "rate": float(source_means.mean()), "ci95_lower": low, "ci95_upper": high,
                          "interval": "source_bootstrap_mean_over_13_conditions"}
                overall.append(result)
                by_key[(control, method, query, "overall_equal_condition_mean")] = result
                result = aggregate_binary(matrix.any(axis=1).tolist(), control=control, detector=method,
                                          query=query, condition="at_least_one_of_13_conditions")
                overall.append({**result, "n_condition_outputs": int(matrix.size)})
    cross_results = []
    for embedded in METHODS:
        decoder = next(m for m in METHODS if m != embedded)
        for condition in CONDITIONS:
            block = [r for r in frame if r["watermark_method"] == embedded and r["detector"] == decoder
                     and r["condition"] == condition]
            assert len(block) == 612
            for payload in PAYLOADS:
                chosen = [r for r in block if r["payload"] == payload]
                assert len(chosen) == 204 and len({r["source_id"] for r in chosen}) == 204
                cross_results.append(aggregate_binary([accepts(r, payload) for r in chosen],
                                      embedded_method=embedded, detector=decoder, query=payload, condition=condition))
    table = []
    for condition in (*TABLE_CONDITIONS, "overall_equal_condition_mean"):
        for method in METHODS:
            matching = [r for r in frame if r["detector"] == method and r["watermark_method"] == method
                        and (condition == "overall_equal_condition_mean" or r["condition"] == condition)]
            per_source = defaultdict(list)
            for r in matching:
                per_source[r["source_id"]].append(int(r["verified"]))
            assert len(per_source) == 204
            source_means = np.array([np.mean(v) for _, v in sorted(per_source.items())])
            low, high = bootstrap_interval(source_means, f"recovery|{method}|{condition}")
            entry = {"condition": condition, "detector": method, "n_sources_per_origin": 204,
                     "recovery_n_marked_outputs": len(matching), "recovery_matches": sum(int(r["verified"]) for r in matching),
                     "recovery_mean_over_payloads": float(source_means.mean()),
                     "recovery_ci95_source_lower": low, "recovery_ci95_source_upper": high}
            for control in controls:
                for query, label in ((DEFAULT, "default"), ("any_of_three", "any_three")):
                    result = by_key[(control, method, query, condition)]
                    for name in ("matches", "rate", "ci95_lower", "ci95_upper"):
                        entry[f"{control}_{label}_{name}"] = result[name]
            table.append(entry)
    # Illustrative uniform-bit calculation only; empirical image outputs are dependent/nonuniform.
    chance = sum(math.comb(16, k) for k in range(13, 17))/2**16
    any_chance = sum(any(sum(a == b for a, b in zip(f"{i:016b}", p)) >= 13 for p in PAYLOADS)
                     for i in range(2**16))/2**16
    write_csv(OUT / "control_per_condition.csv", per_condition)
    write_csv(OUT / "control_overall.csv", overall)
    write_csv(OUT / "cross_method_acceptance.csv", cross_results)
    write_csv(OUT / "table_ready.csv", table)
    write_json(OUT / "summary.json", {
        "analysis": "posthoc descriptive expected-payload verification under original fixed rules",
        "table_rows": table,
        "operating_point_comparison": {"matched_fpr": False, "roc_comparison": False,
            "reason": "No independent validation split for the 204-source learned-mark experiment. Exact TrustMark ECC/schema/message verification is a fixed boolean event; first-16-bit agreement is not the same criterion. No thresholds selected from test."},
        "rules": {"dwt_dct": "agreement>=13/16", "trustmark": "valid ECC; BCH5 schema=1; exact expected 61-bit message"},
        "uniform_independent_bit_illustration": {"dwt_single_payload_probability": chance,
                                                "dwt_any_three_payload_probability": any_chance,
                                                "not_empirical_or_security_guarantee": True},
        "large_regeneration_controls_included": False,
        "controls": {"photographs": "204 COCO sources; one unmarked transformed cover per condition",
                     "ai": "204 AI sources, 68 per collection; one unmarked transformed cover per condition"},
        "payload_queries_not_independent_samples": True,
        "interval_limit": "Exact binomial bounds assume common independent source success probabilities and describe this selected set. Source bootstrap preserves all repeated payload/condition outcomes but can be degenerate for zero events; per-condition exact bounds prevent interpreting zero observed as zero population risk."
    })
    write_json(OUT / "analysis_audit.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan_sha256": sha(plan_path), "reporting_script_sha256": sha(Path(__file__)),
        "input_sha256": input_hashes, "measurement_rows": len(frame), "transformed_covers": len(attacks),
        "covers": len(covers), "conditions": len(CONDITIONS), "decoders_per_transformed_cover": 2,
        "source_counts": {key: len(value) for key, value in controls.items()},
        "matched_ai_covers_per_method_per_payload": 204, "cross_method_rows": 15912,
        "unmarked_control_measurement_rows": 10608, "matched_marked_measurement_rows": 15912,
        "saved_fixed_rule_verified_flags_recomputed": len(frame),
        "expected_trustmark_schema": 1, "expected_trustmark_message_bits": 61,
        "all_checks_passed": True,
        "no_new_inference_or_test_calibration": True,
        "outputs_sha256": {p.name: sha(p) for p in (OUT / "control_per_condition.csv", OUT / "control_overall.csv", OUT / "cross_method_acceptance.csv", OUT / "table_ready.csv", OUT / "summary.json")}
    })
    print(json.dumps({"rows": len(frame), "controls": 408, "table_rows": len(table), "all_checks_passed": True}))


if __name__ == "__main__":
    main()
