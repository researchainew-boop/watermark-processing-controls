"""Read-only post hoc pixel/aHash overlap audit of every locked generated pair.

The sole written artifact is an aggregate, hash-bound audit report. This does
not filter images, resample seeds, rerun detectors, or change any measurement.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5/paired_content"
cv2.setNumThreads(1)


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def records(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pixel_digest(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def ahash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((small >= small.mean()).ravel()).tobytes(), "big")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(generated, references, within=False):
    exact_pairs = ahash_pairs = either_pairs = 0
    generated_exact, generated_ahash, generated_either = set(), set(), set()
    references_exact, references_ahash, references_either = set(), set(), set()
    cross = Counter()
    minimum_hd = []
    for i, image in enumerate(generated):
        nearest = 65
        candidates = references[:i] if within else references
        for j, reference in enumerate(candidates):
            matches = {f"generated_{a}_vs_reference_{b}": bool(image[a] and reference[b] and image[a] == reference[b])
                       for a in ["native", "standardized"] for b in ["native", "standardized"]}
            exact = any(matches.values())
            distance = (image["ahash"] ^ reference["ahash"]).bit_count()
            nearest = min(nearest, distance)
            average = distance <= 2
            if exact:
                exact_pairs += 1
                generated_exact.add(i)
                references_exact.add(j)
                cross.update({key: 1 for key, match in matches.items() if match})
            if average:
                ahash_pairs += 1
                generated_ahash.add(i)
                references_ahash.add(j)
            if exact or average:
                either_pairs += 1
                generated_either.add(i)
                references_either.add(j)
                if within:
                    # Both members of an undirected generated-image pair count.
                    generated_either.add(j)
                    references_either.add(i)
                    if exact:
                        generated_exact.add(j)
                        references_exact.add(i)
                    if average:
                        generated_ahash.add(j)
                        references_ahash.add(i)
        if candidates:
            minimum_hd.append(nearest)
    return {"comparison_pairs": len(generated) * (len(generated)-1)//2 if within else len(generated)*len(references),
        "exact_pixel_overlap_pairs": exact_pairs, "ahash_HD_le2_overlap_pairs": ahash_pairs,
        "either_overlap_pairs": either_pairs,
        "generated_images_with_exact_overlap": len(generated_exact),
        "generated_images_with_ahash_HD_le2_overlap": len(generated_ahash),
        "generated_images_with_either_overlap": len(generated_either),
        "reference_records_with_exact_overlap": len(references_exact),
        "reference_records_with_ahash_HD_le2_overlap": len(references_ahash),
        "reference_records_with_either_overlap": len(references_either),
        "cross_stage_exact_pair_counts": {f"generated_{a}_vs_reference_{b}": cross[f"generated_{a}_vs_reference_{b}"]
            for a in ["native", "standardized"] for b in ["native", "standardized"]},
        "minimum_ahash_HD": min(minimum_hd),
        "nearest_preceding_reference_HD_histogram" if within else "nearest_reference_HD_histogram":
            {str(key): count for key, count in sorted(Counter(minimum_hd).items())}}


def audit():
    started = datetime.now(timezone.utc).isoformat()
    input_names = ["protocol.json", "runtime_protocol.json", "generation_assets_lock.json", "selection_complete.json",
                   "generation_complete.json", "source_pairs.csv", "generated_sources.csv", "exclusion_manifest.csv",
                   "manifest.csv", "summary.json", "complete.json"]
    bound = {name: digest(OUT / name) for name in input_names}
    selection, generation, done, runtime = (read(name) for name in
        ["selection_complete.json", "generation_complete.json", "complete.json", "runtime_protocol.json"])
    require(datetime.fromisoformat(read("summary.json")["completed_at_utc"].replace("Z", "+00:00"))
            <= datetime.fromisoformat(started), "Postgeneration audit precedes completed test analysis")
    require(selection["source_pairs_sha256"] == bound["source_pairs.csv"]
            and selection["exclusion_manifest_sha256"] == bound["exclusion_manifest.csv"], "Changed locked paired selection/exclusion")
    require(generation["generated_sources_sha256"] == bound["generated_sources.csv"], "Changed locked generation")
    for name, value in done["output_sha256"].items():
        require(digest(OUT / name) == value, "Changed completed paired artifact: " + name)
    for name, value in runtime["input_sha256"].items():
        require(digest(ROOT / name) == value, "Changed runtime/head lock: " + name)
    pairs, gen_rows, prior_rows = (records(name) for name in ["source_pairs.csv", "generated_sources.csv", "exclusion_manifest.csv"])
    require(len(pairs) == len(gen_rows) == 200 and len(prior_rows) == 7592, "Unexpected locked cohort sizes")
    require(len({row["pair_id"] for row in pairs}) == len({row["pair_id"] for row in gen_rows}) == 200
            and {row["pair_id"] for row in pairs} == {row["pair_id"] for row in gen_rows}, "Generated pair membership mismatch")
    generated = []
    for row in gen_rows:
        require(digest(row["ai_native_path"]) == row["ai_native_file_sha256"]
                and digest(row["ai_path"]) == row["ai_file_sha256"], "Changed generated image")
        native = cv2.imread(row["ai_native_path"], cv2.IMREAD_COLOR)
        standard = cv2.imread(row["ai_path"], cv2.IMREAD_COLOR)
        require(native is not None and native.shape == (512, 512, 3)
                and standard is not None and standard.shape == (256, 256, 3), "Generated pixel shape/decode mismatch")
        native_hash, standard_hash = pixel_digest(native), pixel_digest(standard)
        require(standard_hash == row["ai_standardized_pixel_sha256"], "Changed generated standardized fingerprint")
        prejpeg = cv2.resize(native, (256, 256), interpolation=cv2.INTER_AREA)
        generated.append({"native": native_hash, "standardized": standard_hash, "ahash": ahash(prejpeg)})
    prior = [{"native": row["raw_pixel_sha256"], "standardized": row["standardized_pixel_sha256"], "ahash": int(row["ahash"])} for row in prior_rows]
    photos = [{"native": row["raw_pixel_sha256"], "standardized": row["standardized_pixel_sha256"], "ahash": int(row["prejpeg_ahash"])} for row in pairs]
    scopes = {"prior_acquisition_records": compare(generated, prior),
              "selected_caption_photographs": compare(generated, photos),
              "within_generated": compare(generated, generated, within=True)}
    for name, value in bound.items():
        require(digest(OUT / name) == value, "Artifact changed during read-only overlap audit: " + name)
    return {"audit_completed": True, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_hoc": True, "after_generation_and_test_analysis": True,
        "selected_pairs": 200, "generated_images": 200, "prior_records": 7592, "selected_photographs": 200,
        "input_sha256": bound, "auditor_script_sha256": digest(__file__),
        "generated_fingerprint_inventory_sha256": hashlib.sha256(json.dumps(generated, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "method": "SHA256 of decoded BGR shape/pixels at native and standardized stages, all four cross-stage comparisons; preJPEG256 RGB-to-BGR area-resize, gray8x8 INTER_AREA mean-threshold aHash, Hamming distance<=2; archived reference fingerprints unchanged",
        "comparisons": scopes,
        "no_detected_overlap_under_these_pixel_and_ahash_rules": all(value["either_overlap_pairs"] == 0 for value in scopes.values()),
        "all_locked_test_items_retained": True,
        "filtering_resampling_model_inference_or_score_changes": False,
        "limitations": ["This additional screen was performed after generation and test scoring and was not a prospective source-selection rule.",
            "No generated image is removed or regenerated regardless of overlap findings.",
            "Pixel fingerprints and coarse average hashes do not establish semantic independence or rule out model-pretraining overlap.",
            "Prior reference aHashes use their archived processing basis; differences in previous encoding can affect hash distances.",
            "Matching fingerprints across stages is an overlap indicator, not proof that two sources depict the same scene."],
        "scope": "CPU-only read of all locked images and archives; aggregate counts/hashes only, no image bytes, captions, prompts, paths, source IDs or detector scores released"}


if __name__ == "__main__":
    report = audit()
    target = OUT / "postgeneration_overlap_audit.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        require(previous["input_sha256"] == report["input_sha256"] and previous["comparisons"] == report["comparisons"],
                "Refusing to overwrite an audit of changed inputs or different overlap findings")
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_completed": True, "generated_images": 200,
        "overlap_pair_counts": {name: value["either_overlap_pairs"] for name, value in report["comparisons"].items()},
        "no_filtering_or_resampling": True}, indent=2))
