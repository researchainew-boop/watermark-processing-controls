"""Prospectively fixed, source-disjoint watermark calibration and comparison.

No detector fitting occurs. CPU preparation embeds DWT without decoding.
TrustMark encoding and calibration/test decoding are separate GPU stages.
Raw margins are empirical scores, not probabilities or authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiwstudy.attacks import apply_attack
from aiwstudy.watermark import DwtDctWatermark
import benchmark_watermarks as previous

OUT = ROOT / "outputs/phase5/fair_watermarks"
DATA = ROOT / "data/phase5/fair_watermarks"
SEED = 20261015
PAYLOADS = previous.PAYLOADS
CONDITIONS = (("identity", "identity", 0), ("jpeg50", "jpeg", 2),
              ("crop", "crop", 1), ("lowres32", "lowres", 2))
TARGET_PSNR = 40.0
PSNR_TOLERANCE = .1
METHODS = ("dwt_dct", "trustmark")
QUANTIZATION = "stochastic_rounding_v1"
INPUTS = {"original_sources": ROOT / "outputs/conference_clean/source_audit.csv",
          "openimages": ROOT / "outputs/phase3/controls/openimages_sources.csv",
          "existing_AI_and_COCO": ROOT / "outputs/phase3/watermarks/selected_sources.csv",
          "existing_pairs": ROOT / "outputs/phase3/watermarks/selected_pairs.csv"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return previous.digest(Path(path))


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write(path, value):
    previous.write_json(Path(path), value)


def ident(*values):
    return previous.identity("phase5_fair_watermarks", *values)


def ordered(frame, arm):
    frame = frame.copy()
    frame["order"] = frame.source_id.map(lambda s: hashlib.sha256(f"{SEED}|{arm}|{s}".encode()).hexdigest())
    return frame.sort_values(["order", "source_id"]).drop(columns="order")


def ahash(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for bit in (small >= small.mean()).flat:
        value = (value << 1) | int(bit)
    return value


def source_pixels(row):
    im = Image.open(row["native_path"]).convert("RGB")
    rotation = float(row.get("Rotation", "") or 0)
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Unsupported metadata rotation")
    if rotation:
        im = im.rotate(rotation, expand=True)
    if min(im.size) < 256:
        raise ValueError("Short source side below 256")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def common_pixels(native):
    resized = cv2.resize(native, (256, 256), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        protocol = json.loads(path.read_text())
        for name, input_path in INPUTS.items():
            if digest(input_path) != protocol["input_sha256"][name]:
                raise RuntimeError(f"Changed selection input: {name}")
        return protocol
    protocol = {"frozen_at_utc": utc(), "seed": SEED,
        "sources": {"calibration_photographs": 400, "test_photographs": 400, "test_AI": 204},
        "photographic_collections_per_split": {"coco": 200, "openimages": 200},
        "AI_per_collection": 68, "payloads": list(PAYLOADS),
        "conditions": [x[0] for x in CONDITIONS], "source_selection":
        "Fixed source-ID SHA ordering; retain 200 of old 204 test COCO controls; calibration COCO from old validation; Open Images ranked 1000-source screened pool split 200 calibration/200 test; preserve all old 204 AI IDs",
        "source_reuse": "Already examined collections/items; newly fixed watermark scores and source-disjoint calibration, not a new origin-detection benchmark",
        "preprocessing": "PIL RGB; positive CCW Open Images rotation; OpenCV area resize256 square; identical OpenCV JPEG95 encode/decode; PNG staging",
        "prior_encoding_limit": "Original synthetic exports already had JPEG95; common recoding does not erase earlier collection encoding differences",
        "embedding": {"base_DWT_strength": 12.0, "TrustMark_version": "0.9.2", "variant": "Q", "ECC": "BCH5", "base_TrustMark_strength": 1.0,
            "target_PSNR_dB": TARGET_PSNR, "tolerance_dB": PSNR_TOLERANCE,
            "gain_search": "Whole RGB residual from base encoded uint8 cover, gain[0,64] expanding from1 then48 binary steps; rint, clip[0,255]; pick closest PSNR evaluated pixels, error if outside tolerance; distortion only, no decoder",
            "interpretation": "Generic residual gain is outside the official TrustMark encoding configuration; shared16 information bits but embedding redundancy remains method-specific"},
        "score": {"dwt_dct": "mean((2*expected16-1)*median_coefficient_margins/12)",
            "trustmark": "mean((2*expected16-1)*raw_decoder_logits[:16]); official Q preprocessing",
            "interpretation": "Uncalibrated expected-information-bit margins, not message probabilities, ECC verification, or issuer authentication",
            "queries": {"default": "First existing payload", "any3": "Maximum score across all three preexisting payloads"}},
        "calibration": {"nominal_FPR": .05, "unit": "Equal-weight 1600 real images (400 sources x4 conditions)",
            "rule": "Lowest score threshold whose empirical calibration FPR <=.05; accept >=threshold, retain whole ties; separate method/default and method/any3 thresholds",
            "strict_separation": "Calibration margins then threshold file locked before test image decoding; no marked calibration images needed",
            "claim": "Shared nominal validation target, not exact equal test FPR or population error guarantee"},
        "primary_test": {"default_positive": "Default-payload marked AI scored by matching method", "any3_positive": "All three marked payload covers scored by matching method, source-average and payload-specific results",
            "negative": "Unmarked photographs; unmarked AI false watermark matches reported separately", "quality": "PSNR and SSIM against original unmarked cover and attack input"},
        "secondary_verification": "DWT>=13/16; TrustMark valid BCH5/schema1 and complete expected61-bit message; different native rules remain secondary",
        "uncertainty": "2000 collection-stratified source bootstrap draws seed20261016; derivatives/payloads paired within source; exact two-sided95% binomial intervals for one event per source; source-ever event distinguished from per-image rate",
        "input_sha256": {name: digest(path) for name, path in INPUTS.items()},
        "initial_driver_sha256": digest(__file__), "no_decoder_run_before_lock": True}
    write(path, protocol)
    return protocol


def select():
    freeze()
    if (OUT / "sources.csv").exists():
        validate_sources()
        print("Source manifest already locked", flush=True)
        return
    old = read_csv(INPUTS["original_sources"])
    old = old.loc[old.retained.str.lower().eq("true")].set_index("source_id", verify_integrity=True)
    selected = read_csv(INPUTS["existing_AI_and_COCO"])
    oi = read_csv(INPUTS["openimages"])
    ai = selected.loc[selected.label.eq("1")].copy()
    test_coco = ordered(selected.loc[selected.label.eq("0")], "test_coco").head(200)
    cal_coco = ordered(old.reset_index().loc[lambda x: x.label.eq("0") & x.split.isin(("val", "validation"))], "cal_coco").head(200)
    oi = ordered(oi, "openimages")
    records = []
    for role, collection, frame in (("calibration", "coco", cal_coco), ("calibration", "openimages", oi.iloc[:200]),
                                    ("test", "coco", test_coco), ("test", "openimages", oi.iloc[200:400]), ("test", "AI", ai)):
        for row in frame.to_dict("records"):
            sid = row["source_id"]
            original = row if collection == "openimages" else old.loc[sid].to_dict()
            path = original.get("native_path") or original["path"]
            gen = row["generator"] if collection == "AI" else collection
            records.append({"source_id": sid, "role": role, "collection": gen, "label": int(collection == "AI"),
                "native_path": path, "input_file_sha256": digest(path), "Rotation": original.get("Rotation", ""),
                "original_id": original.get("original_id") or Path(path).name, "prior_split": original.get("split", "already-used-pool"),
                "prior_watermark_test": int(sid in set(selected.source_id)), "prior_collected_source": 1})
    frame = pd.DataFrame(records).sort_values(["role", "collection", "source_id"]).reset_index(drop=True)
    if len(frame) != 1004 or frame.source_id.nunique() != 1004:
        raise RuntimeError("Source selection count/identity error")
    if frame.groupby(["role", "collection"]).size().to_dict() != {
        ("calibration", "coco"): 200, ("calibration", "openimages"): 200,
        ("test", "coco"): 200, ("test", "openimages"): 200,
        ("test", "stable-diffusion"): 68, ("test", "dalle-3"): 68, ("test", "midjourney"): 68}:
        raise RuntimeError("Unexpected collection counts")
    hashes, raw_hashes, cluster_references = [], [], []
    for i, row in enumerate(frame.to_dict("records")):
        pixels = source_pixels(row)
        raw_hashes.append(hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest())
        hashes.append(ahash(pixels))
        cluster_references.append(row["source_id"])
    duplicates = []
    for i in range(len(frame)):
        for j in range(i):
            if raw_hashes[i] == raw_hashes[j] or (hashes[i] ^ hashes[j]).bit_count() <= 2:
                duplicates.append([cluster_references[j], cluster_references[i]])
    if duplicates:
        write(OUT / "selection_overlap_failure.json", {"pairs": duplicates})
        raise RuntimeError("Selected sources overlap under fixed duplicate screen; do not alter sample based on scores")
    frame["native_pixel_sha256"] = raw_hashes
    frame["native_ahash"] = [str(x) for x in hashes]
    frame.to_csv(OUT / "sources.csv", index=False)
    write(OUT / "selection_complete.json", {"completed_at_utc": utc(), "source_count": 1004,
        "counts": [{"role": r, "collection": c, "sources": int(n)} for (r, c), n in frame.groupby(["role", "collection"]).size().items()],
        "source_manifest_sha256": digest(OUT / "sources.csv"), "protocol_sha256": digest(OUT / "protocol.json"),
        "ID_or_pixel_or_ahash_pairs": 0, "preserved_prior_AI_sources": sorted(ai.source_id.tolist()),
        "retained_prior_COCO_controls": sorted(test_coco.source_id.tolist()), "new_independent_source_dataset": False})
    print("Locked 1004 sources:400 calibration photos,400 test photos,204 existing AI", flush=True)


def validate_sources():
    freeze()
    selection = json.loads((OUT / "selection_complete.json").read_text())
    if digest(OUT / "sources.csv") != selection["source_manifest_sha256"] or digest(OUT / "protocol.json") != selection["protocol_sha256"]:
        raise RuntimeError("Changed source/protocol manifest")
    frame = read_csv(OUT / "sources.csv")
    if len(frame) != 1004 or frame.source_id.nunique() != 1004:
        raise RuntimeError("Invalid source identities")
    return frame


def rounding_seed(source_id, payload):
    key = f"{SEED}|{QUANTIZATION}|{source_id}|{payload}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def freeze_quantization_amendment():
    validate_sources()
    path = OUT / "protocol_amendment.json"
    if path.exists():
        amendment = json.loads(path.read_text())
        if amendment["quantization"] != QUANTIZATION or amendment["original_protocol_sha256"] != digest(OUT / "protocol.json"):
            raise RuntimeError("Changed quantization amendment")
        return
    if any((OUT / f"{role}_raw.npz").exists() or (OUT / f"{role}_inference_started.json").exists() for role in ("calibration", "test")):
        raise RuntimeError("Embedding amendment cannot follow scoring")
    seed_records = [{"source_id": sid, "payload": payload, "seed_uint64": str(rounding_seed(sid, payload))}
                    for sid in sorted(validate_sources().loc[lambda x: x.label.eq("1")].source_id) for payload in PAYLOADS]
    pd.DataFrame(seed_records).to_csv(OUT / "rounding_seeds.csv", index=False)
    write(path, {"locked_at_utc": utc(), "quantization": QUANTIZATION,
        "reason": "Nearest-integer quantization cannot always reach40±0.1dB with scalar gain; first preparation stopped at39.873250152390234dB before decoding",
        "authorization": "Root approved deterministic stochastic rounding for both methods before any scoring",
        "original_protocol_sha256": digest(OUT / "protocol.json"), "source_manifest_sha256": digest(OUT / "sources.csv"),
        "seed_manifest_sha256": digest(OUT / "rounding_seeds.csv"),
        "seed_derivation": "uint64 from first16hex SHA256(f'{20261015}|stochastic_rounding_v1|{source_id}|{payload}'); method excluded so both methods use identical uniforms for each source/payload",
        "generator": "numpy.random.Generator(PCG64(seed)); one float64 uniform[0,1) array with cover shape256x256x3, fixed across every gain-search evaluation",
        "pixels": "BGR uint8 cover and encoded reference; float64 full-channel residual/gain; floor(cover+gain*residual+uniform); clip[0,255]; uint8 output",
        "target_PSNR_dB": 40.0, "tolerance_dB": .1, "gain_bracket": [0., 64.], "binary_search_steps": 48,
        "interpretation": "Residual gain and stochastic quantization are outside both original/default encoding configurations",
        "selection_criterion": "Image PSNR only; no decoder output, new seed search, recovery selection, or source changes",
        "decoding_has_not_started": True, "driver_sha256": digest(__file__)})


def validate_cpu_manifests():
    validate_sources()
    lock = json.loads((OUT / "cpu_preparation_complete.json").read_text())
    for name, key in (("base_sources.csv", "base_manifest_sha256"), ("cpu_covers.csv", "covers_sha256"), ("sources.csv", "sources_sha256")):
        if digest(OUT / name) != lock[key]:
            raise RuntimeError(f"Changed CPU preparation manifest:{name}")
    if digest(OUT / "protocol_amendment.json") != lock["amendment_sha256"]:
        raise RuntimeError("Changed encoding amendment")


def psnr(reference, candidate):
    mse = float(np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2))
    return 120.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def matched_residual(reference, encoded, source_id, payload):
    residual = encoded.astype(np.float64) - reference.astype(np.float64)
    original = reference.astype(np.float64)
    uniforms = np.random.default_rng(rounding_seed(source_id, payload)).random(reference.shape, dtype=np.float64)
    def at(gain):
        result = np.clip(np.floor(original + gain * residual + uniforms), 0, 255).astype(np.uint8)
        return result, psnr(reference, result)
    lo, hi = 0.0, 1.0
    candidate, value = at(hi)
    while value > TARGET_PSNR and hi < 64:
        hi *= 2
        candidate, value = at(hi)
    if value > TARGET_PSNR:
        raise RuntimeError("Encoded residual cannot reach fixed distortion target")
    best = (abs(value - TARGET_PSNR), hi, candidate, value)
    for _ in range(48):
        mid = (lo + hi) / 2
        candidate, value = at(mid)
        if abs(value - TARGET_PSNR) < best[0]:
            best = (abs(value - TARGET_PSNR), mid, candidate, value)
        if value > TARGET_PSNR:
            lo = mid
        else:
            hi = mid
    if best[0] > PSNR_TOLERANCE:
        raise RuntimeError(f"PSNR target failure:{best[3]}")
    return best[2], best[1], best[3]


def save_cover(source, method, payload, pixels, gain=0, base_psnr=120):
    cid = ident("cover", source["source_id"], method, payload, QUANTIZATION if method != "unmarked" else "unmarked")
    path = DATA / "covers" / QUANTIZATION / method / f"{cid}.png"
    previous.save_pixels(path, pixels)
    reference = previous.read_image(Path(source["base_path"]))
    return {"cover_id": cid, "source_id": source["source_id"], "role": source["role"], "collection": source["collection"],
        "label": int(source["label"]), "watermark_method": method, "payload": payload, "path": str(path),
        "output_file_sha256": digest(path), "base_path": source["base_path"], "base_sha256": source["base_sha256"],
        "residual_gain": gain, "base_encoder_PSNR_dB": base_psnr,
        "rounding_seed_uint64": str(rounding_seed(source["source_id"], payload)) if payload else "",
        "quantization": QUANTIZATION if method != "unmarked" else "unmodified",
        "embedding_PSNR_dB": psnr(reference, pixels), "embedding_SSIM": float(structural_similarity(reference, pixels, channel_axis=2, data_range=255))}


def prepare_cpu():
    select()
    freeze_quantization_amendment()
    sources = validate_sources()
    bases, covers = [], []
    for index, source in enumerate(sources.to_dict("records")):
        if digest(source["native_path"]) != source["input_file_sha256"]:
            raise RuntimeError("Changed native source")
        pixels = common_pixels(source_pixels(source))
        path = DATA / "base" / f"{source['source_id']}.png"
        previous.save_pixels(path, pixels)
        source = {**source, "base_path": str(path), "base_sha256": digest(path)}
        bases.append(source)
        covers.append(save_cover(source, "unmarked", "", pixels))
        if int(source["label"]):
            for payload in PAYLOADS:
                encoded = DwtDctWatermark(payload, 12).embed(pixels)
                matched, gain, _ = matched_residual(pixels, encoded, source["source_id"], payload)
                covers.append(save_cover(source, "dwt_dct", payload, matched, gain, psnr(pixels, encoded)))
        if (index + 1) % 100 == 0:
            print("CPU bases/DWT", index + 1, "/1004", flush=True)
    pd.DataFrame(bases).to_csv(OUT / "base_sources.csv", index=False)
    pd.DataFrame(covers).to_csv(OUT / "cpu_covers.csv", index=False)
    write(OUT / "cpu_preparation_complete.json", {"completed_at_utc": utc(), "source_count": len(bases), "covers": len(covers),
        "sources_sha256": digest(OUT / "sources.csv"), "base_manifest_sha256": digest(OUT / "base_sources.csv"), "covers_sha256": digest(OUT / "cpu_covers.csv"),
        "decoder_run": False, "driver_sha256": digest(__file__), "amendment_sha256": digest(OUT / "protocol_amendment.json")})


def prepare_learned(args):
    validate_cpu_manifests()
    if (OUT / "encoding_complete.json").exists() or any((OUT / f"{role}_inference_started.json").exists() for role in ("calibration", "test")):
        raise RuntimeError("Refusing additional encoding after completion or decoding")
    started = {"started_at_utc": utc(), "source_count": 204, "marked_covers": 612,
        "protocol_sha256": digest(OUT / "protocol.json"), "amendment_sha256": digest(OUT / "protocol_amendment.json"),
        "source_manifest_sha256": digest(OUT / "sources.csv"), "cpu_covers_sha256": digest(OUT / "cpu_covers.csv"),
        "driver_sha256": digest(__file__), "decoder_run": False, "device": args.device}
    write(OUT / "learned_encoding_started.json", started)
    timer = time.perf_counter()
    marker = previous.Watermarks(args.device, args.threads)
    bases = read_csv(OUT / "base_sources.csv")
    covers = read_csv(OUT / "cpu_covers.csv").to_dict("records")
    learned = []
    for index, source in enumerate(bases.loc[bases.label.eq("1")].to_dict("records")):
        pixels = previous.read_image(Path(source["base_path"]))
        if digest(source["base_path"]) != source["base_sha256"]:
            raise RuntimeError("Changed unmarked base")
        for payload in PAYLOADS:
            marker.set_payload(payload)
            encoded = marker.encode(pixels, "trustmark")
            matched, gain, _ = matched_residual(pixels, encoded, source["source_id"], payload)
            learned.append(save_cover(source, "trustmark", payload, matched, gain, psnr(pixels, encoded)))
        if (index + 1) % 12 == 0:
            pd.DataFrame(learned).to_csv(OUT / "learned_covers.partial.csv", index=False)
            print("Matched TrustMark covers", index + 1, "/204; elapsed seconds", round(time.perf_counter() - timer, 1), flush=True)
    pd.DataFrame(covers + learned).to_csv(OUT / "covers.csv", index=False)
    if len(covers) + len(learned) != 2228:
        raise RuntimeError("Cover count mismatch")
    write(OUT / "encoding_complete.json", {**started, "completed_at_utc": utc(), "elapsed_seconds": time.perf_counter() - timer, "covers": 2228,
        "cover_manifest_sha256": digest(OUT / "covers.csv"), "official_checkpoint_provenance": marker.provenance(),
        "decoder_run": False, "driver_sha256": digest(__file__), "amendment_sha256": digest(OUT / "protocol_amendment.json")})


def derive_cover_rows(covers):
    rows = []
    for index, cover in enumerate(covers.to_dict("records")):
        if digest(cover["path"]) != cover["output_file_sha256"]:
            raise RuntimeError("Changed cover")
        pixels = previous.read_image(Path(cover["path"]))
        original = previous.read_image(Path(cover["base_path"]))
        for condition, attack, severity in CONDITIONS:
            result = apply_attack(pixels, attack, severity)
            iid = ident(cover["cover_id"], condition)
            path = DATA / "images" / condition / f"{iid}.png"
            previous.save_pixels(path, result)
            rows.append({**cover, "image_id": iid, "condition": condition, "input_cover_path": cover["path"],
                "input_cover_sha256": cover["output_file_sha256"], "path": str(path), "output_file_sha256": digest(path),
                "PSNR_vs_original_dB": psnr(original, result), "SSIM_vs_original": float(structural_similarity(original, result, channel_axis=2, data_range=255)),
                "PSNR_vs_attack_input_dB": psnr(pixels, result), "SSIM_vs_attack_input": float(structural_similarity(pixels, result, channel_axis=2, data_range=255))})
        if (index + 1) % 200 == 0:
            print("Derived conditions", index + 1, "/", len(covers), "covers", flush=True)
    return pd.DataFrame(rows)


def derive_cpu():
    validate_cpu_manifests()
    if (OUT / "cpu_derivation_complete.json").exists():
        raise RuntimeError("CPU derivative preparation already completed")
    started = utc()
    frame = derive_cover_rows(read_csv(OUT / "cpu_covers.csv"))
    if len(frame) != 6464 or frame.image_id.nunique() != 6464:
        raise RuntimeError("CPU derivative count mismatch")
    frame.to_csv(OUT / "cpu_manifest.csv", index=False)
    write(OUT / "cpu_derivation_complete.json", {"started_at_utc": started, "completed_at_utc": utc(), "images": 6464,
        "manifest_sha256": digest(OUT / "cpu_manifest.csv"), "cpu_covers_sha256": digest(OUT / "cpu_covers.csv"),
        "amendment_sha256": digest(OUT / "protocol_amendment.json"), "decoder_run": False})


def derive():
    validate_sources()
    covers = read_csv(OUT / "covers.csv")
    lock = json.loads((OUT / "encoding_complete.json").read_text())
    if digest(OUT / "covers.csv") != lock["cover_manifest_sha256"]:
        raise RuntimeError("Changed cover manifest")
    if (OUT / "cpu_derivation_complete.json").exists():
        cpu_lock = json.loads((OUT / "cpu_derivation_complete.json").read_text())
        if digest(OUT / "cpu_manifest.csv") != cpu_lock["manifest_sha256"] or digest(OUT / "cpu_covers.csv") != cpu_lock["cpu_covers_sha256"] or digest(OUT / "protocol_amendment.json") != cpu_lock["amendment_sha256"]:
            raise RuntimeError("Changed CPU derivatives or inputs")
        cpu_frame = read_csv(OUT / "cpu_manifest.csv")
        cpu_covers = read_csv(OUT / "cpu_covers.csv")
        if len(cpu_frame) != 6464 or set(cpu_frame.cover_id) != set(cpu_covers.cover_id) or not set(cpu_covers.cover_id).issubset(set(covers.cover_id)):
            raise RuntimeError("CPU derivative identities mismatch")
        for row in cpu_frame.to_dict("records"):
            if digest(row["path"]) != row["output_file_sha256"]:
                raise RuntimeError("Changed CPU derivative image")
        new_frame = derive_cover_rows(covers.loc[~covers.cover_id.isin(cpu_covers.cover_id)])
        frame = pd.concat([cpu_frame, new_frame], ignore_index=True)
    else:
        frame = derive_cover_rows(covers)
    if len(frame) != 8912 or frame.image_id.nunique() != 8912:
        raise RuntimeError("Derived image count mismatch")
    frame.to_csv(OUT / "manifest.csv", index=False)
    write(OUT / "derivation_complete.json", {"completed_at_utc": utc(), "images": len(frame),
        "counts_per_role": frame.role.value_counts().to_dict(), "manifest_sha256": digest(OUT / "manifest.csv"),
        "cover_manifest_sha256": digest(OUT / "covers.csv"), "protocol_sha256": digest(OUT / "protocol.json"),
        "amendment_sha256": digest(OUT / "protocol_amendment.json"), "decoder_run": False})


def validate_manifest():
    validate_sources()
    lock = json.loads((OUT / "derivation_complete.json").read_text())
    if digest(OUT / "manifest.csv") != lock["manifest_sha256"]:
        raise RuntimeError("Changed derived manifest")
    if digest(OUT / "protocol_amendment.json") != lock["amendment_sha256"]:
        raise RuntimeError("Changed encoding/quantization amendment")
    return read_csv(OUT / "manifest.csv")


def anchors(marker):
    old = read_csv(ROOT / "outputs/phase3/watermarks/measurements.csv")
    candidates = old.loc[old.detector.eq("trustmark") & old.condition.eq("identity")]
    selected = candidates.sort_values("image_id").drop_duplicates(["watermark_method", "paired_generator"]).head(10)
    records = []
    for row in selected.to_dict("records"):
        pixels = previous.read_image(Path(row["path"]))
        if digest(row["path"]) != row["output_file_sha256"]:
            raise RuntimeError("Changed archived API anchor")
        im = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        patch = marker.tm.get_the_image_for_processing(im).resize((245, 245), Image.Resampling.BILINEAR)
        tensor = marker.transforms.ToTensor()(patch).unsqueeze(0).to(marker.tm.decoder.device) * 2 - 1
        with marker.torch.inference_mode():
            raw = (marker.tm.decoder.decoder(tensor) > 0).cpu().numpy().astype(np.uint8)
        message, valid, schema = marker.tm.ecc.decode_bitstream(raw, MODE="binary")[0]
        official = marker.tm.decode(im, MODE="binary", ROTATION=False, DETECTFIRST=False)
        same = (bool(valid) == bool(official[1]) and (not valid or (message == official[0] and int(schema) == int(official[2]))))
        bits = "".join(map(str, raw[0, :16].tolist()))
        old_same = bits == row["payload_bits"] and int(bool(valid)) == int(row["ecc_valid"])
        if not same or not old_same:
            raise RuntimeError("Official/API archived-bit anchor mismatch")
        records.append({"image_id": row["image_id"], "file_sha256": row["output_file_sha256"], "matches_public_API": same, "matches_archived_bits_ECC": old_same})
    return records


def decode(args, role):
    manifest = validate_manifest()
    if role == "test":
        thresholds = json.loads((OUT / "thresholds.json").read_text())
        if thresholds["manifest_sha256"] != digest(OUT / "manifest.csv"):
            raise RuntimeError("Threshold manifest mismatch")
        threshold_sha = digest(OUT / "thresholds.json")
    else:
        threshold_sha = None
    target = OUT / f"{role}_raw.npz"
    if target.exists():
        raise RuntimeError(f"Refusing additional {role} decoding; raw file exists")
    marker = previous.Watermarks(args.device, args.threads)
    checkpoint_provenance = marker.provenance()["checkpoints"]
    encoded = json.loads((OUT / "encoding_complete.json").read_text())
    if checkpoint_provenance != encoded["official_checkpoint_provenance"]["checkpoints"]:
        raise RuntimeError("Encoding/decoding checkpoint fingerprints differ")
    if role == "test":
        cal_inference = json.loads((OUT / "calibration_inference_complete.json").read_text())
        if checkpoint_provenance != cal_inference["checkpoints"]:
            raise RuntimeError("Calibration/test checkpoint fingerprints differ")
    check = anchors(marker)
    part = manifest.loc[manifest.role.eq(role)].sort_values("image_id").reset_index(drop=True)
    started = {"started_at_utc": utc(), "role": role, "image_count": len(part), "manifest_sha256": digest(OUT / "manifest.csv"),
        "source_sha256": digest(OUT / "sources.csv"), "protocol_sha256": digest(OUT / "protocol.json"),
        "amendment_sha256": digest(OUT / "protocol_amendment.json"), "threshold_sha256_before_test": threshold_sha,
        "driver_sha256": digest(__file__), "device": args.device, "batch_size": args.batch_size,
        "checkpoints": checkpoint_provenance, "anchors": check,
        "anchor_scope": "Archived previously decoded API checks; not newly prepared test images or analysis outcomes"}
    write(OUT / f"{role}_inference_started.json", started)
    timer = time.perf_counter()
    dwt_values, trust_values, decoded_messages, valid_flags, schemas = [], [], [], [], []
    for begin in range(0, len(part), args.batch_size):
        block = part.iloc[begin:begin + args.batch_size].to_dict("records")
        tensors = []
        for row in block:
            if digest(row["path"]) != row["output_file_sha256"]:
                raise RuntimeError("Changed test/calibration image")
            pixels = previous.read_image(Path(row["path"]))
            dwt_values.append(DwtDctWatermark(PAYLOADS[0], 12).scores(pixels))
            im = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            patch = marker.tm.get_the_image_for_processing(im).resize((245, 245), Image.Resampling.BILINEAR)
            tensors.append(marker.transforms.ToTensor()(patch) * 2 - 1)
        with marker.torch.inference_mode():
            logits = marker.tm.decoder.decoder(marker.torch.stack(tensors).to(marker.tm.decoder.device)).cpu().numpy().astype(np.float32)
        if not np.isfinite(logits).all() or logits.shape != (len(block), 100):
            raise RuntimeError("Invalid raw TrustMark logits")
        trust_values.extend(logits)
        decoded = marker.tm.ecc.decode_bitstream((logits > 0).astype(np.uint8), MODE="binary")
        for message, valid, schema in decoded:
            decoded_messages.append(str(message) if valid else "")
            valid_flags.append(bool(valid))
            schemas.append(int(schema))
        if begin % (args.batch_size * 16) == 0:
            print("Decode", role, begin + len(block), "/", len(part), "; elapsed seconds", round(time.perf_counter() - timer, 1), flush=True)
    np.savez_compressed(target, image_id=part.image_id.to_numpy(dtype=str),
        dwt_raw=np.asarray(dwt_values, dtype=np.float32), trust_raw=np.asarray(trust_values, dtype=np.float32),
        trust_message=np.asarray(decoded_messages, dtype="U61"), trust_ECC_valid=np.asarray(valid_flags, dtype=bool), trust_schema=np.asarray(schemas, dtype=np.int8))
    write(OUT / f"{role}_inference_complete.json", {**started, "completed_at_utc": utc(), "elapsed_seconds": time.perf_counter() - timer, "raw_sha256": digest(target), "additional_passes": False})
    write_scores(role)


def write_scores(role):
    manifest = validate_manifest()
    raw_path = OUT / f"{role}_raw.npz"
    locked = json.loads((OUT / f"{role}_inference_complete.json").read_text())
    if digest(raw_path) != locked["raw_sha256"]:
        raise RuntimeError("Changed raw decoder archive")
    raw = np.load(raw_path, allow_pickle=False)
    part = manifest.set_index("image_id").loc[raw["image_id"]].reset_index()
    rows = []
    for method in METHODS:
        margins = raw["dwt_raw"] / 12 if method == "dwt_dct" else raw["trust_raw"][:, :16]
        hard = raw["dwt_raw"] >= 0 if method == "dwt_dct" else raw["trust_raw"][:, :16] > 0
        for payload in PAYLOADS:
            expected = np.array(list(payload), dtype=np.int8)
            signed = 2 * expected - 1
            score = np.mean(margins.astype(np.float64) * signed, axis=1)
            agreements = np.sum(hard == expected, axis=1)
            if method == "dwt_dct":
                native_verifies = agreements >= 13
            else:
                native_verifies = raw["trust_ECC_valid"] & (raw["trust_schema"] == 1) & (raw["trust_message"] == payload + "0" * 45)
            block = part[["image_id", "source_id", "role", "collection", "label", "watermark_method", "payload", "condition"]].copy()
            block = block.rename(columns={"payload": "embedded_payload"})
            block["method"] = method
            block["query_payload"] = payload
            block["score"] = score
            block["information_bits_agree"] = agreements
            block["native_verifies"] = native_verifies.astype(int)
            rows.extend(block.to_dict("records"))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{role}_scores.csv", index=False)
    return frame


def calibrate():
    if (OUT / "test_raw.npz").exists():
        raise RuntimeError("Calibration cannot change after test decoding")
    if (OUT / "thresholds.json").exists():
        print("Thresholds already locked", flush=True)
        return
    frame = read_csv(OUT / "calibration_scores.csv")
    frame["score"] = frame.score.astype(float)
    if frame.source_id.nunique() != 400 or set(frame.label) != {"0"}:
        raise RuntimeError("Invalid calibration population")
    records = []
    for method in METHODS:
        subset = frame.loc[frame.method.eq(method)]
        for query_rule in ("default", "any3"):
            values = subset.loc[subset.query_payload.eq(PAYLOADS[0])].score.to_numpy() if query_rule == "default" else subset.groupby("image_id").score.max().to_numpy()
            candidates = np.unique(values)
            eligible = [float(t) for t in candidates if np.mean(values >= t) <= .05]
            threshold = min(eligible) if eligible else float(np.nextafter(values.max(), np.inf))
            count = int(np.sum(values >= threshold))
            records.append({"method": method, "query_rule": query_rule, "threshold": threshold,
                "calibration_images": len(values), "calibration_sources": 400, "accepted_images": count, "empirical_FPR": count / len(values)})
    write(OUT / "thresholds.json", {"locked_at_utc": utc(), "nominal_FPR": .05, "records": records,
        "manifest_sha256": digest(OUT / "manifest.csv"), "protocol_sha256": digest(OUT / "protocol.json"),
        "amendment_sha256": digest(OUT / "protocol_amendment.json"),
        "calibration_scores_sha256": digest(OUT / "calibration_scores.csv"), "calibration_raw_sha256": digest(OUT / "calibration_raw.npz"),
        "test_decoding_has_not_started": not (OUT / "test_inference_started.json").exists(), "whole_ties": True})
    print("Locked four thresholds before test decoding", flush=True)


def exact_interval(successes, total):
    return [float(beta.ppf(.025, successes, total - successes + 1)) if successes else 0.,
            float(beta.ppf(.975, successes + 1, total - successes)) if successes < total else 1.]


def describe_events(frame, event, name, seed):
    block = frame.copy()
    block["event"] = np.asarray(event, dtype=int)
    grouped = block.groupby(["source_id", "collection"]).event.agg(["mean", "min", "max", "size"])
    source_means = grouped["mean"].to_numpy()
    rng = np.random.default_rng(seed)
    collections = grouped.index.get_level_values("collection").to_numpy()
    draws = np.zeros(2000)
    for collection in sorted(set(collections)):
        values = source_means[collections == collection]
        draws += values[rng.integers(0, len(values), (2000, len(values)))].sum(axis=1)
    draws /= len(source_means)
    rate = float(source_means.mean())
    all_count, ever_count = int(grouped["min"].sum()), int(grouped["max"].sum())
    result = {"population": name, "images": len(block), "sources": len(grouped), "accepted_images": int(block.event.sum()),
        "per_image_rate": float(block.event.mean()), "source_mean_rate": rate, "source_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
        "source_ever_count": ever_count, "source_ever_rate": ever_count / len(grouped), "source_ever_exact_ci95": exact_interval(ever_count, len(grouped)),
        "source_all_count": all_count, "source_all_rate": all_count / len(grouped), "source_all_exact_ci95": exact_interval(all_count, len(grouped))}
    if grouped["size"].eq(1).all():
        result["source_binomial_ci95"] = exact_interval(ever_count, len(grouped))
    return result, draws


def analyze():
    thresholds_path = OUT / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text())
    test_lock = json.loads((OUT / "test_inference_complete.json").read_text())
    if test_lock["threshold_sha256_before_test"] != digest(thresholds_path):
        raise RuntimeError("Threshold changed after test inference began")
    if thresholds["calibration_scores_sha256"] != digest(OUT / "calibration_scores.csv"):
        raise RuntimeError("Changed calibration scores")
    frame = read_csv(OUT / "test_scores.csv")
    frame["score"] = frame.score.astype(float)
    frame["native_verifies"] = frame.native_verifies.astype(int)
    measurements, bootstrap_arrays = [], {}
    threshold_lookup = {(r["method"], r["query_rule"]): r["threshold"] for r in thresholds["records"]}
    counter = 0
    for method in METHODS:
        subset = frame.loc[frame.method.eq(method)]
        for query_rule in ("default", "any3"):
            if query_rule == "default":
                queries = subset.loc[subset.query_payload.eq(PAYLOADS[0])].copy()
            else:
                keys = ["image_id", "source_id", "collection", "label", "watermark_method", "embedded_payload", "condition"]
                queries = subset.groupby(keys, as_index=False).agg(score=("score", "max"), native_verifies=("native_verifies", "max"))
            for condition in [x[0] for x in CONDITIONS] + ["overall"]:
                considered = queries if condition == "overall" else queries.loc[queries.condition.eq(condition)]
                groups = {"photographs": considered.loc[considered.label.eq("0") & considered.watermark_method.eq("unmarked")],
                    "unmarked_AI": considered.loc[considered.label.eq("1") & considered.watermark_method.eq("unmarked")],
                    "marked_AI": considered.loc[considered.label.eq("1") & considered.watermark_method.eq(method) & (considered.embedded_payload.eq(PAYLOADS[0]) if query_rule == "default" else True)]}
                for population, inputs in groups.items():
                    for collection in ["all"] + sorted(set(inputs.collection)):
                        part = inputs if collection == "all" else inputs.loc[inputs.collection.eq(collection)]
                        for criterion in ("calibrated_margin", "native_rule"):
                            events = part.score.to_numpy() >= threshold_lookup[(method, query_rule)] if criterion == "calibrated_margin" else part.native_verifies.to_numpy()
                            counter += 1
                            record, draws = describe_events(part, events, population, 20261016)
                            record.update({"method": method, "query_rule": query_rule, "condition": condition, "collection": collection,
                                "criterion": criterion, "threshold": threshold_lookup[(method, query_rule)] if criterion == "calibrated_margin" else None})
                            measurements.append(record)
                            bootstrap_arrays[f"record_{counter:04d}"] = draws
    paired = []
    for query_rule in ("default", "any3"):
        for condition in [x[0] for x in CONDITIONS] + ["overall"]:
            for population in ("photographs", "unmarked_AI", "marked_AI"):
                matching = [(i + 1, r) for i, r in enumerate(measurements) if r["query_rule"] == query_rule and r["condition"] == condition and r["population"] == population and r["collection"] == "all" and r["criterion"] == "calibrated_margin"]
                a = next((i, r) for i, r in matching if r["method"] == "dwt_dct")
                b = next((i, r) for i, r in matching if r["method"] == "trustmark")
                delta = bootstrap_arrays[f"record_{b[0]:04d}"] - bootstrap_arrays[f"record_{a[0]:04d}"]
                paired.append({"query_rule": query_rule, "condition": condition, "population": population,
                    "TrustMark_minus_DWT_source_mean_rate": b[1]["source_mean_rate"] - a[1]["source_mean_rate"],
                    "paired_source_ci95": np.quantile(delta, [.025, .975]).tolist(), "conditional_on_frozen_calibration": True})
    manifest = read_csv(OUT / "manifest.csv")
    marked = manifest.loc[manifest.watermark_method.isin(METHODS)].copy()
    quality = []
    for (method, condition), part in marked.groupby(["watermark_method", "condition"]):
        record = {"method": method, "condition": condition, "sources": part.source_id.nunique(), "images": len(part)}
        for column in ("embedding_PSNR_dB", "embedding_SSIM", "base_encoder_PSNR_dB", "residual_gain", "PSNR_vs_original_dB", "SSIM_vs_original", "PSNR_vs_attack_input_dB", "SSIM_vs_attack_input"):
            values = part[column].astype(float)
            record[column] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}
        quality.append(record)
    pd.DataFrame(measurements).to_csv(OUT / "rates.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_method_differences.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_rates.npz", **bootstrap_arrays)
    write(OUT / "summary.json", {"completed_at_utc": utc(), "sources": 1004, "calibration_photographs": 400,
        "test_photographs": 400, "test_AI": 204, "images": 8912, "test_images": 7312,
        "nominal_validation_FPR": .05, "same_observed_test_FPR_guaranteed": False,
        "protocol_sha256": digest(OUT / "protocol.json"), "amendment_sha256": digest(OUT / "protocol_amendment.json"), "source_manifest_sha256": digest(OUT / "sources.csv"),
        "thresholds_sha256": digest(thresholds_path), "test_raw_sha256": digest(OUT / "test_raw.npz"),
        "score_definitions": json.loads((OUT / "protocol.json").read_text())["score"],
        "residual_rescaling_outside_official_configuration": True, "quantization": QUANTIZATION,
        "shared_source_payload_uniforms_for_methods": True, "bootstrap_draws": 2000,
        "calibration_threshold_uncertainty_not_bootstrapped": True, "source_reuse": True,
        "thresholds": thresholds["records"], "measurements": measurements, "paired_method_differences": paired, "quality": quality})
    print("Saved fair-mark analysis", len(measurements), "rate records", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "prepare-cpu", "prepare-learned", "derive-cpu", "derive", "decode-calibration", "calibrate", "decode-test", "analyze"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.stage == "select": select()
    elif args.stage == "prepare-cpu": prepare_cpu()
    elif args.stage == "prepare-learned": prepare_learned(args)
    elif args.stage == "derive-cpu": derive_cpu()
    elif args.stage == "derive": derive()
    elif args.stage == "decode-calibration": decode(args, "calibration")
    elif args.stage == "calibrate": calibrate()
    elif args.stage == "decode-test": decode(args, "test")
    else: analyze()


if __name__ == "__main__":
    main()
