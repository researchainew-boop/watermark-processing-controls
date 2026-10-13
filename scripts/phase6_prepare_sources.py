"""Prepare source-disjoint Phase 6 calibration/test inputs without inference.

Only public source availability, dimensions, metadata, and predeclared duplicate
screening affect eligibility. This script imports no detector or watermark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/phase6/runtime-deps"))
sys.path.insert(0, str(ROOT / ".tools/extension-deps"))
import cv2
import numpy as np
import pyarrow as pa
import requests

cv2.setNumThreads(1)
OUT = ROOT / "outputs/phase6/data_inventory"
DATA = ROOT / "data/phase6"
SEED = 20261011
REPO = "nebula/GenImage-arrow"
REVISION = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"
FAMILIES = {"BigGAN": "GAN", "ADM": "pixel-space guided diffusion",
            "VQDM": "discrete latent diffusion"}
SOURCE_TABLES = [
    "outputs/conference_clean/source_audit.csv",
    "outputs/extension/fresh/manifest.csv",
    "outputs/phase3/controls/openimages_sources.csv",
    "outputs/phase4/fresh_replication/sources.csv",
    "outputs/phase5/paired_content/exclusion_manifest.csv",
    "outputs/phase5/paired_content/source_pairs.csv",
    "outputs/phase5/paired_content/generated_sources.csv",
    "outputs/phase5/fair_watermarks/base_sources.csv",
    "outputs/phase5/calibration/calibration_extra_manifest.csv",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixel_digest(pixels):
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def order(collection, key):
    return hashlib.sha256(f"phase6|{SEED}|{collection}|{key}".encode()).hexdigest()


def ahash(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((small >= small.mean()).ravel()).tobytes(), "big")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def request(url):
    return urllib.request.Request(url, headers={"User-Agent": "phase6-reproducibility/1.0"})


def download(url, target, expected_sha=None, expected_size=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_size is not None and target.stat().st_size != expected_size:
            raise ValueError(f"Existing public asset has wrong size: {target.name}")
        if expected_sha is not None and digest(target) != expected_sha:
            raise ValueError(f"Existing public asset has wrong digest: {target.name}")
        return target
    partial = target.with_suffix(target.suffix + ".partial")
    if expected_size is not None and expected_size > 64 * 1024 * 1024:
        # Public CDN full-body requests can stall. Bounded checked byte ranges
        # preserve the exact commit-pinned file while supporting safe retries.
        range_download(url, partial, expected_size)
        if digest(partial) != expected_sha:
            raise ValueError(f"Downloaded public asset has wrong digest: {target.name}")
        partial.replace(target)
        print("Acquired public asset", target.name, target.stat().st_size, flush=True)
        return target
    with urllib.request.urlopen(request(url), timeout=120) as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ValueError(f"Downloaded public asset has wrong size: {target.name}")
    if expected_sha is not None and digest(partial) != expected_sha:
        raise ValueError(f"Downloaded public asset has wrong digest: {target.name}")
    partial.replace(target)
    print("Acquired public asset", target.name, target.stat().st_size, flush=True)
    return target


def range_download(url, partial, size):
    parts = partial.parent / (partial.name + ".ranges")
    parts.mkdir(parents=True, exist_ok=True)
    step = 16 * 1024 * 1024
    starts = list(range(0, size, step))
    def acquire(start):
        end = min(start + step, size) - 1
        target = parts / f"{start:012d}.bin"
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(3):
            try:
                public_url = url + ("&" if "?" in url else "?") + f"download=true&phase6_range={start}"
                with requests.get(public_url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(20, 60), stream=True) as response:
                    if response.status_code != 206 or response.headers.get("content-range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Public CDN returned an unexpected byte range/status")
                    with target.open("wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                if target.stat().st_size != end - start + 1:
                    raise ValueError("Public range response has wrong length")
                return target
            except Exception:
                if attempt == 2:
                    raise RuntimeError(f"Could not acquire public byte range at offset {start}") from None
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        acquired = list(pool.map(acquire, starts))
    with partial.open("wb") as stream:
        for path in acquired:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
    if partial.stat().st_size != size:
        raise ValueError("Joined public archive has wrong length")
    # A completed file is checked by the caller before being accepted; chunks
    # remain available for transport recovery until that exact hash succeeds.


def canonical_collection(row):
    if row.get("real_source_id"):
        return "coco"
    value = row.get("collection") or row.get("generator") or ""
    return "coco" if value in {"real", "real_coco", "coco"} else value


def public_id(collection, row):
    keys = ["public_coco_image_id", "public_image_id", "original_id", "ImageID",
            "archive_entry", "native_path", "path"]
    for key in keys:
        value = row.get(key, "")
        if collection == "coco":
            matches = re.findall(r"(?<!\d)(\d{12})(?!\d)", value)
            if matches:
                return matches[-1]
        elif collection == "openimages":
            matches = re.findall(r"(?<![0-9a-f])([0-9a-f]{16})(?![0-9a-f])", value)
            if matches:
                return matches[-1]
    return row.get("original_id") or row.get("source_id") or ""


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if (OUT / "sampling_protocol.json").exists():
        old = json.loads((OUT / "sampling_protocol.json").read_text())
        if old["seed"] != SEED or old["genimage_revision"] != REVISION:
            raise ValueError("Existing sampling protocol differs")
        return
    prior = []
    files = {}
    for relative in SOURCE_TABLES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        files[relative] = digest(path)
        for row in read_csv(path):
            collection = canonical_collection(row)
            identity = row.get("source_id") or row.get("real_source_id") or row.get("ai_source_id")
            hashes = [row.get(key, "") for key in ["raw_pixel_sha256", "native_pixel_sha256",
                       "base_pixel_sha256", "standardized_pixel_sha256", "ai_standardized_pixel_sha256"]]
            coarse = row.get("ahash") or row.get("prejpeg_ahash") or row.get("native_ahash")
            native = row.get("ai_native_path")
            base = row.get("ai_path")
            if native and not coarse:
                image = cv2.imread(native, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot audit preceding generated source: {identity}")
                hashes.append(pixel_digest(image))
                coarse = str(ahash(image))
                if base:
                    hashes.append(pixel_digest(cv2.imread(base, cv2.IMREAD_COLOR)))
                collection = "realistic_vision_lcm"
            prior.append({"source_id": identity or "", "collection": collection,
                          "public_id": public_id(collection, row),
                          "pixel_sha256": "|".join(x for x in hashes if x),
                          "ahash": coarse or "", "source_table": relative})
    unique = list({(row["source_id"], row["collection"], row["public_id"], row["pixel_sha256"],
                    row["ahash"]): row for row in prior}.values())
    write_csv(OUT / "preceding_source_exclusions.csv", unique)
    write_json(OUT / "sampling_protocol.json", {
        "recorded_at_utc": utc(), "seed": SEED, "planned_sources": {
            "calibration": {"coco": 400, "openimages": 400},
            "test": {"coco": 200, "openimages": 200, "BigGAN": 200, "ADM": 200, "VQDM": 200}},
        "genimage_repository": REPO, "genimage_revision": REVISION,
        "family_definitions": FAMILIES,
        "selection": "SHA256(phase6|seed|collection|public ID) order; first eligible sources until fixed quota",
        "eligibility": "Known source-ID exclusion, decode availability, native minimum side 128, exact native/base pixel hash exclusion, 64-bit average hash Hamming distance <=2 exclusion against preceding and selected sources",
        "real_role_assignment": "Within each real collection, first 400 eligible ordered IDs are calibration, next 200 test",
        "processing": "OpenCV BGR decode; Open Images metadata CCW rotation; INTER_AREA resize 256 square; common OpenCV JPEG95 encode/decode; PNG storage",
        "scores_observed": False, "detector_or_watermark_imported": False,
        "final_test_may_be_scored_only_after_method_lock": True,
        "study_fit_families": ["Stable Diffusion", "DALL-E3"],
        "preceding_test_families": ["Midjourney", "Realistic Vision SD1.x/LCM"],
        "exposure_caveat": "Unseen refers to study supervised fitting/development. Public benchmark or pretraining exposure of frozen backbones and external detectors is not excluded.",
        "content_caveat": "GenImage ImageNet-class generation is not COCO-caption-paired; common JPEG processing removes file encoding differences but not collection/content differences.",
        "preceding_tables_sha256": files,
        "exclusion_manifest_sha256": digest(OUT / "preceding_source_exclusions.csv"),
        "script_sha256_at_sampling": digest(__file__)})
    print("Recorded score-free sampling protocol; preceding source records", len(unique), flush=True)


def supplement_observed_exclusions():
    """Include available originals from superseded pilot/demo experiments."""
    path = OUT / "preceding_source_exclusions.csv"
    prior = read_csv(path)
    known = {row["source_id"] for row in prior}
    extras = []
    for image_path in sorted((ROOT / "data/processed/base").glob("*.png")):
        if image_path.stem not in known:
            extras.append((image_path.stem, "legacy_processed_original", image_path))
    for row in read_csv(ROOT / "data/demo_manifest.csv"):
        if row["source_id"] not in known:
            extras.append((row["source_id"], "demonstration_original", Path(row["path"])))
    for identity, collection, image_path in extras:
        if identity in known:
            continue
        pixels = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError("Available preceding pilot/demo original failed decoding")
        prior.append({"source_id": identity, "collection": collection,
                      "public_id": image_path.name, "pixel_sha256": pixel_digest(pixels),
                      "ahash": str(ahash(pixels)), "source_table": str(image_path.relative_to(ROOT))})
        known.add(identity)
    # Correct source-pair metadata namespace without changing any pixel hashes.
    pairs = {row["real_source_id"]: row for row in read_csv(ROOT / "outputs/phase5/paired_content/source_pairs.csv")}
    for row in prior:
        if row["source_table"] == "outputs/phase5/paired_content/source_pairs.csv":
            pair = pairs[row["source_id"]]
            row.update(collection="coco", public_id=pair["public_coco_image_id"])
    write_csv(path, prior)
    protocol_path = OUT / "sampling_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    amendments = protocol.setdefault("prescore_amendments", [])
    if not any(item.get("name") == "available-pilot-demo-exclusions" for item in amendments):
        amendments.append({"name": "available-pilot-demo-exclusions", "recorded_at_utc": utc(),
                           "additional_available_originals": len(extras),
                           "source_pair_public_ID_namespace_corrected": True,
                           "scores_observed": False,
                           "reason": "Audit all available preceding study/pilot/demo originals, including the superseded CIFAKE pilot retained in processed/base"})
    protocol["exclusion_manifest_sha256"] = digest(path)
    protocol["preceding_audit_records"] = len(prior)
    write_json(protocol_path, protocol)
    print("Supplemented all available preceding originals", len(extras), "audit records", len(prior), flush=True)


def load_screen():
    rows = read_csv(OUT / "preceding_source_exclusions.csv")
    selected = read_csv(OUT / "sources.partial.csv") if (OUT / "sources.partial.csv").exists() else []
    ids = {(row["collection"], row["public_id"]) for row in rows}
    hashes = {h for row in rows for h in row["pixel_sha256"].split("|") if h}
    coarse = {int(row["ahash"]) for row in rows if row["ahash"]}
    for row in selected:
        ids.add((row["collection"], row["public_id"]))
        hashes.update([row["raw_pixel_sha256"], row["base_pixel_sha256"]])
        if row.get("diagnostic_base_pixel_sha256"):
            hashes.add(row["diagnostic_base_pixel_sha256"])
        coarse.add(int(row["ahash"]))
    return selected, ids, hashes, coarse


def consider(collection, key, payload, role, selected, ids, hashes, coarse, metadata):
    if (collection, key) in ids:
        return "preceding_or_selected_public_id"
    raw = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        return "decode_failure"
    if min(raw.shape[:2]) < 128:
        return "native_minimum_side_below_128"
    rotation = float(metadata.get("Rotation") or 0)
    if rotation not in (0, 90, 180, 270):
        return "unsupported_metadata_rotation"
    if rotation:
        raw = np.rot90(raw, int(rotation // 90)).copy()
    pre = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", pre, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Common JPEG encoding failed")
    base = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    raw_hash, base_hash, h = pixel_digest(raw), pixel_digest(base), ahash(pre)
    if raw_hash in hashes or base_hash in hashes:
        return "exact_native_or_standardized_pixel_overlap"
    if any((h ^ old).bit_count() <= 2 for old in coarse):
        return "average_hash_HD_le_2_overlap"
    identity = order(collection, key)[:24]
    native = DATA / "native" / collection / f"{identity}.png"
    target = DATA / "base_jpeg95" / collection / f"{identity}.png"
    native.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(native), raw) or not cv2.imwrite(str(target), base):
        raise RuntimeError("Source image storage failed")
    row = {"source_id": identity, "image_id": identity, "collection": collection,
           "generator": collection, "label": int(collection in FAMILIES),
           "role": role, "split": role, "public_id": key, "original_id": key,
           "native_path": str(native), "file_sha256": digest(native),
           "input_file_sha256": hashlib.sha256(payload).hexdigest(),
           "raw_pixel_sha256": raw_hash, "ahash": str(h),
           "native_height": raw.shape[0], "native_width": raw.shape[1],
           "base_path": str(target), "path": str(target),
           "output_sha256": digest(target), "base_pixel_sha256": base_hash,
           "jpeg_quality": 95, "family": FAMILIES.get(collection, "photograph"),
           **metadata}
    selected.append(row)
    ids.add((collection, key))
    hashes.update([raw_hash, base_hash])
    coarse.add(h)
    return ""


def genimage_metadata():
    for family in FAMILIES:
        target = DATA / "public_assets" / "genimage" / family
        target.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data/test/{family}"
        listing = json.load(urllib.request.urlopen(request(url), timeout=60))
        write_json(target / "hub_tree.json", listing)
        for name in ["mapping.json", "dataset_info.json", "bundle_manifest.json", "state.json"]:
            download(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/test/{family}/{name}", target / name)
    download("https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/License",
             DATA / "public_assets/genimage/upstream_License.txt")
    download(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md",
             DATA / "public_assets/genimage/mirror_README.md")


def prepare_ai_assets():
    """Acquire AI assets only; safe alongside separate real-source sampling."""
    genimage_metadata()
    for family in FAMILIES:
        folder = DATA / "public_assets/genimage" / family
        mapping = json.loads((folder / "mapping.json").read_text())
        listing = json.loads((folder / "hub_tree.json").read_text())
        shards = sorted((entry for entry in listing if entry["path"].endswith(".arrow")), key=lambda entry: entry["path"])
        total_rows = json.loads((folder / "bundle_manifest.json").read_text())["rows"]
        rows_per_shard = (total_rows + len(shards) - 1) // len(shards)
        relevant = {row // rows_per_shard for name, row in mapping.items() if "/ai/" in name}
        def acquire(entry):
            return download(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}",
                            folder / Path(entry["path"]).name, entry["lfs"]["oid"], entry["size"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(acquire, [entry for i, entry in enumerate(shards) if i in relevant]))
        print("Public AI family assets hash-verified", family, flush=True)


def prepare_ai(families=None):
    genimage_metadata()
    selected, ids, hashes, coarse = load_screen()
    rejected = read_csv(OUT / "selection_rejections.csv") if (OUT / "selection_rejections.csv").exists() else []
    for family in families or FAMILIES:
        count = sum(row["collection"] == family for row in selected)
        if count == 200:
            continue
        folder = DATA / "public_assets/genimage" / family
        mapping = json.loads((folder / "mapping.json").read_text())
        features = json.loads((folder / "dataset_info.json").read_text())["features"]
        generator_code = features["generator"]["names"].index(family)
        names = sorted((name for name in mapping if "/ai/" in name), key=lambda name: order(family, name))
        listing = json.loads((folder / "hub_tree.json").read_text())
        shards = sorted((entry for entry in listing if entry["path"].endswith(".arrow")), key=lambda entry: entry["path"])
        # Initial acquisition covers trailing AI-bearing shards. Public shards
        # are bounded by bytes, so actual global row offsets are verified from
        # the first/last original image IDs rather than assuming equal rows.
        total_rows = json.loads((folder / "bundle_manifest.json").read_text())["rows"]
        rows_per_shard = (total_rows + len(shards) - 1) // len(shards)
        relevant = {mapping[name] // rows_per_shard for name in names}
        def acquire(entry):
            name = Path(entry["path"]).name
            return download(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}",
                            folder / name, entry["lfs"]["oid"], entry["size"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            acquired = list(pool.map(acquire, [entry for i, entry in enumerate(shards) if i in relevant]))
        tables, offsets = {}, {}
        for i, entry in enumerate(shards):
            if i in relevant:
                with pa.memory_map(str(folder / Path(entry["path"]).name), "r") as stream:
                    tables[i] = pa.ipc.open_stream(stream).read_all()
                first = tables[i].column("image_path")[0].as_py()
                last = tables[i].column("image_path")[-1].as_py()
                offsets[i] = mapping[first]
                if mapping[last] != offsets[i] + len(tables[i]) - 1:
                    raise ValueError("Public shard original-ID row offsets disagree with mapping")
        for name in names:
            if count == 200:
                break
            if (family, name) in ids:
                continue
            ordinal = mapping[name]
            available = [i for i, table in tables.items() if offsets[i] <= ordinal < offsets[i] + len(table)]
            if len(available) != 1:
                raise ValueError("AI source row is not covered by the acquired public shards")
            shard_index = available[0]
            table = tables[shard_index]
            item = table.slice(ordinal - offsets[shard_index], 1).to_pylist()[0]
            if item["image_path"] != name or item["label"] != 1 or item["generator"] != generator_code:
                raise ValueError("Original public image ID or AI label disagrees with mapping")
            result = consider(family, name, item["image"], "test", selected, ids, hashes, coarse,
                              {"public_repository": REPO, "public_revision": REVISION,
                               "public_row": ordinal, "original_release": "GenImage official val/test",
                               "License": "CC BY-NC-SA 4.0 plus upstream dataset terms"})
            if result:
                rejected.append({"collection": family, "public_id": name, "reason": result})
            else:
                count += 1
        write_csv(OUT / "sources.partial.csv", selected)
        write_csv(OUT / "selection_rejections.csv", rejected)
        print("Prepared AI sources", family, count, "/ 200; no inference", flush=True)
        if count != 200:
            raise RuntimeError(f"Public family cannot fulfill quota: {family}")


def prepare_real():
    selected, ids, hashes, coarse = load_screen()
    rejected = read_csv(OUT / "selection_rejections.csv") if (OUT / "selection_rejections.csv").exists() else []
    archive = ROOT / "data/raw/val2017.zip"
    count = sum(row["collection"] == "coco" for row in selected)
    with zipfile.ZipFile(archive) as z:
        names = sorted([name for name in z.namelist() if name.endswith(".jpg")], key=lambda name: order("coco", Path(name).stem))
        for name in names:
            if count == 600:
                break
            key = Path(name).stem
            if ("coco", key) in ids:
                continue
            reason = consider("coco", key, z.read(name), "calibration" if count < 400 else "test",
                              selected, ids, hashes, coarse,
                              {"public_repository": "COCO val2017", "public_revision": "2017",
                               "archive_entry": name, "original_release": "COCO 2017 validation",
                               "License": "Individual Flickr image license; obtain original image metadata before redistribution"})
            if reason:
                rejected.append({"collection": "coco", "public_id": key, "reason": reason})
            else:
                count += 1
    write_csv(OUT / "sources.partial.csv", selected)
    write_csv(OUT / "selection_rejections.csv", rejected)
    print("Prepared fresh COCO sources", count, "/ 600; no inference", flush=True)
    if count != 600:
        raise RuntimeError("Fresh COCO quota unavailable")
    metadata = read_csv(ROOT / ".tools/extension-assets/openimages_test_metadata.csv")
    metadata.sort(key=lambda entry: order("openimages", entry["ImageID"]))
    count = sum(row["collection"] == "openimages" for row in selected)
    candidates = [entry for entry in metadata if ("openimages", entry["ImageID"]) not in ids]
    def fetch(entry):
        key = entry["ImageID"]
        target = DATA / "public_assets/openimages" / f"{key}.jpg"
        try:
            download(f"https://open-images-dataset.s3.amazonaws.com/test/{key}.jpg", target)
            return entry, target, ""
        except Exception as exc:
            return entry, None, "download_failure:" + type(exc).__name__
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(candidates), 16):
            if count == 600:
                break
            for entry, path, error in pool.map(fetch, candidates[start:start + 16]):
                if count == 600:
                    break
                key = entry["ImageID"]
                if error:
                    rejected.append({"collection": "openimages", "public_id": key, "reason": error})
                    continue
                reason = consider("openimages", key, path.read_bytes(), "calibration" if count < 400 else "test",
                                  selected, ids, hashes, coarse,
                                  {**entry, "public_repository": "Open Images V7 test", "public_revision": "test-images-with-rotation.csv 2018_04",
                                   "original_release": "Official unsigned S3 Open Images test",
                                   "download_url": f"https://open-images-dataset.s3.amazonaws.com/test/{key}.jpg"})
                if reason:
                    rejected.append({"collection": "openimages", "public_id": key, "reason": reason})
                else:
                    count += 1
            write_csv(OUT / "sources.partial.csv", selected)
            write_csv(OUT / "selection_rejections.csv", rejected)
            print("Prepared fresh Open Images", count, "/ 600; no inference", flush=True)
    if count != 600:
        raise RuntimeError("Fresh Open Images quota unavailable")


def complete():
    selected = read_csv(OUT / "sources.partial.csv")
    if {row.get("primary_processing") for row in selected} != {"common128-before-marking"}:
        raise ValueError("All fixed sources must receive primary common128 processing before final completion")
    if len({row["source_id"] for row in selected}) != len(selected):
        raise ValueError("Source identifiers are not unique")
    for collection in ["coco", "openimages", *FAMILIES]:
        planned = 600 if collection in {"coco", "openimages"} else 200
        if sum(row["collection"] == collection for row in selected) != planned:
            raise ValueError(f"Fixed quota incomplete: {collection}")
    prior = read_csv(OUT / "preceding_source_exclusions.csv")
    preceding_ids = {(row["collection"], row["public_id"]) for row in prior}
    preceding_pixels = {value for row in prior for value in row["pixel_sha256"].split("|") if value}
    preceding_coarse = {int(row["ahash"]) for row in prior if row["ahash"]}
    selected_ids, selected_pixels, selected_coarse = set(), set(), set()
    minimum_hd = 65
    for row in selected:
        identity = (row["collection"], row["public_id"])
        pixels = {row["raw_pixel_sha256"], row.get("diagnostic_base_pixel_sha256") or row["base_pixel_sha256"]}
        h = int(row["ahash"])
        distances = [(h ^ value).bit_count() for value in preceding_coarse | selected_coarse]
        nearest = min(distances)
        minimum_hd = min(minimum_hd, nearest)
        if identity in preceding_ids | selected_ids or pixels & (preceding_pixels | selected_pixels) or nearest <= 2:
            raise ValueError("A finalized source violates the preceding/within-new overlap exclusion")
        selected_ids.add(identity)
        selected_pixels.update(pixels)
        selected_coarse.add(h)
    write_json(OUT / "source_overlap_audit.json", {
        "recorded_at_utc": utc(), "preceding_records": len(prior),
        "selected_source_count": len(selected), "public_ID_overlap": 0,
        "exact_native_or_direct256_pixel_overlap": 0,
        "average_hash_HD_le2_overlap": 0,
        "minimum_average_hash_HD_to_preceding_or_earlier_selected": minimum_hd,
        "exclusion_manifest_sha256": digest(OUT / "preceding_source_exclusions.csv"),
        "scores_observed": False})
    target = OUT / "primary_source_manifest.csv"
    if target.exists():
        old = read_csv(target)
        if old != selected:
            raise ValueError("Existing primary source selection differs; refusing reselection")
    else:
        write_csv(target, selected)
    public_fields = ["source_id", "collection", "generator", "family", "label", "role",
                     "public_id", "public_repository", "public_revision", "public_row",
                     "original_release", "input_file_sha256", "raw_pixel_sha256",
                     "native_height", "native_width", "output_sha256", "base_pixel_sha256",
                     "diagnostic_output_sha256", "diagnostic_base_pixel_sha256",
                     "primary_processing", "common_bottleneck_pixels", "jpeg_quality", "License"]
    public_rows = [{key: row.get(key, "") for key in public_fields} for row in selected]
    with zipfile.ZipFile(ROOT / ".tools/extension-assets/annotations_trainval2017.zip") as archive:
        annotations = json.loads(archive.read("annotations/instances_val2017.json"))
    image_lookup = {str(image["id"]).zfill(12): image for image in annotations["images"]}
    license_lookup = {entry["id"]: entry for entry in annotations["licenses"]}
    for row in public_rows:
        if row["collection"] == "coco":
            row["License"] = license_lookup[image_lookup[row["public_id"]]["license"]]["url"]
    write_csv(OUT / "public_source_index.csv", public_rows)
    for row in selected:
        if digest(row["native_path"]) != row["file_sha256"] or digest(row["base_path"]) != row["output_sha256"]:
            raise ValueError("Selected source bytes changed")
    write_json(OUT / "selection_complete.json", {
        "completed_at_utc": utc(), "sources": len(selected), "real_calibration_sources": 800,
        "real_test_sources": 400, "ai_test_sources": 600, "AI_per_family": 200,
        "manifest_sha256": digest(target), "sampling_protocol_sha256": digest(OUT / "sampling_protocol.json"),
        "primary_processing": "common128-before-marking",
        "calibration_manifest_sha256": digest(OUT / "calibration_source_manifest.csv"),
        "scores_observed": False, "detector_or_watermark_inference_performed": False,
        "scores_observed_scope": "No test/model score is used or computed by this source-preparation driver. Independent calibration inference by the method owner is allowed after delivery of its immutable source manifest.",
        "source_preparation_used_any_model_score": False,
        "test_inference_performed_by_preparation": False,
        "method_lock_required_before_test_inference": True,
        "script_sha256": digest(__file__), "OpenCV_version": cv2.__version__,
        "raw_assets_bytes": sum(p.stat().st_size for p in (DATA / "public_assets").rglob("*") if p.is_file()),
        "stored_native_and_common_JPEG95_bytes": sum(p.stat().st_size for key in ["native", "base_jpeg95"] for p in (DATA / key).rglob("*.png")),
        "precision_limits": "At true FPR 5%, 400 final independent photographs yield approximate SE 1.09 percentage points; per-source-collection n=200 gives SE 1.54 points. TPR worst-case SE at n=200 per family is 3.54 points. Edited versions do not multiply independent source count. Zero of 200 events has a two-sided 95% binomial upper bound about 1.83%. Calibration empirical quantile granularity is 1/801 pooled or 1/401 per real collection, conditional on exchangeability."})
    print("Completed locked score-free source inventory", len(selected), flush=True)


def common128():
    """Prescore resolution-control amendment; fixed selected originals only."""
    initial_manifest = OUT / "primary_source_manifest.csv"
    if not initial_manifest.exists():
        initial_manifest = OUT / "sources.partial.csv"
    selected = read_csv(initial_manifest)
    protocol_path = OUT / "sampling_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    before_sha = digest(initial_manifest)
    protocol.setdefault("prescore_amendments", [])
    marker = "common128-before-marking"
    if not any(row.get("name") == marker for row in protocol["prescore_amendments"]):
        protocol["prescore_amendments"].append({
            "name": marker, "recorded_at_utc": utc(),
            "prior_manifest_sha256": before_sha,
            "reason": "BigGAN native outputs are 128 pixels; use the same 128-pixel bottleneck for every calibration/test origin before watermarking",
            "source_ids_or_roles_changed": False,
            "scores_observed": False,
            "primary_processing": "OpenCV decoded/oriented native BGR -> INTER_AREA 128x128 -> INTER_LINEAR 256x256 -> common JPEG95 encode/decode -> PNG",
            "diagnostic_processing": "Native directly INTER_AREA 256x256 -> same JPEG95 encode/decode -> PNG",
            "screening_processing_unchanged": "Duplicate screening uses original native and native-to256 standardized hashes; no resampling of fixed selected IDs"})
    protocol.setdefault("original_direct256_processing", protocol["processing"])
    protocol["processing"] = "Primary: decoded/oriented native BGR -> INTER_AREA 128x128 -> INTER_LINEAR 256x256 -> common OpenCV JPEG95 encode/decode -> PNG"
    protocol["diagnostic_processing"] = "Decoded/oriented native BGR -> INTER_AREA 256x256 -> same OpenCV JPEG95 encode/decode -> PNG"
    protocol["negative_label_caveat"] = "COCO/Open Images source-collection ground truth, not an individual forensic certificate of camera capture or photographic style"
    protocol["score_separation"] = "Source sampling never uses model scores. Independent real calibration may be scored after its immutable manifest is delivered; no final test score is computed by this preparation driver, and final test inference requires a separately locked method."
    write_json(protocol_path, protocol)
    for i, row in enumerate(selected):
        native = cv2.imread(row["native_path"], cv2.IMREAD_COLOR)
        if native is None or digest(row["native_path"]) != row["file_sha256"]:
            raise ValueError("Selected native source changed before resolution amendment")
        diagnostic_path = row.get("diagnostic_base_path") or row["base_path"]
        diagnostic_sha = row.get("diagnostic_output_sha256") or row["output_sha256"]
        diagnostic_pixel_sha = row.get("diagnostic_base_pixel_sha256") or row["base_pixel_sha256"]
        if digest(diagnostic_path) != diagnostic_sha:
            raise ValueError("Existing direct256 diagnostic changed")
        low = cv2.resize(native, (128, 128), interpolation=cv2.INTER_AREA)
        up = cv2.resize(low, (256, 256), interpolation=cv2.INTER_LINEAR)
        ok, encoded = cv2.imencode(".jpg", up, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("Primary common128 JPEG encoding failed")
        base = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        path = DATA / "base_common128_jpeg95" / row["collection"] / f"{row['source_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), base):
            raise ValueError("Primary common128 base write failed")
        row.update({"diagnostic_base_path": diagnostic_path,
                    "diagnostic_output_sha256": diagnostic_sha,
                    "diagnostic_base_pixel_sha256": diagnostic_pixel_sha,
                    "base_path": str(path), "path": str(path),
                    "output_sha256": digest(path), "base_pixel_sha256": pixel_digest(base),
                    "primary_processing": marker, "common_bottleneck_pixels": 128})
        if i % 200 == 0:
            print("Common128 prescore processing", i + 1, "/", len(selected), flush=True)
    full = len(selected) == 1800
    write_csv(OUT / ("primary_source_manifest.csv" if full else "processed_source_manifest.partial.csv"), selected)
    write_csv(OUT / "sources.partial.csv", selected)
    calibration = [row for row in selected if row["role"] == "calibration"]
    if len(calibration) == 800:
        calibration_path = OUT / "calibration_source_manifest.csv"
        if not calibration_path.exists():
            write_csv(calibration_path, calibration)
        else:
            existing = {row["source_id"]: row for row in read_csv(calibration_path)}
            current = {row["source_id"]: row for row in calibration}
            critical = ["source_id", "collection", "generator", "label", "role", "public_id",
                        "native_path", "file_sha256", "base_path", "output_sha256",
                        "base_pixel_sha256", "diagnostic_base_path", "diagnostic_output_sha256",
                        "primary_processing"]
            if set(existing) != set(current) or any(
                    str(existing[key][field]) != str(current[key][field])
                    for key in existing for field in critical):
                raise ValueError("Independently published calibration source manifest changed")
    if full:
        complete()
    report_path = OUT / ("selection_complete.json" if full else "primary_preprocessing.partial.json")
    report = json.loads(report_path.read_text()) if report_path.exists() else {"scores_observed": False}
    report.update({"primary_processing": marker,
                   "diagnostic_processing": "native-to256-JPEG95",
                   "primary_and_diagnostic_views": len(selected) * 2,
                   "source_ids_or_roles_changed_in_amendment": False,
                   "all_fixed_source_quotas_completed": full,
                   "calibration_sources_available": len(calibration),
                   "calibration_manifest_sha256": digest(OUT / "calibration_source_manifest.csv") if len(calibration) == 800 else None,
                   "stored_native_and_primary_and_diagnostic_bytes": sum(
                       p.stat().st_size for key in ["native", "base_jpeg95", "base_common128_jpeg95"]
                       for p in (DATA / key).rglob("*.png"))})
    write_json(report_path, report)
    # Retain COCO's actual per-image Flickr license metadata separately. No ID
    # eligibility or role assignment changes are made from this annotation.
    with zipfile.ZipFile(ROOT / ".tools/extension-assets/annotations_trainval2017.zip") as archive:
        annotations = json.loads(archive.read("annotations/instances_val2017.json"))
    image_lookup = {str(image["id"]).zfill(12): image for image in annotations["images"]}
    license_lookup = {entry["id"]: entry for entry in annotations["licenses"]}
    license_rows = []
    for row in selected:
        if row["collection"] != "coco":
            continue
        image = image_lookup[row["public_id"]]
        license_entry = license_lookup[image["license"]]
        license_rows.append({"source_id": row["source_id"], "public_id": row["public_id"],
                             "role": row["role"], "license_id": image["license"],
                             "License": license_entry["url"], "license_name": license_entry["name"],
                             "original_flickr_url": image["flickr_url"], "coco_url": image["coco_url"],
                             "license_annotation_source": "COCO instances_val2017.json"})
    write_csv(OUT / "coco_image_licenses.csv", license_rows)
    print("Finalized score-free common128 primary and direct256 diagnostics", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["inventory", "ai-assets", "ai", "real", "complete", "common128", "all"], default="all")
    parser.add_argument("--family", action="append", choices=list(FAMILIES),
                        help="Acquire/select only specified AI family; full completion still requires all fixed quotas")
    args = parser.parse_args()
    inventory()
    if args.stage not in {"ai-assets", "inventory"}:
        supplement_observed_exclusions()
    if args.stage == "ai-assets":
        prepare_ai_assets()
    if args.stage in {"real", "all"}:
        prepare_real()
    if args.stage in {"ai", "all"}:
        prepare_ai(args.family)
    if args.stage == "complete":
        complete()
    if args.stage in {"common128", "all"}:
        common128()


if __name__ == "__main__":
    main()
