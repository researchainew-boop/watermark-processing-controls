"""Build an audited public-item index without redistributing images or metadata.

The original generated-image export is reconstructed in the exact archive order
used by prepare_native_sources.py. Recreated JPEG95 pixels must match the saved
source audit; ordinal filenames alone are never accepted as sufficient evidence.
Only explicitly allowed public IDs, pseudonyms, fingerprints and splits are saved.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pyarrow
import pyarrow.parquet as pq
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs/phase4/reproduction_index"
FIELDS = ["source_id", "collection", "archive_filename", "archive_entry",
          "archive_row", "public_image_id", "raw_pixel_sha256",
          "standardized_pixel_sha256", "standardized_file_sha256",
          "retained", "original_split"]
ORIGINAL_FIELDS = ["source_id", "collection", "split", "retained"]
SEED_FIELDS = ["seed", "source_id", "collection", "split"]
MJ_CANONICAL = {
    "mj-000.parquet": ("test-00000-of-00030.parquet", "428ddebc9237666b6f49af3abf640619473bbfe18b61b6ce55377793d9ea720f"),
    "mj-001.parquet": ("test-00001-of-00030.parquet", "6b8084a67661592694442601f81a88040604371b42489891095b830ef2ac8a27"),
    "mj-002.parquet": ("test-00002-of-00030.parquet", "f025b8131f42b5e61bc3d0b01973adc3bd1e0f73eb8d6749cd64efa217e80979"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pixels_digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def collection(generator: str) -> str:
    return "coco" if generator == "real" else generator


def jpeg95_pixel_digest(image: Image.Image) -> str:
    image.load()
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    pixels = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError("JPEG replay decode failed")
    return pixels_digest(pixels)


def zip_candidates(archive: zipfile.ZipFile):
    for name in archive.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and "thumb" not in name.lower() and not name.endswith("/"):
            yield name


def replay_zip(path: Path, expected: dict[str, dict], stem: str,
               generator: str, start: int, limit: int) -> dict[str, dict]:
    mapped = {}
    with zipfile.ZipFile(path) as archive:
        for name in zip_candidates(archive):
            if len(mapped) == limit:
                break
            image = Image.open(io.BytesIO(archive.read(name)))
            image.load()
            if min(image.size) < 256:
                continue
            filename = f"{stem}_{start + len(mapped):05d}__{generator}.jpg"
            row = expected[filename]
            if jpeg95_pixel_digest(image) != row["raw_pixel_sha256"]:
                raise ValueError(f"Eligible export replay mismatch: {filename}")
            mapped[filename] = {"archive_filename": path.name, "archive_entry": name,
                                "archive_row": "", "public_image_id": ""}
    if len(mapped) != limit:
        raise ValueError("Insufficient eligible ZIP sources")
    print(f"Verified {path.name}: {len(mapped)} original exports", flush=True)
    return mapped


def replay_midjourney(expected: dict[str, dict]) -> dict[str, dict]:
    mapped = {}
    for alias, (canonical, _) in MJ_CANONICAL.items():
        parquet = pq.ParquetFile(RAW / "downloads" / alias)
        fields = parquet.schema_arrow.names
        image_column = next((name for name in ("image", "img", "jpg") if name in fields), None)
        if image_column is None:
            raise ValueError("Missing parquet image column")
        columns = [image_column] + (["label"] if "label" in fields else [])
        offset = 0
        for batch in parquet.iter_batches(batch_size=16, columns=columns):
            for index, item in enumerate(batch.to_pylist()):
                if len(mapped) == 1000:
                    return mapped
                if "label" in item and int(item["label"]) != 0:
                    continue
                cell = item[image_column]
                payload = cell.get("bytes") if isinstance(cell, dict) else cell
                if not isinstance(payload, (bytes, bytearray)) or not payload:
                    continue
                image = Image.open(io.BytesIO(payload))
                if min(image.size) < 256:
                    continue
                filename = f"midjourney_{len(mapped):05d}__midjourney.jpg"
                if jpeg95_pixel_digest(image) != expected[filename]["raw_pixel_sha256"]:
                    raise ValueError(f"Eligible parquet export replay mismatch: {filename}")
                mapped[filename] = {"archive_filename": canonical, "archive_entry": "",
                                    "archive_row": str(offset + index), "public_image_id": ""}
            offset += batch.num_rows
        print(f"Verified original exports through {canonical}: {len(mapped)}", flush=True)
    if len(mapped) != 1000:
        raise ValueError("Insufficient eligible Midjourney sources")
    return mapped


def base_fingerprints(path: Path, expected_pixels: str = "", expected_file: str = "") -> tuple[str, str]:
    if not path.is_file():
        raise ValueError("Missing standardized source")
    pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError("Unreadable standardized source")
    pixel_hash, file_hash = pixels_digest(pixels), digest(path)
    if expected_pixels and pixel_hash != expected_pixels:
        raise ValueError("Standardized pixel fingerprint mismatch")
    if expected_file and file_hash != expected_file:
        raise ValueError("Standardized file fingerprint mismatch")
    return pixel_hash, file_hash


def normalized_row(value: str) -> str:
    if value in ("", None):
        return ""
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ValueError("Noninteger public parquet row")
    return str(int(number))


def public_locator(row: dict[str, str]) -> dict[str, str]:
    origin = row["original_id"]
    kind = row["collection"]
    archive_name = Path(row.get("archive", "")).name if row.get("archive") else ""
    archive_entry, archive_row, public_id = "", "", ""
    if kind == "openimages":
        public_id = origin
    elif kind == "midjourney":
        local_name, ordinal = origin.split("::", 1)
        if archive_name != local_name or local_name not in MJ_CANONICAL:
            raise ValueError("Fresh parquet archive locator mismatch")
        archive_name = MJ_CANONICAL[local_name][0]
        archive_row = normalized_row(ordinal)
        if row.get("archive_row") and archive_row != normalized_row(row["archive_row"]):
            raise ValueError("Fresh parquet row mismatch")
        if int(archive_row) >= pq.ParquetFile(Path(row["archive"])).metadata.num_rows:
            raise ValueError("Fresh parquet row outside archive")
    else:
        if "::" in origin:
            prefix, origin = origin.split("::", 1)
            if prefix != archive_name:
                raise ValueError("Fresh ZIP archive locator mismatch")
        archive_entry = origin
        if kind == "coco":
            public_id = PurePosixPath(archive_entry).stem
    return {"archive_filename": archive_name, "archive_entry": archive_entry,
            "archive_row": archive_row, "public_image_id": public_id}


def validate_safe_rows(rows: list[dict]) -> None:
    identifiers = set()
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Public-index field allowlist mismatch")
        sid = row["source_id"]
        if not re.fullmatch(r"[0-9a-f]{16,64}", sid) or sid in identifiers:
            raise ValueError("Invalid or repeated source pseudonym")
        identifiers.add(sid)
        for field in ("raw_pixel_sha256", "standardized_pixel_sha256", "standardized_file_sha256"):
            if row[field] and not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                raise ValueError("Invalid image fingerprint")
        name, entry = row["archive_filename"], row["archive_entry"]
        if name and (Path(name).name != name or any(c in name for c in "\\/:")):
            raise ValueError("Unsafe archive filename")
        if entry and (entry.startswith("/") or "\\" in entry or ":" in entry or ".." in PurePosixPath(entry).parts):
            raise ValueError("Unsafe archive entry")
        if row["archive_row"] and not row["archive_row"].isdigit():
            raise ValueError("Invalid public row")
        if row["original_split"] not in ("", "train", "val", "test", "excluded"):
            raise ValueError("Invalid original split")
        if row["collection"] not in ("coco", "openimages", "stable-diffusion", "dalle-3", "midjourney"):
            raise ValueError("Unexpected collection")


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "index_audit.json"
    audit_path.write_text(json.dumps({"all_checks_passed": False, "status": "building", "files_sha256": {}}, indent=2), encoding="utf-8")
    inputs = {
        "original": ROOT / "outputs/conference_clean/source_audit.csv",
        "openimages": ROOT / "outputs/phase3/controls/openimages_sources.csv",
        "fresh192": ROOT / "outputs/extension/fresh/manifest.csv",
        "fresh400": ROOT / "outputs/phase4/fresh_replication/sources.csv",
        "five_seed_assignments": ROOT / "outputs/phase3/controls/split_seeds/assignments.csv",
    }
    datasets = {key: read_csv(path) for key, path in inputs.items()}
    expected_counts = {"original": 6000, "openimages": 1000, "fresh192": 192, "fresh400": 400, "five_seed_assignments": 34115}
    if {key: len(rows) for key, rows in datasets.items()} != expected_counts:
        raise ValueError("Unexpected input source/split counts")
    archive_sha = {}
    aliases = {}
    for alias, (canonical, expected_hash) in MJ_CANONICAL.items():
        actual = digest(RAW / "downloads" / alias)
        if actual != expected_hash:
            raise ValueError("Local Midjourney alias does not match canonical public shard")
        archive_sha[canonical] = actual
        aliases[alias] = canonical
    print("Verified all three canonical Midjourney TEST shard hashes", flush=True)
    old = datasets["original"]
    expected = {Path(row["path"]).name: row for row in old if row["generator"] != "real"}
    if len(expected) != 4000:
        raise ValueError("Original generated-image filename ambiguity")
    mapped = {}
    mapped.update(replay_zip(RAW / "downloads/part-000001.zip", expected, "sd", "stable-diffusion", 0, 1000))
    mapped.update(replay_zip(RAW / "downloads/part-000002.zip", expected, "sd", "stable-diffusion", 1000, 1000))
    mapped.update(replay_zip(RAW / "downloads/Dalle3-dataset.zip", expected, "dalle", "dalle-3", 0, 1000))
    mapped.update(replay_midjourney(expected))
    result, original_assignments = [], []
    with zipfile.ZipFile(RAW / "val2017.zip") as coco_archive:
        for row in old:
            sid = row["source_id"]
            if row["generator"] == "real":
                filename = Path(row["path"]).name
                entry = "val2017/" + filename
                pixels = cv2.imdecode(np.frombuffer(coco_archive.read(entry), np.uint8), cv2.IMREAD_COLOR)
                if pixels is None or pixels_digest(pixels) != row["raw_pixel_sha256"]:
                    raise ValueError("Original COCO public item fingerprint mismatch")
                locator = {"archive_filename": "val2017.zip", "archive_entry": entry,
                           "archive_row": "", "public_image_id": Path(filename).stem}
            else:
                locator = mapped[Path(row["path"]).name]
            base_hash, file_hash = base_fingerprints(ROOT / "data/processed/base" / f"{sid}.png", row["base_pixel_sha256"])
            retained = row["retained"] == "True"
            if row["retained"] not in ("True", "False"):
                raise ValueError("Invalid retained flag")
            kind = collection(row["generator"])
            result.append({"source_id": sid, "collection": kind, **locator,
                           "raw_pixel_sha256": row["raw_pixel_sha256"],
                           "standardized_pixel_sha256": base_hash, "standardized_file_sha256": file_hash,
                           "retained": retained, "original_split": row["split"]})
            original_assignments.append({"source_id": sid, "collection": kind, "split": row["split"], "retained": retained})
    for cohort in ("openimages", "fresh192", "fresh400"):
        for row in datasets[cohort]:
            locator = public_locator(row)
            base_hash, file_hash = base_fingerprints(Path(row["path"]), expected_file=row["output_file_sha256"])
            result.append({"source_id": row["source_id"], "collection": row["collection"], **locator,
                           "raw_pixel_sha256": row["raw_pixel_sha256"],
                           "standardized_pixel_sha256": base_hash, "standardized_file_sha256": file_hash,
                           "retained": True, "original_split": ""})
    archive_paths = {"val2017.zip": RAW / "val2017.zip",
                     "part-000001.zip": RAW / "downloads/part-000001.zip",
                     "part-000002.zip": RAW / "downloads/part-000002.zip",
                     "Dalle3-dataset.zip": RAW / "downloads/Dalle3-dataset.zip"}
    for cohort in ("fresh192", "fresh400"):
        for row in datasets[cohort]:
            if row["collection"] not in ("openimages", "midjourney"):
                archive_paths[Path(row["archive"]).name] = Path(row["archive"])
    for name, path in archive_paths.items():
        with zipfile.ZipFile(path) as archive:
            public_entries = set(archive.namelist())
            if any(row["archive_entry"] not in public_entries for row in result if row["archive_filename"] == name):
                raise ValueError("Public ZIP index contains an absent entry")
        archive_sha[name] = digest(path)
    validate_safe_rows(result)
    by_id = {row["source_id"]: row for row in result}
    if len(result) != 7592 or sum(row["retained"] for row in result) != 7415:
        raise ValueError("Unexpected public-index retained counts")
    seed_rows, seen = [], set()
    for row in datasets["five_seed_assignments"]:
        key = (row["seed"], row["source_id"])
        indexed = by_id[row["source_id"]]
        if key in seen or not indexed["retained"] or indexed["collection"] != collection(row["generator"]) or row["split"] not in ("train", "val", "test"):
            raise ValueError("Inconsistent repeated-split assignment")
        seen.add(key)
        seed_rows.append({"seed": row["seed"], "source_id": row["source_id"],
                          "collection": indexed["collection"], "split": row["split"]})
    seeds = sorted({row["seed"] for row in seed_rows})
    expected_pool = {row["source_id"] for row in result[:7000] if row["retained"]}
    if len(seeds) != 5 or any({row["source_id"] for row in seed_rows if row["seed"] == seed} != expected_pool for seed in seeds):
        raise ValueError("Repeated-split pool mismatch")
    files = {"public_source_index.csv": (FIELDS, sorted(result, key=lambda row: row["source_id"])),
             "original_split_assignments.csv": (ORIGINAL_FIELDS, sorted(original_assignments, key=lambda row: row["source_id"])),
             "five_seed_split_assignments.csv": (SEED_FIELDS, sorted(seed_rows, key=lambda row: (row["seed"], row["source_id"])))}
    for name, (fields, rows) in files.items():
        write_csv(OUT / name, fields, rows)
    audit = {
        "all_checks_passed": True, "status": "complete_unambiguous",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "counts": {"indexed_sources": len(result), "retained_sources": 7415,
                   "original_sources": 6000, "original_retained": 5823, "original_excluded": 177,
                   "additional_openimages": 1000, "fresh192": 192, "fresh400": 400,
                   "original_split_rows": len(original_assignments), "five_seed_split_rows": len(seed_rows),
                   "collections": dict(Counter(row["collection"] for row in result)),
                   "seed_pool_sources": len(expected_pool), "seeds": seeds},
        "checks": {"all_4000_original_ai_eligible_exports_pixel_verified": True,
                   "all_2000_original_coco_archive_pixels_verified": True,
                   "all_7592_standardized_pixel_and_file_hashes_verified": True,
                   "all_public_zip_entries_exist": True,
                   "canonical_midjourney_test_shards_sha256_verified": True,
                   "fresh_parquet_rows_in_bounds_and_manifest_consistent": True,
                   "source_pseudonyms_unique": True, "split_pools_and_allowlist_valid": True},
        "files_sha256": {name: digest(OUT / name) for name in files},
        "input_manifest_sha256": {key: digest(path) for key, path in inputs.items()},
        "public_archive_sha256": archive_sha,
        "local_midjourney_alias_to_public_basename": aliases,
        "reconstruction": "Original ZIP member order and parquet row order; minimum side 256; parquet label 0; RGB PIL JPEG95 export; decoded BGR shape plus bytes SHA256 must match original source audit.",
        "fingerprints": "Pixel fingerprints hash str(BGR_array.shape).encode() + BGR_array.tobytes(); raw original AI fingerprint is after native RGB JPEG95 export. Added-source raw fingerprints retain their saved native/rotation scope. Standardized file fingerprints identify existing base PNGs.",
        "split_scope": "Original assignments cover all 6000 selected records, including 177 retained=False exclusions. Five seeds cover the 5823 retained originals plus 1000 added Open Images sources. Both fresh samples have blank original_split and do not enter these training splits.",
        "public_locator_scope": "Parquet rows are zero based. Canonical Midjourney shards reside in the public dataset data directory. COCO and Open Images public image IDs are identifiers, not downloads. No paths, URLs, usernames, authors, prompts, captions, images or license grants are included.",
        "release_scope": "Local unpublished source index only. Reacquisition must follow the cited public collections and their terms; this artifact makes no redistribution or licensing assertion.",
        "library_versions": {"pyarrow": pyarrow.__version__, "pillow": pillow_version,
                             "opencv": cv2.__version__, "numpy": np.__version__},
        "script_sha256": digest(Path(__file__)),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": True, "counts": audit["counts"], "runtime_seconds": audit["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "index_audit.json").write_text(json.dumps({"all_checks_passed": False,
            "status": "incomplete_or_ambiguous", "error_type": type(error).__name__,
            "files_sha256": {}}, indent=2) + "\n", encoding="utf-8")
        raise
