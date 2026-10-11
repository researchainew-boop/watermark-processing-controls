"""Export public COCO/caption locators and seeds under a strict field allowlist.

Caption text, prompts, local paths and image bytes are never written. The
frozen paired-source preparation and generation locks remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5/paired_content"
FIELDS = ["pair_id", "real_source_id", "ai_source_id", "public_coco_image_id",
          "archive_filename", "archive_entry", "caption_annotation_id",
          "generation_seed", "split", "native_file_sha256",
          "real_standardized_pixel_sha256", "real_standardized_file_sha256"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    audit_path = OUT / "public_pair_index_audit.json"
    audit_path.write_text(json.dumps({"all_checks_passed": False, "status": "building", "files_sha256": {}}, indent=2), encoding="utf-8")
    selection = json.loads((OUT / "selection_complete.json").read_text(encoding="utf-8"))
    protocol_hash = digest(OUT / "protocol.json")
    selection_hash = digest(OUT / "selection_complete.json")
    sources_hash = digest(OUT / "source_pairs.csv")
    if protocol_hash != selection["protocol_sha256"] or sources_hash != selection["source_pairs_sha256"] or selection["pairs"] != 200:
        raise ValueError("Paired source/protocol lock mismatch")
    source_rows = read_csv(OUT / "source_pairs.csv")
    prior = read_csv(ROOT / "outputs/phase4/reproduction_index/public_source_index.csv")
    prior_coco = {row["public_image_id"] for row in prior if row["collection"] == "coco"}
    prior_pseudonyms = {row["source_id"] for row in prior}
    asset = ROOT / ".tools/extension-assets/annotations_trainval2017.zip"
    if digest(asset) != selection["caption_archive_sha256"]:
        raise ValueError("Official annotation archive changed")
    with zipfile.ZipFile(asset) as archive:
        caption_bytes = archive.read("annotations/captions_val2017.json")
    if hashlib.sha256(caption_bytes).hexdigest() != selection["captions_val2017_json_sha256"]:
        raise ValueError("Caption member changed")
    caption_ids = {}
    for row in json.loads(caption_bytes)["annotations"]:
        iid = str(int(row["image_id"])).zfill(12)
        caption_ids[iid] = min(caption_ids.get(iid, int(row["id"])), int(row["id"]))
    result = []
    with zipfile.ZipFile(ROOT / "data/raw/val2017.zip") as archive:
        for row in source_rows:
            iid = row["public_coco_image_id"]
            if not re.fullmatch(r"[0-9]{12}", iid) or iid in prior_coco:
                raise ValueError("Invalid or previously used public COCO ID")
            order = hashlib.sha256(f"paired-content|20261010|coco|{iid}".encode()).hexdigest()
            pair_id = order[:24]
            real_id = hashlib.sha256((pair_id + "|real").encode()).hexdigest()[:24]
            ai_id = hashlib.sha256((pair_id + "|generated").encode()).hexdigest()[:24]
            if [row["pair_id"], row["real_source_id"], row["ai_source_id"]] != [pair_id, real_id, ai_id]:
                raise ValueError("Pair/source pseudonym derivation mismatch")
            if real_id in prior_pseudonyms or ai_id in prior_pseudonyms:
                raise ValueError("Pair origin pseudonym collides with prior source")
            if int(row["generation_seed"]) != int(order[:16], 16) % (2 ** 31):
                raise ValueError("Fixed generation seed mismatch")
            if int(row["caption_annotation_id"]) != caption_ids[iid]:
                raise ValueError("Selected caption is not minimum annotation ID")
            entry = f"val2017/{iid}.jpg"
            if row["archive_filename"] != "val2017.zip" or row["archive_entry"] != entry:
                raise ValueError("Public archive locator mismatch")
            if hashlib.sha256(archive.read(entry)).hexdigest() != row["native_file_sha256"]:
                raise ValueError("Public native COCO byte fingerprint mismatch")
            if digest(Path(row["real_path"])) != row["standardized_file_sha256"]:
                raise ValueError("Standardized real-source file changed")
            output = {"pair_id": pair_id, "real_source_id": real_id, "ai_source_id": ai_id,
                      "public_coco_image_id": iid, "archive_filename": "val2017.zip", "archive_entry": entry,
                      "caption_annotation_id": str(int(row["caption_annotation_id"])),
                      "generation_seed": str(int(row["generation_seed"])), "split": "test",
                      "native_file_sha256": row["native_file_sha256"],
                      "real_standardized_pixel_sha256": row["standardized_pixel_sha256"],
                      "real_standardized_file_sha256": row["standardized_file_sha256"]}
            for field in ["native_file_sha256", "real_standardized_pixel_sha256", "real_standardized_file_sha256"]:
                if not re.fullmatch(r"[0-9a-f]{64}", output[field]):
                    raise ValueError("Invalid image fingerprint")
            if any(".." in PurePosixPath(output[field]).parts or "\\" in output[field] or ":" in output[field] for field in ["archive_filename", "archive_entry"]):
                raise ValueError("Unsafe public archive locator")
            if set(output) != set(FIELDS) or row["split"] != "test":
                raise ValueError("Public field/split allowlist mismatch")
            result.append(output)
    if len(result) != 200 or any(len({row[key] for row in result}) != 200 for key in ["pair_id", "real_source_id", "ai_source_id", "public_coco_image_id"]):
        raise ValueError("Public paired index is not 200 unique pairs")
    index_path = OUT / "public_pair_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(sorted(result, key=lambda row: row["pair_id"]))
    audit = {"all_checks_passed": True, "status": "complete_unambiguous",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {"pairs": 200, "planned_origins": 400, "planned_copies": 1200,
                   "prior_acquisition_records": 7592, "prior_coco_ids": len(prior_coco)},
        "fields": FIELDS, "files_sha256": {index_path.name: digest(index_path)},
        "protocol_sha256": protocol_hash, "selection_complete_sha256": selection_hash,
        "source_pairs_sha256": sources_hash,
        "caption_archive_sha256": selection["caption_archive_sha256"],
        "captions_val2017_json_sha256": selection["captions_val2017_json_sha256"],
        "checks": {"public_field_allowlist": True, "minimum_caption_id": True,
            "deterministic_seeds_and_pseudonyms": True, "prior_coco_id_and_source_pseudonym_intersections_zero": True,
            "native_archive_bytes_verified": True, "standardized_real_files_verified": True,
            "all_pairs_unique_and_test_only": True, "source_protocol_and_selection_unchanged": True},
        "scope": "Local unpublished public locators/fingerprints only. No caption or prompt text, local paths, URLs, usernames, author metadata or image bytes. Generated images and measurement completion are not claimed by this preparation index.",
        "reproduction": "Reacquire official COCO val2017 and captions_val2017.json; use each image ID and minimum caption annotation ID with the protocol's fixed Photograph-prefix and generation seed. Generation asset/model hashes are in the separate frozen protocol/lock.",
        "script_sha256": digest(Path(__file__))}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": True, "pairs": 200, "fields": FIELDS, "files_sha256": audit["files_sha256"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (OUT / "public_pair_index_audit.json").write_text(json.dumps({"all_checks_passed": False,
            "status": "incomplete_or_ambiguous", "error_type": type(error).__name__, "files_sha256": {}}, indent=2) + "\n", encoding="utf-8")
        raise
