"""Freeze and select unused COCO caption pairs; GPU stages are scheduled separately.

Selection uses public IDs and pixel/hash overlap only, never detector responses.
This preparation script does not run generation or feature extraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5/paired_content"
DATA = ROOT / "data/phase5/paired_content"
RAW = ROOT / "data/raw"
ASSET = ROOT / ".tools/extension-assets/annotations_trainval2017.zip"
CAPTION_MEMBER = "annotations/captions_val2017.json"
CACHE = Path.home() / ".cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
CAL = ROOT / "outputs/phase4/domain_generalization/frozen_seed_thresholds.json"
HEADS = ROOT / "outputs/phase3/controls/split_seeds/20260920"
SEED = 20261010
COUNT = 200
ADAPTER = ROOT / ".tools/extension-assets/lcm/pytorch_lora_weights.safetensors"
ADAPTER_REVISION = "cf2fced511dbe7e26c8d1d397e728fbab875db4b"
ADAPTER_SHA256 = "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4"
cv2.setNumThreads(1)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty preparation artifact")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def order(image_id):
    return hashlib.sha256(f"paired-content|{SEED}|coco|{image_id}".encode()).hexdigest()


def ahash(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((small >= small.mean()).ravel()).tobytes(), "big")


def pixel_digest(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def frozen_input_paths():
    return {
        "original_source_audit": ROOT / "outputs/conference_clean/source_audit.csv",
        "openimages_sources": ROOT / "outputs/phase3/controls/openimages_sources.csv",
        "fresh192": ROOT / "outputs/extension/fresh/manifest.csv",
        "fresh400": ROOT / "outputs/phase4/fresh_replication/sources.csv",
        "public_index": ROOT / "outputs/phase4/reproduction_index/public_source_index.csv",
        "public_index_audit": ROOT / "outputs/phase4/reproduction_index/index_audit.json",
        "frozen_validation_calibration": CAL,
    }


def validate_heads():
    calibration = json.loads(CAL.read_text(encoding="utf-8"))
    recorded = {Path(name).name: value for name, value in calibration["head_sha256"].items()}
    current = {name: digest(HEADS / name) for name in recorded}
    if current != recorded or len(current) != 4:
        raise ValueError("Previously frozen detector heads changed")
    return current


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    inputs = {name: digest(path) for name, path in frozen_input_paths().items()}
    heads = validate_heads()
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved["input_sha256"] != inputs or saved["head_sha256"] != heads:
            raise ValueError("Paired protocol's frozen inputs changed")
        if saved["generator"].get("adapter", {}).get("model") == "latent-consistency/lcm-lora-sdv1-5":
            if not saved["generator"]["adapter"].get("revision"):
                if (OUT / "source_pairs.csv").exists():
                    raise ValueError("Adapter identity must be bound before source selection")
                if not ADAPTER.is_file() or digest(ADAPTER) != ADAPTER_SHA256:
                    raise ValueError("Pinned official adapter file is missing or changed")
                previous_hash = digest(target)
                saved["generator"]["adapter"].update(revision=ADAPTER_REVISION,
                    weight_filename=ADAPTER.name, file_sha256=ADAPTER_SHA256)
                saved["adapter_bound_at_utc"] = utc()
                saved["pre_adapter_protocol_sha256"] = previous_hash
                saved["preparation_script_sha256"] = digest(Path(__file__))
                write_json(target, saved)
            if digest(ADAPTER) != saved["generator"]["adapter"]["file_sha256"]:
                raise ValueError("Pinned adapter fingerprint changed")
            lock = {"locked_at_utc": utc(), "adapter_model": "latent-consistency/lcm-lora-sdv1-5",
                "adapter_revision": ADAPTER_REVISION, "adapter_filename": ADAPTER.name,
                "adapter_sha256": ADAPTER_SHA256, "base_revision": CACHE.name,
                "base_checkpoint_sha256": saved["generator"]["checkpoint_sha256"],
                "protocol_sha256": digest(target), "head_sha256": heads,
                "calibration_sha256": digest(CAL), "generation_performed": False}
            lock_path = OUT / "generation_assets_lock.json"
            if lock_path.exists():
                previous_lock = json.loads(lock_path.read_text(encoding="utf-8"))
                for key in ["adapter_revision", "adapter_sha256", "base_checkpoint_sha256", "protocol_sha256", "head_sha256", "calibration_sha256"]:
                    if previous_lock[key] != lock[key]:
                        raise ValueError("Frozen generation asset lock changed")
            else:
                write_json(lock_path, lock)
            return saved
        if (OUT / "source_pairs.csv").exists() or (OUT / "generation_complete.json").exists():
            raise ValueError("Cannot supersede the DDIM draft after pair selection or generation")
        superseded = OUT / "protocol_ddim50_superseded.json"
        if superseded.exists():
            raise FileExistsError("Superseded draft already exists")
        old_hash = digest(target)
        target.replace(superseded)
        write_json(OUT / "protocol_amendment.json", {
            "amended_at_utc": utc(), "superseded_protocol_sha256": old_hash,
            "reason": "Before source selection or generation, replace the 50-step guided DDIM draft with the official 8-step LCM-LoRA accelerated SD1.5-family pipeline to fit available compute.",
            "source_selection_performed": False, "generation_performed": False,
            "detector_scores_observed_for_new_pairs": False,
            "parameter_selection_based_on_detector_results": False,
        })
    previous = json.loads((ROOT / "outputs/phase3/regeneration/protocol.json").read_text(encoding="utf-8"))
    checkpoint = {name.replace("\\", "/"): expected for name, expected in previous["checkpoint_sha256"].items()}
    for name, expected in checkpoint.items():
        if digest(CACHE / name) != expected:
            raise ValueError("Cached generator checkpoint fingerprint changed")
    saved = {
        "frozen_at_utc": utc(), "stage": "protocol_frozen_before_source_selection_or_generation",
        "seed": SEED, "planned_pairs": COUNT, "planned_sources": 2 * COUNT,
        "planned_image_copies": 6 * COUNT,
        "primary_endpoint": "Identity unmarked origin ROC-AUC for each previously frozen probe; fixed validation-threshold TPR and actual FPR reported separately.",
        "secondary_endpoints": "JPEG50 and 32x32 round-trip AUC/TPR/FPR, and ordinary fusion/gate comparisons using the unchanged expected-payload decoder.",
        "source_split": "All 200 pairs are prospective test-only sources; no pair member or derivative enters fitting/calibration. Existing detector source splits and heads remain unchanged.",
        "selection": {"collection": "COCO val2017", "minimum_side": 256,
            "order": "SHA256('paired-content|20261010|coco|'+12-digit public photo ID)",
            "excluded_prior_records": 7592,
            "exclude": "All previously used native COCO public IDs, any previous native or standardized exact pixel fingerprint, 8x8 aHash Hamming distance<=2 against all prior records and selected new photographs.",
            "detector_score_selection": False, "within_sample_screen": True},
        "caption_rule": {"asset": ASSET.name, "member": CAPTION_MEMBER,
            "annotation": "Lowest integer caption annotation ID for each selected COCO photo; exact caption text with fixed 'Photograph: ' prefix, no manual editing or quality selection.",
            "caption_source": "Official COCO 2017 train/val annotation archive"},
        "generator": {"model": "SG161222/Realistic_Vision_V5.1_noVAE", "cached_revision": CACHE.name,
            "pipeline": "StableDiffusionPipeline text-to-image; no photograph supplied as initialization",
            "size": [512, 512], "scheduler": "LCMScheduler.from_config(cached scheduler config)",
            "num_inference_steps": 8, "guidance_scale": 1.0,
            "adapter": {"model": "latent-consistency/lcm-lora-sdv1-5",
                "weight": 1.0, "application": "Load official SD1.5 LCM-LoRA and fuse exactly once; no weight tuning",
                "commit_and_file_sha256": "Bound in generation_assets_lock.json before source selection/generation; never resolved from test scores",
                "primary_model_card": "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5"},
            "negative_prompt": "", "batch_size": 1,
            "seed": "int(SHA256('paired-content|20261010|coco|'+12-digit public photo ID)[:16],16) modulo 2**31; CPU torch.Generator",
            "unet_text_dtype": "float16", "vae_dtype": "float32",
            "attention": "Default torch2 SDPA; no simultaneous attention slicing",
            "memory": "Resident CUDA first; on OOM use model CPU offload without changing prompt/seed/model/parameters",
            "resampling_or_output_quality_selection": False, "checkpoint_sha256": checkpoint},
        "common_processing": "Both origins convert to RGB/BGR, OpenCV INTER_AREA resize to256square, identical OpenCV JPEG95 encode/decode, lossless PNG staging; same three unmarked transformations afterward.",
        "conditions": [{"name": "identity", "severity": 0}, {"name": "jpeg", "severity": 2}, {"name": "lowres", "severity": 2}],
        "mark_embedding": False, "expected_payload": "1011010010110100", "gate_threshold": 13 / 16,
        "bootstrap": {"replicates": 2000, "seed": 20261012,
            "cluster": "Caption/photo pair; both real and generated source plus all derivatives stay together in every paired draw.",
            "interval": "Percentile95%; conditional on frozen heads/thresholds and the selected generator."},
        "test_inference_passes": 1, "test_refitting": False, "test_threshold_retuning": False,
        "input_sha256": inputs, "head_sha256": heads,
        "calibration_caveat": "Existing fusion heads reused calibration validation rows; fusion/gate calibration is retrospective. The paired test remains prospectively selected after head/threshold lock.",
        "limitations": ["Caption pairing aligns described scene content but does not create identical scenes or pixel correspondence.",
            "One cached Realistic Vision plus LCM-LoRA SD1.5-family pipeline does not establish new-generator-family generalization.",
            "COCO or caption overlap in generator/detector pretraining is not ruled out.",
            "Common final JPEG encoding does not erase original photographs' compression histories.",
            "Memory placement and fixed seeds do not promise bitwise equality across different hardware/library versions."],
        "preparation_script_sha256": digest(Path(__file__)),
    }
    write_json(target, saved)
    print("Frozen paired-content protocol; no generation/inference performed", flush=True)
    return saved


def select():
    protocol = freeze()
    target = OUT / "source_pairs.csv"
    if target.exists():
        complete = json.loads((OUT / "selection_complete.json").read_text(encoding="utf-8"))
        if complete["source_pairs_sha256"] != digest(target) or complete["protocol_sha256"] != digest(OUT / "protocol.json"):
            raise ValueError("Locked pair selection changed")
        print("Reused locked 200-pair selection", flush=True)
        return
    if not ASSET.is_file():
        raise FileNotFoundError("Official COCO annotation archive is required before selection")
    with zipfile.ZipFile(ASSET) as captions_archive:
        captions_bytes = captions_archive.read(CAPTION_MEMBER)
    captions_data = json.loads(captions_bytes)
    captions = {}
    for annotation in captions_data["annotations"]:
        iid = str(int(annotation["image_id"])).zfill(12)
        if iid not in captions or int(annotation["id"]) < int(captions[iid]["id"]):
            captions[iid] = annotation
    paths = frozen_input_paths()
    index = read_csv(paths["public_index"])
    index_audit = json.loads(paths["public_index_audit"].read_text(encoding="utf-8"))
    if not index_audit["all_checks_passed"] or len(index) != 7592:
        raise ValueError("Completed public source index is required")
    by_id = {row["source_id"]: row for row in index}
    prior_hash_by_id = {}
    for scope, hash_field in (("original_source_audit", "ahash"), ("openimages_sources", "ahash"), ("fresh192", "prejpeg_ahash"), ("fresh400", "prejpeg_ahash")):
        for row in read_csv(paths[scope]):
            prior_hash_by_id[row["source_id"]] = str(int(row[hash_field]))
    if set(prior_hash_by_id) != set(by_id):
        raise ValueError("Prior aHash source inventory differs from public index")
    exclusion_rows = [{"source_id": row["source_id"], "collection": row["collection"],
                       "public_image_id": row["public_image_id"],
                       "raw_pixel_sha256": row["raw_pixel_sha256"],
                       "standardized_pixel_sha256": row["standardized_pixel_sha256"],
                       "ahash": prior_hash_by_id[row["source_id"]]} for row in index]
    exclusion_rows.sort(key=lambda row: row["source_id"])
    write_csv(OUT / "exclusion_manifest.csv", exclusion_rows)
    excluded_ids = {row["public_image_id"] for row in index if row["collection"] == "coco"}
    exact = {row[field] for row in index for field in ("raw_pixel_sha256", "standardized_pixel_sha256") if row[field]}
    hashes = [int(value) for value in prior_hash_by_id.values()]
    chosen, rejection = [], []
    with zipfile.ZipFile(RAW / "val2017.zip") as archive:
        candidates = [(Path(name).stem, name) for name in archive.namelist() if name.lower().endswith(".jpg")]
        for iid, entry in sorted(candidates, key=lambda value: order(value[0])):
            if len(chosen) == COUNT:
                break
            reason = ""
            if iid in excluded_ids:
                reason = "previously_used_native_id"
            elif iid not in captions:
                reason = "caption_unavailable"
            else:
                payload = archive.read(entry)
                im = Image.open(io.BytesIO(payload)).convert("RGB")
                if min(im.size) < 256:
                    reason = "short_side_below256"
                else:
                    raw = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                    raw_hash = pixel_digest(raw)
                    base = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
                    average_hash = ahash(base)
                    ok, jpeg = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise ValueError("Common JPEG95 encoding failed")
                    standardized = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                    standardized_hash = pixel_digest(standardized)
                    if raw_hash in exact or standardized_hash in exact or any((average_hash ^ previous).bit_count() <= 2 for previous in hashes):
                        reason = "prior_or_within_new_sample_hash_overlap"
                    else:
                        pair_id = order(iid)[:24]
                        real_id = hashlib.sha256((pair_id + "|real").encode()).hexdigest()[:24]
                        ai_id = hashlib.sha256((pair_id + "|generated").encode()).hexdigest()[:24]
                        path = DATA / "real_base" / f"{real_id}.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(path), standardized):
                            raise ValueError("Standardized real PNG write failed")
                        caption = captions[iid]
                        chosen.append({"pair_id": pair_id, "real_source_id": real_id, "ai_source_id": ai_id,
                            "public_coco_image_id": iid, "archive_filename": "val2017.zip", "archive_entry": entry,
                            "caption_annotation_id": str(int(caption["id"])), "caption": caption["caption"],
                            "prompt": "Photograph: " + caption["caption"],
                            "generation_seed": int(order(iid)[:16], 16) % (2 ** 31),
                            "split": "test", "real_path": str(path), "native_file_sha256": hashlib.sha256(payload).hexdigest(),
                            "raw_pixel_sha256": raw_hash, "prejpeg_ahash": str(average_hash),
                            "standardized_pixel_sha256": standardized_hash, "standardized_file_sha256": digest(path)})
                        hashes.append(average_hash)
                        exact.update((raw_hash, standardized_hash))
            if reason:
                rejection.append({"public_coco_image_id": iid, "reason": reason})
    if len(chosen) != COUNT or len({row["pair_id"] for row in chosen}) != COUNT:
        raise ValueError("Insufficient nonoverlapping unused caption pairs")
    write_csv(target, chosen)
    write_csv(OUT / "selection_rejections.csv", rejection)
    write_json(OUT / "selection_complete.json", {
        "completed_at_utc": utc(), "pairs": COUNT, "planned_origin_sources": 400,
        "planned_unmarked_image_copies": 1200, "prior_source_records": len(index),
        "prior_coco_public_ids": len(excluded_ids), "prior_native_id_intersections": 0,
        "exact_or_ahash_overlap_screen_passed": True,
        "source_pairs_sha256": digest(target), "protocol_sha256": digest(OUT / "protocol.json"),
        "exclusion_manifest_sha256": digest(OUT / "exclusion_manifest.csv"),
        "caption_archive_sha256": digest(ASSET),
        "captions_val2017_json_sha256": hashlib.sha256(captions_bytes).hexdigest(),
        "coco_image_archive_sha256": digest(RAW / "val2017.zip"),
        "rejection_counts": dict(Counter(row["reason"] for row in rejection)),
        "all_pairs_split": "test", "detector_inference_performed": False,
        "generation_performed": False, "selection_script_sha256": digest(Path(__file__)),
        "caption_rule": protocol["caption_rule"],
    })
    print("Locked 200 unused caption pairs; generation/inference not performed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("protocol", "select", "generate", "extract", "analyze"), required=True)
    args = parser.parse_args()
    if args.stage == "protocol":
        freeze()
    elif args.stage == "select":
        select()
    else:
        raise RuntimeError("GPU/scoring stages are not implemented by this preparation driver; root schedules the frozen generation implementation after selection is locked")


if __name__ == "__main__":
    main()
