"""Prepare fixed matched-content and encoding controls without training.

This extension reuses retained sources; it is not a new independent dataset.
Selection uses source IDs and frozen clean CLIP features, never detector
scores. Raw source files and all existing experiment outputs remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, JpegImagePlugin
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = ROOT / "configs" / "extension_encoding.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def pixel_digest(pixels: np.ndarray) -> str:
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def ordered_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(f"Requested {n} sources from a pool of {len(frame)}")
    order = frame["source_id"].map(
        lambda identity: hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()
    )
    return frame.assign(selection_order=order).sort_values(
        ["selection_order", "source_id"]
    ).head(n).drop(columns="selection_order")


def metadata(row: dict) -> dict:
    path = Path(row["path"])
    with Image.open(path) as im:
        tables = getattr(im, "quantization", {}) or {}
        info = {
            "native_width": im.width, "native_height": im.height,
            "native_short_side": min(im.size), "raw_format": im.format,
            "raw_mode": im.mode, "raw_bytes": path.stat().st_size,
            "jpeg_subsampling": JpegImagePlugin.get_sampling(im) if im.format == "JPEG" else None,
            "jpeg_quantization": json.dumps(tables, sort_keys=True),
            "has_icc_profile": bool(im.info.get("icc_profile")),
            "has_exif": bool(im.info.get("exif")),
        }
    return {**row, **info}


def clean_clip_vectors(identities: set[str]) -> dict[str, np.ndarray]:
    result = {}
    paths = [ROOT / "outputs" / "clip_train_embeddings.npz",
             ROOT / "outputs" / "clip_eval_embeddings.npz"]
    for path in paths:
        with np.load(path, allow_pickle=True) as values:
            ids = values["image_id"].astype(str)
            selected = np.flatnonzero(np.isin(ids, list(identities)))
            vectors = values["embedding"][selected].astype(np.float64)
            for identity, vector in zip(ids[selected], vectors):
                if identity not in result:
                    norm = float(np.linalg.norm(vector))
                    if norm <= 0 or not np.isfinite(vector).all():
                        raise ValueError(f"Invalid clean CLIP feature: {identity}")
                    result[identity] = vector / norm
    missing = identities - result.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} clean features: {sorted(missing)[:3]}")
    return result


def cosine_matrix(left: pd.DataFrame, right: pd.DataFrame,
                  vectors: dict[str, np.ndarray]) -> np.ndarray:
    a = np.stack([vectors[i] for i in left["source_id"]])
    b = np.stack([vectors[i] for i in right["source_id"]])
    return np.clip(a @ b.T, -1.0, 1.0)


def caption_inventory(raw_root: Path) -> list[str]:
    # Read-only local discovery; captions are not downloaded or invented.
    roots = [ROOT / "data", raw_root]
    result = []
    for root in roots:
        for pattern in ("*caption*.json", "*annotation*.zip"):
            result.extend(str(path) for path in root.rglob(pattern) if path.is_file())
    return sorted(set(result))


def prepare(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources_path = ROOT / config["retained_sources"]
    sources = pd.read_csv(sources_path, keep_default_na=False,
                          dtype={"source_id": str, "image_id": str})
    if sources["source_id"].duplicated().any():
        raise ValueError("Expected exactly one retained source row per ID")
    seed = int(config["seed"])
    side = int(config["image_side"])
    qualities = [int(q) for q in config["jpeg_qualities"]]
    if side != 256 or qualities != [95, 75, 50]:
        raise ValueError("This fixed extension config uses 256 pixels and JPEG 95/75/50")
    data_dir = ROOT / config["data_directory"]
    output = ROOT / config["output_directory"]
    if (output / "selection.json").exists():
        raise FileExistsError("Selection already frozen; use a new configured directory for a new run")
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    real_candidates = sources.loc[(sources["label"] == 0) & sources["split"].isin(["val", "test"])]
    chosen_synthetic = pd.concat([
        ordered_sample(sources.loc[(sources["split"] == split) & (sources["generator"] == gen)],
                       int(n), seed)
        for split, counts in config["selection"].items()
        for gen, n in counts.items()
    ], ignore_index=True)
    if not (chosen_synthetic["label"] == 1).all():
        raise ValueError("Synthetic candidates must have origin label one")
    if (chosen_synthetic.loc[chosen_synthetic["generator"] == "midjourney", "split"] != "test").any():
        raise ValueError("Midjourney cannot enter validation")

    # A train-only diagnostic threshold is fixed before applying the rule to test.
    mcfg = config["matching"]
    train_real = ordered_sample(sources.loc[(sources["split"] == "train") & (sources["label"] == 0)],
                                int(mcfg["calibration_real"]), seed)
    train_synthetic = pd.concat([
        ordered_sample(sources.loc[(sources["split"] == "train") & (sources["generator"] == gen)],
                       int(mcfg["calibration_per_seen_generator"]), seed)
        for gen in config["selection"]["val"]
    ], ignore_index=True)
    calibration = pd.concat([train_real, train_synthetic])
    identities = set(real_candidates["source_id"]) | set(chosen_synthetic["source_id"]) | set(calibration["source_id"])
    vectors = clean_clip_vectors(identities)
    train_sim = cosine_matrix(train_synthetic, train_real, vectors).max(axis=1)
    caliper = float(np.percentile(train_sim, int(mcfg["calibration_percentile"])))

    pairs = []
    for split in config["selection"]:
        synthetic = chosen_synthetic.loc[chosen_synthetic["split"] == split].sort_values("source_id").reset_index(drop=True)
        real = real_candidates.loc[real_candidates["split"] == split].sort_values("source_id").reset_index(drop=True)
        similarity = cosine_matrix(synthetic, real, vectors)
        si, ri = linear_sum_assignment(-similarity)
        if len(si) != len(synthetic):
            raise ValueError(f"Not enough distinct real partners in {split}")
        for syn_idx, real_idx in zip(si, ri):
            syn = synthetic.iloc[syn_idx]
            photograph = real.iloc[real_idx]
            pair_id = hashlib.sha256(f"{syn.source_id}|{photograph.source_id}".encode()).hexdigest()[:16]
            score = float(similarity[syn_idx, real_idx])
            pairs.append({
                "pair_id": pair_id, "split": split, "generator": syn.generator,
                "synthetic_source_id": syn.source_id, "real_source_id": photograph.source_id,
                "clean_clip_cosine": score, "train_caliper": caliper,
                "below_train_caliper": score < caliper,
            })
    pair_frame = pd.DataFrame(pairs)
    if pair_frame["real_source_id"].duplicated().any() or pair_frame["synthetic_source_id"].duplicated().any():
        raise ValueError("Matching must be one-to-one across retained source IDs")
    selected_ids = set(pair_frame["synthetic_source_id"]) | set(pair_frame["real_source_id"])
    selected = sources.loc[sources["source_id"].isin(selected_ids)].copy()
    if (selected["split"] == "train").any():
        raise ValueError("Extension image preparation is limited to original validation and test")
    pair_map = {identity: (pair["pair_id"], pair["generator"])
                for pair in pairs for identity in [pair["synthetic_source_id"], pair["real_source_id"]]}
    selected["pair_id"] = selected["source_id"].map(lambda identity: pair_map[identity][0])
    selected["paired_generator"] = selected["source_id"].map(lambda identity: pair_map[identity][1])
    pair_frame.to_csv(output / "content_pairs.csv", index=False)
    selected.to_csv(output / "selected_sources.csv", index=False)
    calibration.to_csv(output / "matching_calibration_sources.csv", index=False)
    policy = {
        "seed": seed, "source_count": len(selected), "pair_count": len(pairs),
        "source_counts": {f"{split}|{gen}": int(n) for (split, gen), n in selected.groupby(["split", "generator"]).size().items()},
        "matching": "Fixed ID-hash synthetic sample; maximum total clean CLIP cosine one-to-one real assignment within original split",
        "calibration_split": "train", "diagnostic_caliper": caliper,
        "caliper_exclusions": 0, "below_caliper_pairs": int(pair_frame["below_train_caliper"].sum()),
        "selection_uses_detector_scores": False,
        "original_test_results_previously_observed": True,
        "new_control_scores_not_yet_computed": True,
        "conditions": ["common_png", *[f"jpeg_q{q}" for q in qualities]],
        "caution": "Approximate CLIP semantic matching can include origin/style cues and does not prove same-content matching. Existing collections and prior encoding remain confounds.",
        "config_sha256": digest(config_path), "retained_manifest_sha256": digest(sources_path),
    }
    # Freeze the complete selection before deriving any control images or new features.
    write_json(output / "selection.json", policy)

    inventory = pd.DataFrame([metadata(row) for row in
                              sources.loc[sources["split"].isin(["val", "test"])].to_dict("records")])
    inventory.to_csv(output / "native_inventory.csv", index=False)
    rows, lineage = [], []
    for position, row in enumerate(selected.sort_values(["split", "source_id"]).to_dict("records")):
        path = Path(row["path"])
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Cannot decode {path}")
        base = cv2.resize(raw, (side, side), interpolation=cv2.INTER_AREA)
        previous = cv2.imread(str(ROOT / "data" / "processed" / "base" / f"{row['source_id']}.png"))
        if previous is None:
            raise ValueError("Missing original standardized base")
        rec = metadata(row)
        rec.update({
            "raw_file_sha256": digest(path),
            "raw_decoded_pixels_sha256": pixel_digest(raw),
            "common_pixels_sha256": pixel_digest(base),
            "previous_base_pixels_sha256": pixel_digest(previous),
            "common_equals_previous_base": bool(np.array_equal(base, previous)),
            "input_stage": "selected stored JPEG; synthetic collection had already applied JPEG quality 95",
            "generator_native_original_recovered": False,
        })
        lineage.append(rec)
        conditions = [("common_png", None), *[(f"jpeg_q{q}", q) for q in qualities]]
        for condition, quality in conditions:
            pixels = base
            if quality is not None:
                ok, encoded = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise ValueError("JPEG round-trip failed")
                pixels = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            target_dir = data_dir / condition
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{row['source_id']}.png"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
            if not cv2.imwrite(str(target), pixels):
                raise ValueError(f"Cannot save {target}")
            image_id = hashlib.sha256(f"encoding_control|{row['source_id']}|{condition}".encode()).hexdigest()[:16]
            rows.append({
                "image_id": image_id, "source_id": row["source_id"], "parent_id": row["source_id"],
                "path": str(target), "label": int(row["label"]), "generator": row["generator"],
                "split": row["split"], "group": ("R-C" if quality is None else "R-T") if int(row["label"]) == 0 else "S-U",
                "pair_id": row["pair_id"], "paired_generator": row["paired_generator"],
                "condition": condition, "attack": "encoding_control",
                "severity": quality or 0, "jpeg_quality": quality,
                "watermark_method": "", "payload": "", "working_width": side, "working_height": side,
                "output_file_sha256": digest(target), "output_pixels_sha256": pixel_digest(pixels),
            })
        if position % 100 == 0:
            print(f"Prepared {position + 1}/{len(selected)} selected sources", flush=True)
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    pd.DataFrame(lineage).to_csv(output / "source_lineage.csv", index=False)
    dimensions = {f"{split}|{gen}": {
        "sources": len(sub), "width_range": [int(sub.native_width.min()), int(sub.native_width.max())],
        "height_range": [int(sub.native_height.min()), int(sub.native_height.max())],
        "formats": dict(Counter(sub["raw_format"])),
    } for (split, gen), sub in inventory.groupby(["split", "generator"])}
    raw_root = Path(sources.loc[sources["label"] == 0, "path"].iloc[0]).parent.parent
    summary = {
        **policy, "prepared_images": len(rows), "native_inventory_sources": len(inventory),
        "native_dimensions": dimensions,
        "common_base_mismatches": sum(not row["common_equals_previous_base"] for row in lineage),
        "local_coco_caption_files": caption_inventory(raw_root),
        "training_or_new_feature_inference_performed": False,
        "conditions_are_same_stored_source_with_additional_common_encoding": True,
        "prior_compression_erased": False,
    }
    write_json(output / "summary.json", summary)
    inputs = [sources_path, config_path, Path(__file__),
              ROOT / "outputs" / "clip_train_embeddings.npz", ROOT / "outputs" / "clip_eval_embeddings.npz"]
    provenance = {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "opencv-python-headless", "scipy", "Pillow", "PyYAML"]},
        "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "source_file_hashes": "source_lineage.csv", "derived_file_hashes": "manifest.csv",
        "selection_before_new_encoding_feature_measurements": True,
        "no_generator_native_recovery": True, "no_prospective_independent_collection": True,
    }
    write_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        prepare(args.config.resolve())


if __name__ == "__main__":
    main()
