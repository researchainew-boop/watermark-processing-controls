"""Prospective official UFD native/common identity baseline for Phase6.

Plan and independent photographic calibration precede test inference. The
released detector is unchanged. No sign flipping or generator-based tuning is
allowed. Hash-identified blocks support recovery without another inference pass.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import native_baseline as official

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6/native_baseline"
SOURCES = ROOT / "outputs/phase6/data_inventory/primary_source_manifest.csv"
LOCK = ROOT / "outputs/phase6/frozen_protocol.json"
MODES = ("native", "common")
ALPHA = .05
BOOTSTRAP = 2000
SEED = 20261019
ASSET_HASHES = {".tools/extension-assets/ViT-L-14.pt": official.EXPECTED_BACKBONE,
    ".tools/extension-assets/univfd_fc_weights.pth": official.EXPECTED_HEAD,
    ".tools/extension-assets/official_clip_model.py": official.EXPECTED_CODE}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return official.digest(path)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "plan.json"
    obj = {"planned_at_utc": utc(), "detector_scores_observed": False,
        "detector": "Released UniversalFakeDetect CLIP ViT-L/14 linear head, unchanged official weights",
        "source_manifest": SOURCES.relative_to(ROOT).as_posix(),
        "required_source_counts": {"calibration": {"coco": 400, "openimages": 400},
            "test": {"coco": 200, "openimages": 200, "BigGAN": 200, "ADM": 200, "VQDM": 200}},
        "representations": {"native": "Original decoded pixels, metadata rotation applied by inventory, lossless storage, no resize or re-encoding",
            "common": "Exact frozen common128->256 JPEG95 base image, as fixed in root source protocol"},
        "official_transform": "CenterCrop224; ToTensor; CLIP mean(.48145466,.4578275,.40821073) std(.26862954,.26130258,.27577711); no Resize",
        "native_small_image_behavior": "Official CenterCrop zero padding when a dimension is below224; report source counts/padding by family",
        "feature_normalization": False, "score": "Released linear head logit, larger means synthetic; no polarity tuning",
        "threshold_calibration": {"negative_sources": 800, "alpha": ALPHA,
            "rank_one_based": "ceil((n+1)*(1-alpha))", "decision": "strict logit>cut", "independent_cut_per_representation": True,
            "pooling": "COCO400+OpenImages400 negatives; no synthetic calibration data", "test_tuning": False},
        "additional_original_cut": "logit>=0 retained as descriptive released-head diagnostic",
        "bootstrap": {"replicates": BOOTSTRAP, "seed": SEED,
            "unit": "independent source", "stratification": "collection", "native_common_paired": True},
        "model_asset_sha256": ASSET_HASHES, "implementation_sha256": digest(Path(__file__)),
        "previous_driver_sha256": digest(Path(official.__file__)), "fresh_test_inferred": False}
    if target.exists():
        saved = read(target)
        for key in ["implementation_sha256", "previous_driver_sha256", "model_asset_sha256", "threshold_calibration"]:
            if saved[key] != obj[key]:
                raise ValueError(f"Baseline plan changed: {key}")
        return saved
    write(target, obj)
    return obj


def validate_lock(role=None):
    if not LOCK.exists():
        raise RuntimeError("Root frozen_protocol.json is required before prospective calibration/test inference")
    lock = read(LOCK)
    for relative, expected in lock["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen root input changed: {relative}")
    if digest(SOURCES) != lock["source_manifest_sha256"]:
        raise ValueError("Fresh source manifest differs from root lock")
    saved_plan = plan()
    expected = lock["input_sha256"]
    for relative in ["scripts/phase6_native_baseline.py", "outputs/phase6/native_baseline/plan.json", *ASSET_HASHES]:
        if relative not in expected:
            raise ValueError(f"UFD input must be present in root lock: {relative}")
    for relative, checksum in saved_plan["model_asset_sha256"].items():
        if digest(ROOT / relative) != checksum:
            raise ValueError(f"Official UFD asset changed: {relative}")
    if role == "test":
        cuts = OUT / "thresholds_locked.json"
        main = ROOT / "outputs/phase6/fusion/thresholds_locked.json"
        if not cuts.exists() or not main.exists():
            raise RuntimeError("Both UFD and main fusion thresholds must be locked before fresh test scoring")
        saved = read(cuts)
        if saved["root_protocol_sha256"] != digest(LOCK):
            raise ValueError("UFD thresholds were calibrated under a different root protocol")
        if digest(OUT / "calibration_predictions.csv") != saved["calibration_predictions_sha256"]:
            raise ValueError("Frozen UFD calibration predictions changed")
    return digest(LOCK)


def prepare():
    lockhash = validate_lock()
    source = pd.read_csv(SOURCES, keep_default_na=False, dtype=str)
    if len(source) != 1800 or source.source_id.nunique() != 1800:
        raise ValueError("Expected1800 source-disjoint final baseline inputs")
    counts = {(r, c): len(g) for (r, c), g in source.groupby(["role", "collection"])}
    expected = {(r, c): n for r, groups in plan()["required_source_counts"].items() for c, n in groups.items()}
    if counts != expected:
        raise ValueError(f"Unexpected source quotas: {counts}")
    rows = []
    for r in source.sort_values("source_id").to_dict("records"):
        for mode, path, checksum in [("native", r["native_path"], r["file_sha256"]),
            ("common", r["base_path"], r["output_sha256"])]:
            if digest(path) != checksum:
                raise ValueError(f"Fresh baseline input hash mismatch: {r['source_id']} {mode}")
            with Image.open(path) as image:
                width, height = image.size
            if mode == "common" and (width, height) != (256, 256):
                raise ValueError("Common UFD identity view must be exact frozen256 image")
            rows.append({"image_id": hashlib.sha256(f"UFD|{r['source_id']}|{mode}".encode()).hexdigest()[:24],
                "source_id": r["source_id"], "role": r["role"], "collection": r["collection"], "label": int(r["label"]),
                "representation": mode, "path": path, "output_file_sha256": checksum, "width": width, "height": height,
                "official_crop_needs_padding": bool(width < 224 or height < 224),
                "official_crop_padding_fraction": 1-min(width, 224)*min(height, 224)/(224*224)})
    frame = pd.DataFrame(rows).sort_values(["role", "source_id", "representation"]).reset_index(drop=True)
    target = OUT / "manifest.csv"
    if target.exists():
        prior = pd.read_csv(target, keep_default_na=False)
        if prior.image_id.tolist() != frame.image_id.tolist():
            raise ValueError("UFD baseline selection changed")
        saved = read(OUT / "preparation.json")
        if digest(target) != saved["manifest_sha256"] or lockhash != saved["root_protocol_sha256"]:
            raise ValueError("UFD preparation hash changed")
        return
    frame.to_csv(target, index=False)
    write(OUT / "preparation.json", {"prepared_at_utc": utc(), "root_protocol_sha256": lockhash,
        "source_manifest_sha256": digest(SOURCES), "manifest_sha256": digest(target), "sources": 1800, "views": 3600,
        "native_padding": frame.loc[frame.representation.eq("native")].groupby(["role", "collection"]).agg(
            sources=("source_id", "size"), padded_sources=("official_crop_needs_padding", "sum"),
            mean_padding_fraction=("official_crop_padding_fraction", "mean")).reset_index().to_dict("records")})


def load_official(device):
    import torch
    from torchvision import transforms
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location("phase6_official_clip", ROOT / ".tools/extension-assets/official_clip_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.jit.load(str(ROOT / ".tools/extension-assets/ViT-L-14.pt"), map_location="cpu").state_dict()
    model = module.build_model(state).float().eval()
    visual = model.visual.to(device).eval()
    del model, state
    gc.collect()
    linear = torch.nn.Linear(768, 1)
    linear.load_state_dict(torch.load(ROOT / ".tools/extension-assets/univfd_fc_weights.pth", map_location="cpu", weights_only=True), strict=True)
    linear.to(device).eval()
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
    return visual, linear, transform


def infer(role, device, batch_size, block_size):
    import torch
    lockhash = validate_lock(role)
    prep = read(OUT / "preparation.json")
    manifest = OUT / "manifest.csv"
    if digest(manifest) != prep["manifest_sha256"] or lockhash != prep["root_protocol_sha256"]:
        raise ValueError("UFD input preparation identity mismatch")
    frame = pd.read_csv(manifest, keep_default_na=False)
    frame = frame.loc[frame.role.eq(role)].reset_index(drop=True)
    result = OUT / f"{role}_predictions.csv"
    if result.exists():
        done = read(OUT / f"{role}_complete.json")
        if digest(result) != done["predictions_sha256"] or lockhash != done["root_protocol_sha256"]:
            raise ValueError("Completed UFD predictions changed")
        print(f"Reused completed hash-locked {role} UFD predictions; no inference", flush=True)
        return
    for r in frame.to_dict("records"):
        if digest(r["path"]) != r["output_file_sha256"]:
            raise ValueError("Frozen UFD pixels changed")
    meta = {"started_at_utc": utc(), "role": role, "root_protocol_sha256": lockhash,
        "manifest_sha256": digest(manifest), "rows": len(frame), "device": device, "batch_size": batch_size,
        "block_size": block_size, "models_refitted": False, "polarity_retuned": False,
        "dtype": "float32", "plan_sha256": digest(OUT / "plan.json")}
    started = OUT / f"{role}_started.json"
    if started.exists():
        old = read(started)
        for key in ["root_protocol_sha256", "manifest_sha256", "rows", "block_size", "dtype", "plan_sha256"]:
            if old[key] != meta[key]:
                raise ValueError(f"UFD resume settings differ: {key}")
    else:
        write(started, meta)
    visual, linear, transform = load_official(device)
    timer = time.perf_counter()
    blocks, records = [], []
    for start in range(0, len(frame), block_size):
        part = frame.iloc[start:start+block_size]
        path = OUT / f"{role}_blocks/{start:06d}.npz"
        receipt = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not receipt.exists() or digest(path) != read(receipt)["block_sha256"]:
                raise ValueError("UFD saved block lacks matching hash receipt")
            with np.load(path, allow_pickle=False) as data:
                if str(data["manifest_sha256"]) != digest(manifest) or str(data["root_protocol_sha256"]) != lockhash or not np.array_equal(data["image_id"], part.image_id.to_numpy(str)):
                    raise ValueError("UFD saved block identity mismatch")
                values = data["logits"].copy()
        else:
            values = []
            with torch.inference_mode():
                for substart in range(0, len(part), batch_size):
                    current = part.iloc[substart:substart+batch_size]
                    tensors = []
                    for imagepath in current.path:
                        with Image.open(imagepath) as image:
                            tensors.append(transform(image.convert("RGB")))
                    batch = torch.stack(tensors).to(device)
                    values.extend(linear(visual(batch)).flatten().cpu().tolist())
            values = np.asarray(values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("UFD produced non-finite logits")
            # The block is accepted only once its hash receipt exists.
            np.savez_compressed(path, logits=values, image_id=part.image_id.to_numpy(str),
                manifest_sha256=digest(manifest), root_protocol_sha256=lockhash)
            write(receipt, {"block_sha256": digest(path), "rows": len(part), "root_protocol_sha256": lockhash})
        if len(values) != len(part):
            raise ValueError("UFD saved block has wrong length")
        block = part.copy()
        block["ufd_logit"] = values
        blocks.append(block)
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
        print(f"UFD {role} {start+len(part)}/{len(frame)} views, elapsed{time.perf_counter()-timer:.1f}s", flush=True)
    predictions = pd.concat(blocks, ignore_index=True)
    predictions.to_csv(result, index=False)
    write(OUT / f"{role}_complete.json", {"completed_at_utc": utc(), "role": role, "root_protocol_sha256": lockhash,
        "predictions_sha256": digest(result), "rows": len(predictions), "blocks": records,
        "elapsed_seconds": time.perf_counter()-timer, "dtype": "float32", "device": device,
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "pillow"]}})
    del visual, linear, transform
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def negative_cut(scores, alpha=ALPHA):
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration needs finite photographic scores")
    rank = int(math.ceil((len(scores)+1)*(1-alpha)))
    if rank > len(scores):
        raise ValueError("Calibration sample cannot resolve this target false-positive rate")
    return float(np.sort(scores)[rank-1]), rank


def calibrate():
    lockhash = validate_lock("calibration")
    path = OUT / "thresholds_locked.json"
    if (OUT / "test_started.json").exists() or (OUT / "test_predictions.csv").exists():
        if not path.exists():
            raise ValueError("Cannot choose UFD cuts after test inference")
    predictions = OUT / "calibration_predictions.csv"
    if digest(predictions) != read(OUT / "calibration_complete.json")["predictions_sha256"]:
        raise ValueError("UFD calibration predictions changed")
    if path.exists():
        old = read(path)
        if old["calibration_predictions_sha256"] != digest(predictions) or old["root_protocol_sha256"] != lockhash:
            raise ValueError("UFD threshold lock changed")
        print("Reused independently locked UFD thresholds", flush=True)
        return
    frame = pd.read_csv(predictions, keep_default_na=False)
    if len(frame) != 1600 or frame.label.ne(0).any() or frame.source_id.nunique() != 800:
        raise ValueError("UFD calibration must be800 photographic sources, two representations each")
    cuts = {}
    for mode in MODES:
        block = frame.loc[frame.representation.eq(mode)]
        if len(block) != 800 or block.source_id.nunique() != 800:
            raise ValueError("Incorrect UFD photographic calibration rows")
        cut, rank = negative_cut(block.ufd_logit)
        cuts[mode] = {"threshold_logit": cut, "rank_one_based": rank, "negative_sources": 800,
            "calibration_FPR": float((block.ufd_logit > cut).mean()),
            "per_collection_FPR": {c: float((g.ufd_logit > cut).mean()) for c, g in block.groupby("collection")}}
    write(path, {"locked_at_utc": utc(), "root_protocol_sha256": lockhash, "alpha": ALPHA,
        "decision": "strict logit>cut", "test_scores_observed": False, "calibration_predictions_sha256": digest(predictions),
        "cuts": cuts, "conditionality": "Marginal negative calibration under source exchangeability; no unconditional guarantee under collection or processing shift"})
    print(json.dumps(cuts), flush=True)


def paired_metrics(frame, cuts, replicates=BOOTSTRAP):
    """Paired source bootstrap, stratified by source collection."""
    y = frame.label.to_numpy(int)
    if len(np.unique(y)) != 2:
        raise ValueError("AUC needs both provenance labels")
    scores = frame[list(MODES)].to_numpy(np.float64)
    thresholds = np.array([cuts[m]["threshold_logit"] for m in MODES])
    predicted = scores > thresholds
    strata = [np.flatnonzero(frame.collection.to_numpy() == c) for c in sorted(frame.collection.unique())]
    estimates = np.empty((replicates, 2, 3), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    for b in range(replicates):
        ix = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        for j in range(2):
            estimates[b, j] = [roc_auc_score(y[ix], scores[ix, j]),
                predicted[ix[y[ix] == 1], j].mean(), predicted[ix[y[ix] == 0], j].mean()]
    point = {"sources": len(frame), "AI_sources": int((y == 1).sum()), "real_sources": int((y == 0).sum()),
        "metrics": {}, "paired_native_minus_common_AUC": float(roc_auc_score(y, scores[:, 0])-roc_auc_score(y, scores[:, 1])),
        "paired_native_minus_common_AUC_ci95": np.percentile(estimates[:, 0, 0]-estimates[:, 1, 0], [2.5, 97.5]).tolist()}
    for j, mode in enumerate(MODES):
        values = [roc_auc_score(y, scores[:, j]), predicted[y == 1, j].mean(), predicted[y == 0, j].mean()]
        point["metrics"][mode] = {"threshold_logit": float(thresholds[j]),
            **{name: float(value) for name, value in zip(["AUC", "TPR", "FPR"], values)},
            **{name+"_ci95": np.percentile(estimates[:, j, k], [2.5, 97.5]).tolist() for k, name in enumerate(["AUC", "TPR", "FPR"])},
            "released_threshold0_TPR": float((scores[y == 1, j] >= 0).mean()),
            "released_threshold0_FPR": float((scores[y == 0, j] >= 0).mean())}
    return point


def analyze():
    validate_lock("test")
    predictions = OUT / "test_predictions.csv"
    if digest(predictions) != read(OUT / "test_complete.json")["predictions_sha256"]:
        raise ValueError("UFD test predictions changed")
    rows = pd.read_csv(predictions, keep_default_na=False)
    if len(rows) != 2000 or rows.source_id.nunique() != 1000:
        raise ValueError("Expected1000 final source identities, two views each")
    pivot = rows.pivot(index="source_id", columns="representation", values="ufd_logit").reset_index()
    meta = rows[["source_id", "collection", "label"]].drop_duplicates()
    paired = meta.merge(pivot, on="source_id", validate="one_to_one")
    if not np.isfinite(paired[list(MODES)].to_numpy()).all():
        raise ValueError("UFD paired prediction table has missing scores")
    cuts = read(OUT / "thresholds_locked.json")["cuts"]
    metrics = {"completed_at_utc": utc(), "overall": paired_metrics(paired, cuts),
        "per_fresh_generator": {}, "per_real_collection": {}, "test_tuning": False,
        "polarity_changed": False, "bootstrap_replicates": BOOTSTRAP, "bootstrap_seed": SEED,
        "root_protocol_sha256": digest(LOCK), "test_predictions_sha256": digest(predictions),
        "native_padding": read(OUT / "preparation.json")["native_padding"],
        "interpretation": "Native and common preprocessing jointly differ in resolution, geometry and JPEG processing; native128 images invoke officialCenterCrop padding. Contrast is descriptive, not a causal attribution of detector failure."}
    for family in ["BigGAN", "ADM", "VQDM"]:
        metrics["per_fresh_generator"][family] = paired_metrics(paired.loc[paired.label.eq(0) | paired.collection.eq(family)], cuts)
    for collection in ["coco", "openimages"]:
        metrics["per_real_collection"][collection] = paired_metrics(paired.loc[paired.label.eq(1) | paired.collection.eq(collection)], cuts)
    paired.to_csv(OUT / "paired_test_predictions.csv", index=False)
    write(OUT / "metrics.json", metrics)
    print(json.dumps(metrics["overall"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["plan", "prepare", "calibration", "calibrate", "test", "analyze"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    if args.stage == "plan":
        print(json.dumps(plan(), indent=2))
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "calibrate":
        calibrate()
    elif args.stage == "analyze":
        analyze()
    else:
        infer(args.stage, args.device, args.batch_size, args.block_size)


if __name__ == "__main__":
    main()
