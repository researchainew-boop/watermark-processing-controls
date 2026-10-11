"""Evaluate the released UniversalFakeDetect head on native fresh-source pixels.

Prepare native inputs first, run one frozen inference pass, then compare to the
archived 256-pixel/JPEG95 results. No model fitting or score polarity changes.
The preprocessing contrast includes both resizing and JPEG re-encoding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase3/native_baseline"
DATA = ROOT / "data/phase3/native_baseline"
ASSETS = ROOT / ".tools/extension-assets"
EXPECTED_BACKBONE = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
EXPECTED_HEAD = "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847"
EXPECTED_CODE = "c071d011e92226f1ca0a6f7c5098d8d7d08eadc7d6db125a81b52fc234b1ec59"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "manifest.csv"
    if manifest.exists():
        print("Native inputs already frozen; no reselection.", flush=True)
        return
    fresh_path = ROOT / "outputs/extension/fresh/manifest.csv"
    fresh = pd.read_csv(fresh_path, keep_default_na=False)
    assert len(fresh) == fresh.source_id.nunique() == 192
    archives = {}
    rows = []
    try:
        for row in fresh.to_dict("records"):
            if row["collection"] == "openimages":
                raw = ROOT / "data/extension_fresh/raw_openimages" / (row["ImageID"] + ".jpg")
                payload = raw.read_bytes()
                origin = str(raw)
            else:
                archive = row["archive"]
                if archive not in archives:
                    archives[archive] = zipfile.ZipFile(archive)
                payload = archives[archive].read(row["original_id"])
                origin = archive + "::" + row["original_id"]
            assert hashlib.sha256(payload).hexdigest() == row["input_file_sha256"]
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            rotation = row.get("Rotation", "")
            if rotation != "" and not pd.isna(rotation) and float(rotation):
                image = image.rotate(-float(rotation), expand=True)
            pixels = np.asarray(image)[:, :, ::-1].copy()  # archived hash used BGR
            rawhash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            assert rawhash == row["raw_pixel_sha256"]
            assert min(image.size) >= 256
            target = DATA / (row["source_id"] + ".png")
            image.save(target, format="PNG")
            rows.append({"source_id": row["source_id"], "label": row["label"],
                         "collection": row["collection"], "generator": row["generator"],
                         "path": str(target), "raw_origin": origin,
                         "width": image.width, "height": image.height,
                         "input_file_sha256": row["input_file_sha256"],
                         "raw_pixel_sha256": rawhash, "output_file_sha256": digest(target)})
    finally:
        for z in archives.values():
            z.close()
    pd.DataFrame(rows).to_csv(manifest, index=False)
    protocol = {
        "frozen_at_utc": now(), "benchmark": "same 192 previously locked fresh sources",
        "source_selection_changed": False, "fresh_manifest_sha256": digest(fresh_path),
        "native_manifest_sha256": digest(manifest), "script_sha256": digest(__file__),
        "native_preparation": "Original archive/S3 pixels; same metadata rotation as fresh run; RGB conversion; lossless PNG; no resize or re-encoding",
        "official_transform": "CenterCrop(224), ToTensor, CLIP mean (.48145466,.4578275,.40821073), std (.26862954,.26130258,.27577711); no Resize",
        "feature_normalization": False, "backbone_sha256": EXPECTED_BACKBONE,
        "head_sha256": EXPECTED_HEAD, "official_model_code_sha256": EXPECTED_CODE,
        "classifier_threshold_logit": 0.0, "test_tuning": False, "score_polarity_changed": False,
        "comparison": "native pixels versus archived INTER_AREA 256x256 plus common JPEG95; resizing and JPEG both differ",
        "bootstrap": "2000 paired source draws stratified by collection, seed 20261009",
        "fresh_pipeline_rerun": False,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Prepared and hash verified native sources:", len(rows), flush=True)


def evaluate(device, batch_size):
    import gc
    import importlib.util
    import torch
    from torchvision import transforms

    torch.set_num_threads(4)
    result = OUT / "predictions.csv"
    if result.exists():
        raise RuntimeError("Native inference is frozen; refusing another pass")
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(OUT / "manifest.csv") == protocol["native_manifest_sha256"]
    for name, expected in [("ViT-L-14.pt", EXPECTED_BACKBONE),
                           ("univfd_fc_weights.pth", EXPECTED_HEAD),
                           ("official_clip_model.py", EXPECTED_CODE)]:
        assert digest(ASSETS / name) == expected
    rows = pd.read_csv(OUT / "manifest.csv", keep_default_na=False)
    for row in rows.to_dict("records"):
        assert digest(row["path"]) == row["output_file_sha256"]
    started = now()
    spec = importlib.util.spec_from_file_location("official_clip_model", ASSETS / "official_clip_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.jit.load(str(ASSETS / "ViT-L-14.pt"), map_location="cpu").state_dict()
    model = module.build_model(state).float().eval()
    visual = model.visual.to(device).eval()
    del model, state
    gc.collect()
    linear = torch.nn.Linear(768, 1)
    linear.load_state_dict(torch.load(ASSETS / "univfd_fc_weights.pth", map_location="cpu", weights_only=True), strict=True)
    linear.to(device).eval()
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
    logits = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for path in rows.path.iloc[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).to(device)
            logits.extend(linear(visual(batch)).flatten().cpu().tolist())
            if start % 32 == 0:
                print("Native published detector", start + len(batch), "/", len(rows), flush=True)
    rows["univfd_native_logit"] = logits
    rows.to_csv(result, index=False)
    (OUT / "inference.json").write_text(json.dumps({
        "started_at_utc": started, "completed_at_utc": now(), "rows": len(rows),
        "device": device, "batch_size": batch_size, "dtype": "float32",
        "models_refitted": False, "thresholds_retuned": False,
        "predictions_sha256": digest(result)}, indent=2), encoding="utf-8")
    print("Frozen native predictions saved.", flush=True)


def compare(frame):
    y = frame.label.to_numpy()
    native = frame.univfd_native_logit.to_numpy()
    common = frame.univfd_logit.to_numpy()
    strata = [np.flatnonzero(frame.collection.to_numpy() == g) for g in sorted(frame.collection.unique())]
    rng = np.random.default_rng(20261009)
    estimates = []
    for _ in range(2000):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        a = roc_auc_score(y[ix], native[ix]); b = roc_auc_score(y[ix], common[ix])
        estimates.append([a, b, a-b])
    boot = np.asarray(estimates)
    a = float(roc_auc_score(y, native)); b = float(roc_auc_score(y, common))
    point = {"native_auc": a, "common256_jpeg95_auc": b, "paired_auc_difference": a-b,
             "sources": len(frame), "real_sources": int((y==0).sum()), "ai_sources": int((y==1).sum())}
    for i, name in enumerate(["native_auc", "common256_jpeg95_auc", "paired_auc_difference"]):
        point[name + "_ci95"] = np.percentile(boot[:, i], [2.5, 97.5]).tolist()
    for name, scores in [("native", native), ("common256_jpeg95", common)]:
        point[name + "_threshold0"] = {"fpr": float((scores[y==0] >= 0).mean()),
                                        "tpr": float((scores[y==1] >= 0).mean())}
    return point


def analyze():
    native = pd.read_csv(OUT / "predictions.csv", keep_default_na=False)
    common = pd.read_csv(ROOT / "outputs/extension/published_baseline/predictions.csv", keep_default_na=False)
    common = common.loc[common.benchmark == "fresh", ["source_id", "label", "univfd_logit"]]
    assert len(native) == len(common) == 192
    merged = native.merge(common, on=["source_id", "label"], validate="one_to_one")
    assert len(merged) == 192
    results = {"overall": compare(merged), "per_real_collection": {},
               "no_test_tuning": True, "cause_of_failures_confirmed": False,
               "interpretation": "The paired preprocessing contrast jointly changes image size and JPEG re-encoding. No polarity reversal or selection based on the observed score is performed."}
    for real in ["coco", "openimages"]:
        results["per_real_collection"][real] = compare(merged.loc[(merged.label == 1) | (merged.collection == real)])
    merged.to_csv(OUT / "paired_predictions.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "evaluate", "analyze"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "evaluate": evaluate(args.device, args.batch_size)
    else: analyze()
