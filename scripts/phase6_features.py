"""Received-image evidence and frozen-probe inference for Phase 6.

No original/sibling image is used to compute any feature. Metadata only joins
paired training observations and audits source-disjoint roles. Final source
data cannot be scored before the model/protocol lock is present.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import benchmark_watermarks as marks
from aiwstudy.watermark import DwtDctWatermark
from evaluate_encoding_controls import checkpoint_defaults, load_backbone, infer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6"
FAIR = ROOT / "outputs/phase5/fair_watermarks"
PAYLOADS = tuple(marks.PAYLOADS)
METHODS = ("dwt_dct", "trustmark")
SEED = 20261011
VIEWS = ("received", "jpeg90")
FEATURES = ("agreement", "min_view_agreement", "signed_margin", "q10_margin",
            "mean_abs_margin", "stability", "margin_stability", "full_match",
            "full_match_views", "ecc_valid", "codeword_agreement")
cv2.setNumThreads(1)


def digest(path):
    return marks.digest(Path(path))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def utc():
    return datetime.now(timezone.utc).isoformat()


def received_views(pixels):
    """A deployment-available transformation; never reads a clean reference."""
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG90 view failed")
    return [pixels, cv2.imdecode(encoded, cv2.IMREAD_COLOR)]


def prepare_development():
    target = OUT / "development"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "preparation.json").exists():
        saved = json.loads((target / "preparation.json").read_text())
        assert digest(target / "manifest.csv") == saved["manifest_sha256"]
        return
    sources = pd.read_csv(FAIR / "sources.csv", keep_default_na=False, dtype=str)
    # Midjourney and the formerly fresh RV images are never method-fit sources.
    sources = sources.loc[sources.collection.isin(["coco", "openimages", "stable-diffusion", "dalle-3"])].copy()
    roles = {}
    for collection, block in sources.groupby("collection", sort=True):
        ids = sorted(block.source_id, key=lambda s: hashlib.sha256(f"{SEED}|dev|{collection}|{s}".encode()).hexdigest())
        n = len(ids)
        fit = n // 2
        tune = n // 4
        for i, sid in enumerate(ids):
            roles[sid] = "fit" if i < fit else "tune" if i < fit + tune else "constraint"
    sources["role"] = sources.source_id.map(roles)
    if len(sources) != 936 or sources.source_id.nunique() != 936:
        raise ValueError(f"Unexpected development pools: {sources.groupby('collection').size().to_dict()}")
    manifest = pd.read_csv(FAIR / "manifest.csv", keep_default_na=False, dtype={"payload": str})
    manifest = manifest.loc[manifest.source_id.isin(roles)].copy()
    manifest["role"] = manifest.source_id.map(roles)
    manifest = manifest.loc[manifest.role.eq("constraint") | manifest.condition.isin(["identity", "jpeg50"])].copy()
    manifest = manifest.rename(columns={"payload": "embedded_payload"})
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    sources.to_csv(target / "sources.csv", index=False)
    manifest.to_csv(target / "manifest.csv", index=False)
    write(target / "preparation.json", {"created_at_utc": utc(), "design_is_retrospective": True,
        "existing_test_sources_are_development_only": True,
        "source_split": "SHA256 seeded order, stratified by collection;50%fit25%tune25%constraint",
        "fit_tune_conditions": ["identity", "jpeg50"],
        "constraint_conditions": sorted(manifest.loc[manifest.role.eq('constraint')].condition.unique()),
        "registered_payloads": PAYLOADS, "views": VIEWS, "feature_names": FEATURES,
        "source_counts": [{"role": r, "collection": c, "n": int(n)} for (r, c), n in sources.groupby(["role", "collection"]).size().items()],
        "manifest_sha256": digest(target / "manifest.csv"), "sources_sha256": digest(target / "sources.csv"),
        "input_sha256": {"phase5_sources": digest(FAIR / "sources.csv"), "phase5_manifest": digest(FAIR / "manifest.csv")},
        "probe_heads": {b: digest(ROOT / f"outputs/conference_clean/{b}_full.npz") for b in ["resnet", "clip"]},
        "initial_feature_driver_sha256": digest(__file__)})
    print("Prepared development", len(sources), "sources", len(manifest), "received images", flush=True)


def check_final_lock(kind):
    if kind == "development":
        return None
    path = OUT / "frozen_protocol.json"
    if not path.exists():
        raise RuntimeError("Freeze method/protocol before fresh calibration or test inference")
    lock = json.loads(path.read_text())
    for name, checksum in lock["input_sha256"].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError(f"Frozen input changed: {name}")
    learner = ROOT / lock.get("learner_output", "outputs/phase6/fusion")
    if kind == "test" and not (learner / "thresholds_locked.json").exists():
        raise RuntimeError("Independent thresholds must be locked before fresh test scoring")
    dependency_name = "outputs/phase6/passive_dependencies_locked.json"
    if dependency_name not in lock["input_sha256"]:
        raise RuntimeError("Passive backbone dependencies must be sealed before fresh inference")
    dependencies = json.loads((ROOT / dependency_name).read_text())
    for backbone, checkpoint in zip(("resnet", "clip"), checkpoint_defaults()):
        if digest(checkpoint) != dependencies["backbone_sha256"][backbone]:
            raise RuntimeError(f"Frozen passive backbone changed: {backbone}")
    if kind == "test":
        receipt_path = learner / "calibration_complete.json"
        if not receipt_path.exists():
            raise RuntimeError("Independent calibration completion receipt is missing")
        receipt = json.loads(receipt_path.read_text())
        for name, checksum in receipt["outputs_sha256"].items():
            if digest(learner / name) != checksum:
                raise RuntimeError(f"Calibrated threshold artifact changed: {name}")
        if receipt.get("fit_complete_sha256") != digest(learner / "fit_complete.json"):
            raise RuntimeError("Calibration refers to different fitted models")
        source_receipt = json.loads((OUT / "test_sources_locked.json").read_text())
        if source_receipt["protocol_sha256"] != digest(path) or source_receipt["source_manifest_sha256"] != digest(OUT / "data_inventory/primary_source_manifest.csv"):
            raise RuntimeError("Final test source lock changed")
    return digest(path)


def decoder_block(marker, block):
    import torch
    dwt_values, tensors = [], []
    for row in block.to_dict("records"):
        if digest(row["path"]) != row["output_file_sha256"]:
            raise ValueError("Received image changed")
        pixels = marks.read_image(Path(row["path"]))
        per_image = []
        for view in received_views(pixels):
            per_image.append(DwtDctWatermark(PAYLOADS[0], 12).scores(view))
            im = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            patch = marker.tm.get_the_image_for_processing(im).resize((245, 245), Image.Resampling.BILINEAR)
            tensors.append(marker.transforms.ToTensor()(patch) * 2 - 1)
        dwt_values.append(per_image)
    with torch.inference_mode():
        trust = marker.tm.decoder.decoder(torch.stack(tensors).to(marker.tm.decoder.device)).cpu().numpy()
    trust = trust.reshape(len(block), len(VIEWS), 100).astype(np.float32)
    decoded = marker.tm.ecc.decode_bitstream((trust.reshape(-1, 100) > 0).astype(np.uint8), MODE="binary")
    messages = np.array([str(msg) if valid else "" for msg, valid, schema in decoded], dtype="U61").reshape(len(block), len(VIEWS))
    valid = np.array([bool(v) for m, v, s in decoded]).reshape(len(block), len(VIEWS))
    schema = np.array([int(s) for m, v, s in decoded]).reshape(len(block), len(VIEWS))
    return {"dwt": np.asarray(dwt_values, dtype=np.float32), "trust": trust,
            "messages": messages, "valid": valid, "schema": schema}


def decode(manifest_path, destination, batch_size, device):
    import torch
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path, keep_default_na=False, dtype={"embedded_payload": str}).sort_values("image_id").reset_index(drop=True)
    mhash = digest(manifest_path)
    marker = marks.Watermarks(device, 4)
    strict = destination != OUT / "development"
    dependencies = hashlib.sha256(json.dumps({"manifest":mhash,"driver":digest(__file__),
        "weights":marker.provenance()["checkpoints"]},sort_keys=True).encode()).hexdigest()
    chunks = []
    block_size = 256
    timer = time.perf_counter()
    for start in range(0, len(frame), block_size):
        part = frame.iloc[start:start + block_size]
        path = destination / f"decoder_blocks/{start:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data["manifest_sha256"]) != mhash or not np.array_equal(data["image_id"], part.image_id.to_numpy(str)):
                    raise ValueError("Decoder resume identity mismatch")
                if strict and ("dependencies_sha256" not in data or str(data["dependencies_sha256"]) != dependencies):
                    raise ValueError("Decoder code or checkpoint dependencies changed")
                chunk = {key: data[key].copy() for key in ["dwt", "trust", "messages", "valid", "schema"]}
        else:
            minibatches = [decoder_block(marker, part.iloc[i:i + batch_size]) for i in range(0, len(part), batch_size)]
            chunk = {key: np.concatenate([d[key] for d in minibatches]) for key in minibatches[0]}
            np.savez_compressed(path, **chunk, image_id=part.image_id.to_numpy(str), manifest_sha256=mhash,dependencies_sha256=dependencies)
        chunks.append(chunk)
        print("Received-image decoding", start + len(part), "/", len(frame), "elapsed", round(time.perf_counter()-timer, 1), flush=True)
    values = {key: np.concatenate([d[key] for d in chunks]) for key in chunks[0]}
    codewords = {}
    for payload in PAYLOADS:
        marker.set_payload(payload)
        codewords[payload] = marker.codeword.copy()
    np.savez_compressed(destination / "decoder_raw.npz", **values, image_id=frame.image_id.to_numpy(str), manifest_sha256=mhash)
    write(destination / "decoder_complete.json", {"utc": utc(), "views": VIEWS, "received_images": len(frame),
        "manifest_sha256": mhash, "raw_sha256": digest(destination / "decoder_raw.npz"), "provenance": marker.provenance(),
        "driver_sha256": digest(__file__), "elapsed_seconds": time.perf_counter()-timer})
    del marker
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return codewords


def probe_features(manifest_path, destination, device, batch_size):
    import torch
    torch.set_num_threads(4)
    frame = pd.read_csv(manifest_path, keep_default_na=False).sort_values("image_id").reset_index(drop=True)
    _, checkpoint = checkpoint_defaults()
    mhash = digest(manifest_path)
    logits = {}
    for backbone in ["resnet", "clip"]:
        head_hash = digest(ROOT / f"outputs/conference_clean/{backbone}_full.npz")
        backbone_hash = digest(checkpoint_defaults()[0] if backbone == "resnet" else checkpoint)
        dependencies = hashlib.sha256(json.dumps({"head":head_hash,"backbone":backbone_hash,"driver":digest(__file__)},sort_keys=True).encode()).hexdigest()
        strict = destination != OUT / "development"
        target = destination / f"{backbone}_logits.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                if str(data["manifest_sha256"]) != mhash or not np.array_equal(data["image_id"], frame.image_id.to_numpy(str)):
                    raise ValueError("Probe resume identity mismatch")
                if str(data["head_sha256"]) != head_hash:
                    raise ValueError("Frozen probe head changed")
                if strict and ("dependencies_sha256" not in data or str(data["dependencies_sha256"]) != dependencies):
                    raise ValueError("Probe backbone or preprocessing dependencies changed")
                logits[backbone] = data["logit"].copy()
            continue
        model, transform, forward, preprocessing, identity = load_backbone(backbone, device, checkpoint)
        parts = []
        with np.load(ROOT / f"outputs/conference_clean/{backbone}_full.npz", allow_pickle=False) as head:
            coeff, intercept = head["coefficients"].astype(float), head["intercept"].astype(float)
        for start in range(0, len(frame), 512):
            part = frame.iloc[start:start+512]
            path = destination / f"{backbone}_blocks/{start:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if str(data["manifest_sha256"]) != mhash or not np.array_equal(data["image_id"], part.image_id.to_numpy(str)):
                        raise ValueError("Feature block mismatch")
                    if strict and ("dependencies_sha256" not in data or str(data["dependencies_sha256"]) != dependencies):
                        raise ValueError("Probe feature-block dependencies changed")
                    z = data["logit"].copy()
            else:
                vector = infer(part, transform, forward, device, batch_size).astype(np.float64)
                z = (vector @ coeff.T + intercept).ravel()
                np.savez_compressed(path, image_id=part.image_id.to_numpy(str), logit=z, manifest_sha256=mhash,dependencies_sha256=dependencies)
            parts.append(z)
            print(backbone, "probe", start+len(part), "/", len(frame), flush=True)
        logits[backbone] = np.concatenate(parts)
        np.savez_compressed(target, image_id=frame.image_id.to_numpy(str), logit=logits[backbone], manifest_sha256=mhash,
            head_sha256=head_hash,dependencies_sha256=dependencies)
        write(destination / f"{backbone}_complete.json", {"utc": utc(), "received_images": len(frame), "manifest_sha256":mhash,
              "head_sha256": digest(ROOT / f"outputs/conference_clean/{backbone}_full.npz"), "preprocessing":preprocessing,
              "identity":identity, "inference_sha256":digest(target)})
        del model, transform, forward
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return logits


def feature_rows(frame, raw, logits, codewords):
    rows = []
    records = frame.to_dict("records")
    for method in METHODS:
        margin = raw["dwt"] / 12 if method == "dwt_dct" else raw["trust"][:, :, :16]
        hard = margin >= 0 if method == "dwt_dct" else margin > 0
        same = np.mean(hard[:, 0] == hard[:, 1], axis=1)
        norms = np.linalg.norm(margin[:, 0], axis=1) * np.linalg.norm(margin[:, 1], axis=1)
        consistency = np.sum(margin[:, 0] * margin[:, 1], axis=1) / np.maximum(norms, 1e-12)
        for payload in PAYLOADS:
            expected = np.array(list(payload), dtype=np.uint8)
            agreements = np.mean(hard == expected[None, None, :], axis=2)
            signed = margin[:, 0] * (2*expected.astype(float)-1)
            if method == "trustmark":
                exact = raw["valid"] & (raw["schema"] == 1) & (raw["messages"] == payload + "0" * 45)
                ecc = raw["valid"][:, 0].astype(float)
                code_agree = np.mean((raw["trust"][:, 0] > 0) == codewords[payload], axis=1)
            else:
                exact = agreements == 1
                ecc = np.zeros(len(frame))  # DWT has no ECC validity claim.
                code_agree = agreements[:, 0]
            for i, record in enumerate(records):
                if record["watermark_method"] not in ("unmarked", method):
                    continue
                rows.append({**{key: record[key] for key in ["image_id", "source_id", "role", "collection", "label", "watermark_method", "embedded_payload", "condition"]},
                    "detector":method, "query_payload":payload, "resnet_logit":float(logits["resnet"][i]), "clip_logit":float(logits["clip"][i]),
                    "agreement":float(agreements[i,0]), "min_view_agreement":float(agreements[i].min()),
                    "signed_margin":float(np.tanh(np.mean(signed[i]))), "q10_margin":float(np.tanh(np.quantile(signed[i],.1))),
                    "mean_abs_margin":float(np.log1p(np.mean(np.abs(margin[i,0])))), "stability":float(same[i]),
                    "margin_stability":float(consistency[i]), "full_match":float(exact[i,0]), "full_match_views":float(exact[i].all()),
                    "ecc_valid":float(ecc[i]), "codeword_agreement":float(code_agree[i])})
    result = pd.DataFrame(rows)
    if not np.isfinite(result[[*FEATURES, "resnet_logit", "clip_logit"]].to_numpy()).all():
        raise ValueError("Nonfinite features")
    if result.duplicated(["image_id","detector","query_payload"]).any():
        raise ValueError("Duplicate query row")
    return result


def run(kind, manifest, destination, device, batch_size):
    frozen = check_final_lock(kind)
    if kind != "development":
        source_path = OUT / "data_inventory" / ("calibration_source_manifest.csv" if kind == "calibration" else "primary_source_manifest.csv")
        registered = pd.read_csv(source_path, keep_default_na=False)
        registered = registered.loc[registered.role.eq(kind)]
        supplied = pd.read_csv(manifest, keep_default_na=False)
        if set(supplied.role) != {kind}:
            raise RuntimeError("Prospective manifest has incorrect source roles")
        known = registered.set_index("source_id")[["collection", "label"]].to_dict("index")
        for row in supplied[["source_id", "collection", "label"]].drop_duplicates().to_dict("records"):
            if row["source_id"] not in known or str(row["collection"]) != str(known[row["source_id"]]["collection"]) or int(row["label"]) != int(known[row["source_id"]]["label"]):
                raise RuntimeError("Prospective manifest differs from registered sources")
        if destination.resolve() == (OUT / kind).resolve() and set(supplied.source_id) != set(registered.source_id):
            raise RuntimeError("Primary inference must include the exact registered source set")
    destination.mkdir(parents=True, exist_ok=True)
    started = destination / "inference_started.json"
    if not started.exists():
        write(started, {"utc":utc(), "kind":kind, "protocol_sha256_before_scores":frozen,
              "manifest_sha256":digest(manifest), "driver_sha256":digest(__file__)})
    complete = destination / "features_complete.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        assert digest(destination / "features.csv") == saved["features_sha256"]
        if digest(manifest) != saved["manifest_sha256"]:
            raise RuntimeError("Completed feature manifest changed")
        if kind != "development" and (saved["protocol_sha256_before_scores"] != frozen or saved["driver_sha256"] != digest(__file__)):
            raise RuntimeError("Completed prospective feature code/protocol changed")
        return
    codewords = decode(manifest,destination,batch_size,device)
    logits = probe_features(manifest,destination,device,batch_size)
    frame = pd.read_csv(manifest,keep_default_na=False,dtype={"embedded_payload":str}).sort_values("image_id").reset_index(drop=True)
    with np.load(destination / "decoder_raw.npz",allow_pickle=False) as values:
        assert np.array_equal(values["image_id"],frame.image_id.to_numpy(str))
        raw = {key:values[key] for key in ["dwt","trust","messages","valid","schema"]}
    result = feature_rows(frame,raw,logits,codewords)
    result.to_csv(destination / "features.csv",index=False)
    write(complete,{"utc":utc(), "manifest_sha256":digest(manifest), "features_sha256":digest(destination/"features.csv"),
          "received_images":len(frame),"query_rows":len(result), "sources":frame.source_id.nunique(),
          "probe_refitting":False,"metadata_in_prediction_features":False,"clean_reference_in_prediction_features":False,
          "protocol_sha256_before_scores":frozen,"driver_sha256":digest(__file__)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=["prepare-development","development","calibration","test"])
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--destination",type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--batch-size",type=int,default=16)
    args = parser.parse_args()
    if args.stage == "prepare-development":
        prepare_development()
    elif args.stage == "development":
        prepare_development()
        run("development",OUT/"development/manifest.csv",OUT/"development",args.device,args.batch_size)
    else:
        if not args.manifest or not args.destination:
            parser.error("Final stages require manifest and destination")
        run(args.stage,args.manifest,args.destination,args.device,args.batch_size)


if __name__ == "__main__":
    main()
