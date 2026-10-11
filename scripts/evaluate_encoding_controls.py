"""Evaluate the frozen 600-source encoding controls; no model is fitted.

Uses verified common-PNG caches and fresh JPEG-round-trip embeddings.
All evaluation outputs stay in the extension directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "extension_controls" / "encoding"
FIT = ROOT / "outputs" / "conference_clean"
CONDITIONS = ["common_png", "jpeg_q95", "jpeg_q75", "jpeg_q50"]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def cached_features(paths: list[Path], ids: list[str]) -> np.ndarray:
    wanted, found = set(ids), {}
    for path in paths:
        with np.load(path, allow_pickle=True) as obj:
            image_ids = obj["image_id"].astype(str)
            indexes = np.flatnonzero(np.isin(image_ids, list(wanted)))
            selected = obj["embedding"][indexes].astype(np.float32)
            for identity, vector in zip(image_ids[indexes], selected):
                found.setdefault(identity, vector)
    if set(ids) - found.keys():
        raise ValueError("Missing original clean unmarked cache rows")
    return np.stack([found[identity] for identity in ids])


def check_common_pixels(frame: pd.DataFrame) -> None:
    for row in frame.loc[frame.condition.eq("common_png")].to_dict("records"):
        original = ROOT / "data" / "processed" / "base" / f"{row['source_id']}.png"
        left = cv2.imread(str(original))
        right = cv2.imread(row["path"])
        if left is None or right is None or not np.array_equal(left, right):
            raise ValueError(f"Common-PNG pixels changed: {row['source_id']}")
    provenance = json.loads((FIT / "provenance.json").read_text(encoding="utf-8"))
    for path in original_caches():
        key = path.relative_to(ROOT).as_posix()
        known = {k.replace("\\", "/"): v for k, v in provenance["input_sha256"].items()}
        if digest(path) != known[key]:
            raise ValueError(f"Original feature cache changed: {path}")


def original_caches() -> list[Path]:
    return [ROOT / "outputs/resnet_embeddings.npz",
            ROOT / "outputs/clip_train_embeddings.npz",
            ROOT / "outputs/clip_eval_embeddings.npz"]


def checkpoint_defaults() -> tuple[Path, Path]:
    import torch
    resnet = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    hf = Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots"
    candidates = sorted(hf.glob("*/open_clip_model.safetensors"))
    if not resnet.is_file() or not candidates:
        raise FileNotFoundError("Existing ResNet/OpenAI CLIP checkpoint caches are required; no download is performed")
    return resnet, candidates[-1]


def load_backbone(name: str, device: str, clip_checkpoint: Path):
    import torch
    from aiwstudy.models import FrozenResNet18, image_to_tensor
    if name == "resnet":
        model = FrozenResNet18(pretrained=True).to(device).eval()
        def transform(path):
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise ValueError(path)
            return image_to_tensor(pixels)
        def forward(batch):
            return model.embed(batch)
        return model, transform, forward, repr(image_to_tensor), "torchvision ResNet18_Weights.DEFAULT / ImageNet1K_V1"
    import open_clip
    # Reproduce the study's ViT-B-32 architecture (GELU) and OpenAI weights.
    # The cache comparability anchors must pass before reusing any features.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None, force_quick_gelu=False,
        device=device, image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        image_interpolation="bicubic", image_resize_mode="shortest",
    )
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(clip_checkpoint)), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    def transform(path):
        with Image.open(path) as im:
            return preprocess(im.convert("RGB"))
    def forward(batch):
        return model.encode_image(batch).float().cpu().numpy()
    return model, transform, forward, repr(preprocess), "ViT-B-32 / GELU / OpenAI parameter checkpoint"


def infer(rows: pd.DataFrame, transform, forward, device: str, batch_size: int) -> np.ndarray:
    import torch
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows.iloc[start:start + batch_size]
            tensors = [transform(path) for path in part["path"]]
            batch = torch.stack(tensors).to(device)
            results.append(np.asarray(forward(batch), dtype=np.float32))
            if start % (batch_size * 25) == 0:
                print(f"Feature inference {start + len(part)}/{len(rows)}", flush=True)
    return np.concatenate(results)


def extract(frame: pd.DataFrame, name: str, device: str, clip_checkpoint: Path,
            batch_size: int) -> tuple[np.ndarray, dict]:
    common = frame.loc[frame.condition.eq("common_png")].copy()
    fresh = frame.loc[~frame.condition.eq("common_png")].copy()
    cache_paths = original_caches()[:1] if name == "resnet" else original_caches()[1:]
    old = cached_features(cache_paths, common.source_id.tolist())
    model, transform, forward, preprocessing, model_identity = load_backbone(name, device, clip_checkpoint)
    # Scientific comparability check: re-extract one baseline per split/source
    # collection before mixing old and fresh features from different runtimes.
    anchors = common.groupby(["split", "generator"], sort=True).head(1)
    current = infer(anchors, transform, forward, device, batch_size).astype(np.float64)
    baseline = cached_features(cache_paths, anchors.source_id.tolist()).astype(np.float64)
    cosine = np.sum(current * baseline, axis=1) / (np.linalg.norm(current, axis=1) * np.linalg.norm(baseline, axis=1))
    relative_l2 = np.linalg.norm(current - baseline, axis=1) / np.linalg.norm(baseline, axis=1)
    audit = {
        "model": model_identity, "preprocessing": preprocessing,
        "anchor_sources": anchors.source_id.tolist(), "anchor_cosine": cosine.tolist(),
        "anchor_relative_l2": relative_l2.tolist(), "required_cosine": 0.99999,
        "maximum_relative_l2": 0.005,
    }
    write_json(OUT / f"{name}_comparability.json", audit)
    if cosine.min() < 0.99999 or relative_l2.max() > 0.005:
        raise ValueError(f"{name}: fresh and cached baseline preprocessing/model features differ; cache reuse rejected")
    new = infer(fresh, transform, forward, device, batch_size)
    lookup = {identity: vector for identity, vector in zip(common.image_id, old)}
    lookup.update({identity: vector for identity, vector in zip(fresh.image_id, new)})
    vectors = np.stack([lookup[identity] for identity in frame.image_id])
    np.savez(OUT / f"{name}_encoding_embeddings.npz",
             image_id=frame.image_id.to_numpy(), source_id=frame.source_id.to_numpy(),
             embedding=vectors, reused_common_png=frame.condition.eq("common_png").to_numpy())
    del model, forward, transform
    gc.collect()
    import torch
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return vectors, audit


def model_scores(vectors: np.ndarray, fit_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(fit_path) as obj:
        weights, intercept = obj["coefficients"], obj["intercept"]
        if not np.array_equal(obj["classes"], np.array([0, 1])):
            raise ValueError("Expected binary class order 0/1")
    # Probabilities preserve the saved model's arithmetic dtype for fusion;
    # discrimination uses its unsaturated linear score evaluated in float64.
    native = (vectors.astype(weights.dtype) @ weights.T + intercept).ravel()
    probability = expit(native)
    logit = (vectors.astype(np.float64) @ weights.astype(np.float64).T +
             intercept.astype(np.float64)).ravel()
    return probability, logit


def score(frame: pd.DataFrame, vectors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict, list[Path]]:
    from aiwstudy.watermark import DwtDctWatermark
    previous = json.loads((FIT / "metrics.json").read_text(encoding="utf-8"))
    watermark = DwtDctWatermark("1011010010110100", 12)
    frame = frame.copy()
    frame["active_score"] = [
        watermark.detect(cv2.imread(path)).bit_accuracy for path in frame.path
    ]
    systems, files = {}, []
    for variant in ["full", "no_matched_reals"]:
        for backbone in ["resnet", "clip"]:
            prefix = "" if backbone == "resnet" else "clip_"
            name = f"{backbone}_{variant}"
            head = FIT / f"{backbone}_{variant}.npz"
            probability, logit = model_scores(vectors[backbone], head)
            frame[name + "_probability"], frame[name + "_logit"] = probability, logit
            systems[name] = previous["variants"][variant]["systems"][prefix + "probe"]["threshold"]
            fused = name + "_fusion"
            fusion_path = FIT / f"{backbone}_fusion_{variant}.npz"
            inputs = np.column_stack([probability, frame.active_score.to_numpy()])
            fprob, flogit = model_scores(inputs, fusion_path)
            frame[fused + "_probability"], frame[fused + "_logit"] = fprob, flogit
            systems[fused] = previous["variants"][variant]["systems"][prefix + "fusion"]["threshold"]
            files.extend([head, fusion_path])
    return frame, systems, files


def validation_5pct(frame: pd.DataFrame, column: str) -> dict:
    fpr, tpr, thresholds = roc_curve(frame.label, frame[column], drop_intermediate=False)
    allowed = np.flatnonzero((fpr <= .05) & np.isfinite(thresholds))
    if not len(allowed):
        value = float(np.nextafter(frame[column].max(), np.inf))
        return {"linear_threshold": value, "validation_fpr": 0.0, "validation_tpr": 0.0}
    index = allowed[np.argmax(tpr[allowed])]
    return {"linear_threshold": float(thresholds[index]),
            "validation_fpr": float(fpr[index]), "validation_tpr": float(tpr[index])}


def decisions(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"fpr": float(pred[y == 0].mean()), "tpr": float(pred[y == 1].mean())}


def analysis(test: pd.DataFrame, systems: dict, calibrated: dict, samples: int, seed: int) -> dict:
    # One pair contains two different sources and all four encoding conditions.
    # Resample pairs within paired synthetic collection, preserving both classes.
    test = test.reset_index(drop=True)
    if test.groupby(["pair_id", "condition"]).size().ne(2).any():
        raise ValueError("Exactly two class members required per pair/condition")
    pair_list = sorted(test.pair_id.unique())
    strata = {gen: sorted(sub.pair_id.unique()) for gen, sub in test.groupby("paired_generator")}
    pair_position = {identity: i for i, identity in enumerate(pair_list)}
    indexes = np.empty((len(pair_list), len(CONDITIONS), 2), dtype=int)
    for p, identity in enumerate(pair_list):
        for c, condition in enumerate(CONDITIONS):
            sub = test.loc[test.pair_id.eq(identity) & test.condition.eq(condition)].sort_values("label")
            if sub.label.tolist() != [0, 1]:
                raise ValueError("Each matched pair must have one real and one AI source")
            indexes[p, c] = sub.index.to_numpy()
    y = np.tile([0, 1], len(pair_list))
    probabilities = {name: test[name + "_probability"].to_numpy()[indexes] for name in systems}
    rankings = {name: test[name + "_logit"].to_numpy()[indexes] for name in systems}
    report, draws = {}, {name: [] for name in systems}
    for name, threshold in systems.items():
        report[name] = {}
        for c, condition in enumerate(CONDITIONS):
            ranking, probability = rankings[name][:, c].ravel(), probabilities[name][:, c].ravel()
            report[name][condition] = {
                "sources": len(y), "pairs": len(pair_list), "real": len(pair_list), "ai": len(pair_list),
                "roc_auc": float(roc_auc_score(y, ranking)),
                "original_probability_threshold": threshold,
                "original_fixed": decisions(y, probability >= threshold),
                "extension_validation_5pct": {**calibrated[name], **decisions(y, ranking >= calibrated[name]["linear_threshold"])},
            }
    rng = np.random.default_rng(seed)
    for draw in range(samples):
        chosen = np.concatenate([
            rng.choice([pair_position[identity] for identity in pairs], len(pairs), replace=True)
            for _, pairs in sorted(strata.items())
        ])
        for name, threshold in systems.items():
            values = []
            for c in range(len(CONDITIONS)):
                ranking = rankings[name][chosen, c].ravel()
                probability = probabilities[name][chosen, c].ravel()
                fixed = decisions(y, probability >= threshold)
                adjusted = decisions(y, ranking >= calibrated[name]["linear_threshold"])
                values.append([roc_auc_score(y, ranking), fixed["fpr"], fixed["tpr"], adjusted["fpr"], adjusted["tpr"]])
            draws[name].append(values)
        if draw and draw % 500 == 0:
            print(f"Pair bootstrap {draw}/{samples}", flush=True)
    for name in systems:
        matrix = np.asarray(draws[name])
        for c, condition in enumerate(CONDITIONS):
            point = report[name][condition]
            cis = np.percentile(matrix[:, c], [2.5, 97.5], axis=0)
            point["roc_auc_ci"] = cis[:, 0].tolist()
            point["original_fixed"]["fpr_ci"], point["original_fixed"]["tpr_ci"] = cis[:, 1].tolist(), cis[:, 2].tolist()
            point["extension_validation_5pct"]["fpr_ci"], point["extension_validation_5pct"]["tpr_ci"] = cis[:, 3].tolist(), cis[:, 4].tolist()
            point["auc_change_from_common_png"] = {
                "estimate": point["roc_auc"] - report[name]["common_png"]["roc_auc"],
                "ci95": np.percentile(matrix[:, c, 0] - matrix[:, 0, 0], [2.5, 97.5]).tolist(),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--clip-checkpoint", type=Path)
    parser.add_argument("--reuse-extension-embeddings", action="store_true",
                        help="Use previously verified extension embeddings; no GPU inference")
    args = parser.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    torch.set_num_threads(4)
    frame = pd.read_csv(OUT / "manifest.csv", keep_default_na=False,
                        dtype={"image_id": str, "source_id": str, "pair_id": str})
    if len(frame) != 2400 or frame.source_id.nunique() != 600 or frame.image_id.duplicated().any():
        raise ValueError("Expected fixed 2400-image/600-source control manifest")
    if not set(frame.split).issubset({"val", "test"}):
        raise ValueError("No training images permitted in extension evaluation")
    check_common_pixels(frame)
    resnet_checkpoint, clip_default = checkpoint_defaults()
    clip_checkpoint = args.clip_checkpoint.resolve() if args.clip_checkpoint else clip_default
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(clip_checkpoint)
    vectors, audits = {}, {}
    for name in ["resnet", "clip"]:
        path = OUT / f"{name}_encoding_embeddings.npz"
        if args.reuse_extension_embeddings:
            audit = json.loads((OUT / f"{name}_comparability.json").read_text(encoding="utf-8"))
            if min(audit["anchor_cosine"]) < .99999 or max(audit["anchor_relative_l2"]) > .005:
                raise ValueError("Previously extracted features failed comparability checks")
            with np.load(path, allow_pickle=True) as obj:
                if not np.array_equal(obj["image_id"].astype(str), frame.image_id.to_numpy()):
                    raise ValueError("Extension feature row order differs")
                vectors[name] = obj["embedding"]
            audits[name] = audit
        else:
            print(f"Extract {name} on {device}: 600 verified reused + 1800 fresh JPEG rows", flush=True)
            vectors[name], audits[name] = extract(frame, name, device, clip_checkpoint, args.batch_size)
    scored, systems, model_files = score(frame, vectors)
    scored.to_csv(OUT / "encoding_predictions.csv", index=False)
    validation, test = scored.loc[scored.split.eq("val")], scored.loc[scored.split.eq("test")]
    calibrated = {name: validation_5pct(validation, name + "_logit") for name in systems}
    report = {
        "fixed_models": True, "new_training": False, "source_count": 600,
        "test_sources": 360, "test_pairs": 180, "validation_sources": 240,
        "bootstrap_samples": args.samples, "bootstrap_unit": "matched real/AI pair",
        "bootstrap_strata": "paired synthetic collection",
        "calibration": "Extension validation only, pooled equally across the four fixed conditions; max TPR at FPR<=5%, highest threshold on ties",
        "overall": analysis(test, systems, calibrated, args.samples, 20261007),
        "per_generator": {},
        "caution": "Same selected collections and prior compression; approximate CLIP matching conditions on a detector representation. No independent population or diffusion-removal inference.",
    }
    for generator, subset in test.groupby("paired_generator"):
        print(f"Analyze paired collection {generator}", flush=True)
        report["per_generator"][generator] = analysis(subset, systems, calibrated, args.samples, 20261007)
    write_json(OUT / "encoding_metrics.json", report)
    packages = ["torch", "torchvision", "open-clip-torch", "timm", "numpy", "pandas", "scipy", "scikit-learn", "Pillow"]
    inputs = [Path(__file__), OUT / "manifest.csv", OUT / "selection.json",
              ROOT / "src/aiwstudy/models.py", ROOT / "src/aiwstudy/watermark.py",
              resnet_checkpoint, clip_checkpoint, *original_caches(), *model_files]
    provenance = {
        "python": sys.version, "platform": platform.platform(), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch_cuda": torch.version.cuda, "batch_size": args.batch_size,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "checkpoints": {"resnet": str(resnet_checkpoint), "clip": str(clip_checkpoint)},
        "input_sha256": {str(path): digest(path) for path in inputs},
        "comparability": audits, "inference_rows_per_backbone": 1800,
        "baseline_cache_rows_per_backbone": 600, "anchor_verification_inference_additional": 7,
        "model_fitting_performed": False, "original_artifacts_modified": False,
    }
    write_json(OUT / "evaluation_provenance.json", provenance)
    print(f"Completed frozen encoding evaluation: {OUT}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
