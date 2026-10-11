"""Check whether the CLIP origin score is explained by sampling or low-level cues.

Reports exact and perceptual duplicates, native sizes, single-feature AUCs,
and CLIP AUC on clean images and on attacks applied to both classes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from aiwstudy.evaluation import stratified_cluster_bootstrap_ci
from aiwstudy.models import LinearProbe

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"data/raw")


def _ahash(path: Path) -> int:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits(small.reshape(-1) >= small.mean()).view(np.uint64)[0])


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _pixel_md5(path: Path) -> str:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return hashlib.md5(np.ascontiguousarray(image).tobytes()).hexdigest()


def duplicates(frame: pd.DataFrame) -> dict:
    hashes = []
    for row in frame.itertuples(index=False):
        path = Path(row.path)
        hashes.append((_pixel_md5(path), _ahash(path)))
    frame = frame.copy()
    frame["md5"] = [item[0] for item in hashes]
    frame["ahash"] = [item[1] for item in hashes]
    exact = int(frame["md5"].duplicated().sum())
    exact_groups = int(frame.loc[frame["md5"].duplicated(keep=False), "md5"].nunique())
    near = 0
    by_gen = {}
    for generator, subset in frame.groupby("generator"):
        values = subset["ahash"].to_numpy()
        count = 0
        for index, value in enumerate(values):
            count += sum(_hamming(int(value), int(other)) <= 2 for other in values[index + 1 :])
        by_gen[str(generator)] = int(count)
        near += count
    return {
        "sources": int(len(frame)),
        "exact_duplicate_extras": exact,
        "exact_duplicate_groups": exact_groups,
        "near_pairs_hamming_le_2": near,
        "near_pairs_by_generator": by_gen,
        "native_size": {
            str(generator): {
                "n": int(len(subset)),
                "unique_sizes": int(subset["size"].nunique()) if "size" in subset else 0,
            }
            for generator, subset in frame.groupby("generator")
        },
    }


def _features(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    pixels = image.astype(np.float32)
    mean = pixels.mean(axis=(0, 1))
    std = pixels.std(axis=(0, 1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    yy, xx = np.indices(spectrum.shape)
    radius = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    high = spectrum[radius > 48].mean()
    low = spectrum[radius <= 16].mean()
    return np.array(
        [
            *mean,
            *std,
            hsv[:, :, 1].mean(),
            hsv[:, :, 2].mean(),
            np.log1p(lap),
            np.log1p(high),
            np.log1p(low),
            high / (low + 1e-6),
        ],
        dtype=np.float64,
    )


def low_level(sources: pd.DataFrame) -> dict:
    clean = sources[sources["group"].isin(["R-C", "S-U"])].copy()
    matrix = np.stack([_features(path) for path in clean["path"]])
    labels = clean["label"].to_numpy()
    names = [
        "mean_b",
        "mean_g",
        "mean_r",
        "std_b",
        "std_g",
        "std_r",
        "saturation",
        "value",
        "log_sharpness",
        "log_high_freq",
        "log_low_freq",
        "freq_ratio",
    ]
    single = {}
    for index, name in enumerate(names):
        score = matrix[:, index]
        auc = float(roc_auc_score(labels, score))
        single[name] = auc if auc >= 0.5 else 1.0 - auc
    train = clean["split"] == "train"
    test = clean["split"] == "test"
    probe = LinearProbe.fit(matrix[train.to_numpy()], labels[train.to_numpy()])
    scores = probe.predict_proba(matrix[test.to_numpy()])
    test_labels = labels[test.to_numpy()]
    groups = clean.loc[test, "source_id"].to_numpy()
    lower, upper = stratified_cluster_bootstrap_ci(test_labels, scores, groups, samples=2000, seed=20260920)
    return {
        "single_feature_auc_abs": single,
        "logistic_test_auc": float(roc_auc_score(test_labels, scores)),
        "logistic_test_ci95": [lower, upper],
        "test_n": int(test.sum()),
    }


def clip_slices(sources: pd.DataFrame) -> dict:
    values = np.load(ROOT / "outputs" / "clip_eval_embeddings.npz", allow_pickle=True)
    vectors = pd.DataFrame(
        {"image_id": values["image_id"].astype(str), "embedding": list(values["embedding"])}
    )
    derived = pd.read_csv(ROOT / "data" / "manifest_derived.csv", keep_default_na=False)
    derived["image_id"] = derived["image_id"].astype(str)
    test = derived[derived["split"] == "test"].merge(vectors, on="image_id", how="inner")
    scores = LinearProbe.load(ROOT / "outputs" / "clip.npz").predict_proba(
        np.stack(test["embedding"].to_numpy())
    )
    test = test.copy()
    test["clip_score"] = scores

    def auc(frame: pd.DataFrame) -> dict | None:
        if frame["label"].nunique() != 2 or len(frame) < 20:
            return None
        labels = frame["label"].to_numpy()
        clip = frame["clip_score"].to_numpy()
        lower, upper = stratified_cluster_bootstrap_ci(
            labels, clip, frame["source_id"].to_numpy(), samples=2000, seed=20260920
        )
        return {"roc_auc": float(roc_auc_score(labels, clip)), "ci95": [lower, upper], "n": int(len(frame))}

    report = {
        "clean_unmarked": auc(test[test["group"].isin(["R-C", "S-U"])]),
        "jpeg_q50_both_classes": auc(test[(test["attack"] == "jpeg") & (test["severity"].astype(int) == 2)]),
        "blur_both_classes": auc(test[(test["attack"] == "blur") & (test["severity"].astype(int) == 2)]),
        "crop_both_classes": auc(test[(test["attack"] == "crop") & (test["severity"].astype(int) == 1)]),
        "grid_shift_both_classes": auc(test[test["attack"] == "regenerate"]),
        "spectral_both_classes": auc(test[test["attack"] == "spectral"]),
    }
    clean = test[test["group"].isin(["R-C", "S-U"])]
    real = clean[clean["label"] == 0]
    for generator, subset in clean[clean["label"] == 1].groupby("generator"):
        report[f"clean_{generator}_vs_real"] = auc(pd.concat([real, subset], ignore_index=True))
    return report


def main() -> None:
    sources = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
    sizes = []
    for path in sources["path"]:
        with Image.open(path) as image:
            sizes.append(f"{image.size[0]}x{image.size[1]}")
    sources["size"] = sizes
    size_counts = {
        str(generator): subset["size"].value_counts().head(5).astype(int).to_dict()
        for generator, subset in sources.groupby("generator")
    }
    report = {
        "duplicates": duplicates(sources),
        "top_native_sizes": size_counts,
        "low_level": low_level(sources),
        "clip_slices": clip_slices(sources),
        "coco_name_span": [
            sources.loc[sources["generator"] == "real", "path"].map(lambda p: Path(p).name).min(),
            sources.loc[sources["generator"] == "real", "path"].map(lambda p: Path(p).name).max(),
        ],
        "synth_name_span": {
            str(generator): [
                subset["path"].map(lambda p: Path(p).name).min(),
                subset["path"].map(lambda p: Path(p).name).max(),
            ]
            for generator, subset in sources[sources["generator"] != "real"].groupby("generator")
        },
    }
    destination = ROOT / "outputs" / "cue_audit.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
