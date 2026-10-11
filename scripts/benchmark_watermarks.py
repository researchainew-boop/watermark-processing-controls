"""Fixed-source comparison of the classical mark and official TrustMark Q.

Stages: prepare publishes the covers needed by a separate diffusion runner;
attacks applies the original deterministic transformations; analyze scores
those outputs and optional diffusion outputs. Existing study outputs are read
only. This exploratory extension reuses test sources, not a new dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import beta
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiwstudy.attacks import apply_attack
from aiwstudy.io import read_image, write_image
from aiwstudy.watermark import DwtDctWatermark

DEFAULT_OUTPUT = ROOT / "outputs" / "extension" / "watermarks"
DEFAULT_DATA = ROOT / "data" / "extension_watermarks"
GENERATORS = ("stable-diffusion", "dalle-3", "midjourney")
PAYLOAD = "1011010010110100"
PAYLOADS = (PAYLOAD, "0100101101001011", "1100101000110101")
NOISE_SEEDS = (20261009, 20261010, 20261011)
ATTACKS = (("identity", 0), ("jpeg", 1), ("jpeg", 2), ("blur", 1),
           ("blur", 2), ("noise", 1), ("crop", 1), ("lowres", 2),
           ("regenerate", 2), ("spectral", 1))
OFFICIAL_MD5 = {
    "trustmark_Q.yaml": "fe40df84a7feeebfceb7a7678d7e6ec6",
    "decoder_Q.ckpt": "4ced90e9cfe13e3295ad082887fe9187",
    "encoder_Q.ckpt": "700328b8754db934b2f6cb5e5185d81f",
}
CDN = "https://cai-watermark.adobe.net/watermarking/trustmark-models/"


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(*values) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:24]


def resolve(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def quality(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"Unequal image shapes: {reference.shape}, {candidate.shape}")
    return {
        "psnr": float(cv2.PSNR(reference, candidate)),
        "ssim": float(structural_similarity(reference, candidate, channel_axis=2, data_range=255)),
    }


def save_pixels(path: Path, pixels: np.ndarray) -> None:
    if path.exists():
        if not np.array_equal(read_image(path), pixels):
            raise FileExistsError(f"Refusing to replace different pixels: {path}")
    else:
        write_image(path, pixels)


class Watermarks:
    def __init__(self, device: str, threads: int):
        import torch
        import trustmark
        from trustmark import TrustMark
        from torchvision import transforms

        if importlib.metadata.version("trustmark") != "0.9.2":
            raise ValueError("This protocol is locked to official trustmark==0.9.2")
        torch.set_num_threads(threads)
        torch.manual_seed(20261007)
        np.random.seed(20261007)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(20261007)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.torch = torch
        self.transforms = transforms
        self.tm = TrustMark(use_ECC=True, model_type="Q", device=device,
                            encoding_type=TrustMark.Encoding.BCH_5,
                            loadRemover=False, loadBBoxDetector=False)
        if self.tm.encoder is None or self.tm.decoder is None:
            raise RuntimeError("TrustMark checkpoint loading failed")
        self.capacity = int(self.tm.schemaCapacity())
        if self.capacity != 61 or int(self.tm.enctyp) != 1:
            raise ValueError("Unexpected default BCH5 capacity/schema")
        self.set_payload(PAYLOAD)
        model_dir = Path(trustmark.__file__).resolve().parent / "models"
        self.checkpoints = []
        for name, expected_md5 in OFFICIAL_MD5.items():
            path = model_dir / name
            actual = digest(path, "md5")
            if actual != expected_md5:
                raise ValueError(f"Official checkpoint checksum mismatch: {path}")
            self.checkpoints.append({"filename": name, "url": CDN + name,
                                     "bytes": path.stat().st_size,
                                     "official_md5": expected_md5, "actual_md5": actual,
                                     "sha256": digest(path)})

    def set_payload(self, payload):
        self.payload = payload
        self.expected = payload + "0" * (self.capacity - len(payload))
        self.codeword = self.tm.ecc.encode_binary([self.expected])[0].astype(np.uint8)
        if len(self.codeword) != 100:
            raise ValueError("Expected TrustMark's 100-bit BCH codeword")
        self.dwt = DwtDctWatermark(payload, strength=12.0)

    def encode(self, pixels: np.ndarray, method: str) -> np.ndarray:
        if method == "unmarked":
            return pixels.copy()
        if method == "dwt_dct":
            return self.dwt.embed(pixels)
        if method == "trustmark":
            im = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
            result = self.tm.encode(im, self.expected, MODE="binary", WM_STRENGTH=1.0)
            return cv2.cvtColor(np.asarray(result.convert("RGB")), cv2.COLOR_RGB2BGR)
        raise ValueError(method)

    def detect(self, pixels: np.ndarray) -> list[dict]:
        result = self.dwt.detect(pixels, threshold=13 / 16)
        classical = {"detector": "dwt_dct", "verified": int(result.verified),
                     "payload_bits": result.bits, "payload_bit_accuracy": result.bit_accuracy,
                     "raw_codeword_accuracy": result.bit_accuracy,
                     "ecc_valid": "", "decoded_schema": "", "full_message_match": int(result.bits == self.payload)}
        im = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        # Instrument the official decoder's preprocessing for uncoded bit
        # agreement; final verification still uses the public decode API.
        patch = self.tm.get_the_image_for_processing(im).resize(
            (self.tm.model_resolution_dec, self.tm.model_resolution_dec), Image.Resampling.BILINEAR)
        tensor = self.transforms.ToTensor()(patch).unsqueeze(0).to(self.tm.decoder.device) * 2 - 1
        with self.torch.inference_mode():
            raw = (self.tm.decoder.decoder(tensor) > 0).cpu().numpy()[0].astype(np.uint8)
        decoded, valid, schema = self.tm.decode(im, MODE="binary", ROTATION=False, DETECTFIRST=False)
        expected_match = bool(valid and int(schema) == int(self.tm.enctyp) and decoded == self.expected)
        learned = {"detector": "trustmark", "verified": int(expected_match),
                   "payload_bits": "".join(map(str, raw[:16].tolist())),
                   "payload_bit_accuracy": float(np.mean(raw[:16] == np.array(list(self.payload), dtype=np.uint8))),
                   "raw_codeword_accuracy": float(np.mean(raw == self.codeword)),
                   "ecc_valid": int(valid), "decoded_schema": int(schema),
                   "full_message_match": int(expected_match), "decoded_message": decoded}
        return [classical, learned]

    def provenance(self) -> dict:
        packages = {}
        for name in ("trustmark", "torch", "torchvision", "lightning", "torchmetrics",
                     "numpy", "pandas", "Pillow", "scipy", "scikit-image", "omegaconf", "einops"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "absent"
        cuda = {"available": self.torch.cuda.is_available(), "torch_cuda": self.torch.version.cuda}
        if cuda["available"]:
            cuda.update({"device": self.torch.cuda.get_device_name(0),
                         "capability": list(self.torch.cuda.get_device_capability(0)),
                         "compiled_architectures": self.torch.cuda.get_arch_list()})
        return {"python": sys.version, "platform": platform.platform(), "packages": packages,
                "device": self.tm.device, "cuda": cuda, "checkpoints": self.checkpoints,
                "trustmark_variant": "Q", "trustmark_ECC": "BCH5", "data_capacity": self.capacity,
                "expected_message": self.expected, "codeword_bits": 100,
                "trustmark_strength": 1.0, "dwt_strength": 12.0, "dwt_threshold": 13 / 16,
                "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                (Path(__file__), ROOT / "src/aiwstudy/attacks.py", ROOT / "src/aiwstudy/watermark.py")}}


def freeze_selection(args) -> pd.DataFrame:
    pairs_path = ROOT / "outputs/extension_controls/encoding/content_pairs.csv"
    manifest_path = ROOT / "outputs/extension_controls/encoding/manifest.csv"
    pairs = csv(pairs_path)
    selected = []
    for generator in GENERATORS:
        pool = pairs.loc[(pairs.split == "test") & (pairs.generator == generator)].copy()
        if len(pool) < args.pairs_per_generator:
            raise ValueError(f"Insufficient locked test pairs for {generator}")
        pool["selection_order"] = pool.pair_id.map(lambda x: hashlib.sha256(f"{args.seed}|{x}".encode()).hexdigest())
        selected.append(pool.sort_values(["selection_order", "pair_id"]).head(args.pairs_per_generator))
    chosen = pd.concat(selected).drop(columns="selection_order").reset_index(drop=True)
    if chosen.synthetic_source_id.duplicated().any() or chosen.real_source_id.duplicated().any():
        raise ValueError("Sources must be distinct in the locked matching")
    policy = {"seed": args.seed, "pairs_per_generator": args.pairs_per_generator,
              "pair_ids": chosen.pair_id.tolist(), "selection": "SHA256(seed|pair_id), test only, stratified by generator",
              "uses_detector_scores": False, "original_test_results_previously_observed": True,
              "independent_new_collection": False, "payload16": PAYLOAD, "payloads": PAYLOADS, "noise_seeds": NOISE_SEEDS,
              "trustmark": "0.9.2 Q BCH5; 16 payload bits followed by 45 fixed zero padding bits",
              "dwt": "DWT-DCT strength 12; verification >=13/16",
              "attacks": [[name, severity] for name, severity in ATTACKS],
              "pairs_sha256": digest(pairs_path), "encoding_manifest_sha256": digest(manifest_path)}
    selection_path = args.output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != policy:
            raise ValueError("Frozen selection differs; use a new output directory")
    else:
        write_json(selection_path, policy)
    chosen.to_csv(args.output / "selected_pairs.csv", index=False)
    return chosen


def prepare(args, marks: Watermarks) -> pd.DataFrame:
    chosen = freeze_selection(args)
    sources = csv(ROOT / "outputs/extension_controls/encoding/manifest.csv")
    sources = sources.loc[sources.condition == "common_png"].set_index("source_id", verify_integrity=True)
    rows = []
    started = time.perf_counter()
    for index, pair in enumerate(chosen.to_dict("records")):
        for origin, source_id in ((1, pair["synthetic_source_id"]), (0, pair["real_source_id"])):
            source = sources.loc[source_id]
            original_path = resolve(source.path)
            if digest(original_path) != source.output_file_sha256:
                raise ValueError(f"Changed locked input: {original_path}")
            pixels = read_image(original_path)
            if pixels.shape != (256, 256, 3):
                raise ValueError("Expected fixed 256x256 BGR covers")
            methods = ("unmarked", "dwt_dct", "trustmark") if origin else ("unmarked",)
            for method, payload in [(m,p) for m in methods for p in (PAYLOADS if m != 'unmarked' else ('',))]:
                marks.set_payload(payload or PAYLOAD)
                cover_id = identity("watermark_extension", source_id, method, payload)
                target = args.data / "covers" / method / f"{cover_id}.png"
                encode_started = time.perf_counter()
                candidate = marks.encode(pixels, method)
                encode_seconds = time.perf_counter() - encode_started
                save_pixels(target, candidate)
                rows.append({"cover_id": cover_id, "image_id": cover_id, "source_id": source_id,
                             "parent_id": source_id, "pair_id": pair["pair_id"], "split": "test",
                             "generator": pair["generator"] if origin else "real",
                             "paired_generator": pair["generator"], "label": origin, "input_origin_label": origin,
                             "watermark_method": method, "payload": payload,
                             "group": "R-C" if not origin else ("S-U" if method == "unmarked" else "S-W"),
                             "path": str(target), "original_path": str(original_path),
                             "output_file_sha256": digest(target), "original_file_sha256": digest(original_path),
                             "encode_seconds": encode_seconds, **quality(pixels, candidate)})
        print(f"Covers prepared: pair {index + 1}/{len(chosen)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "base_covers.csv", index=False)
    write_json(args.output / "provenance.json", {**marks.provenance(), "cover_count": len(rows),
                                                "source_count": len(chosen) * 2, "prepare_seconds": time.perf_counter() - started})
    print(f"Published diffusion input covers: {args.output / 'base_covers.csv'}", flush=True)
    return frame


def attacks(args) -> pd.DataFrame:
    covers = csv(args.output / "base_covers.csv")
    rows = []
    for cover in covers.to_dict("records"):
        path = resolve(cover["path"])
        if digest(path) != cover["output_file_sha256"]:
            raise ValueError(f"Changed prepared cover: {path}")
        pixels = read_image(path)
        for name, severity in ATTACKS:
            condition = "identity" if name == "identity" else f"{name}_s{severity}"
            target = args.data / "attacks" / condition / f"{cover['cover_id']}.png"
            save_pixels(target, apply_attack(pixels, name, severity))
            rows.append({"cover_id": cover["cover_id"], "condition": condition,
                         "attack": name, "severity": severity, "path": str(target),
                         "output_file_sha256": digest(target), "input_stage": "base_cover"})
        for seed in NOISE_SEEDS:
            condition = f'noise_sigma6_seed{seed}'
            target = args.data / 'attacks' / condition / f"{cover['cover_id']}.png"
            noise = np.random.default_rng(seed).normal(0,6,pixels.shape)
            save_pixels(target,np.clip(pixels.astype(np.float32)+noise,0,255).astype(np.uint8))
            rows.append({'cover_id':cover['cover_id'],'condition':condition,'attack':'noise','severity':1,'seed':seed,'path':str(target),'output_file_sha256':digest(target),'input_stage':'base_cover'})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "attack_manifest.csv", index=False)
    return frame


def import_conditions(args) -> pd.DataFrame:
    frames = []
    local = args.output / "attack_manifest.csv"
    if local.exists():
        frames.append(csv(local))
    for path in args.diffusion_manifest:
        other = csv(path)
        if "path" not in other and "output_path" in other:
            other = other.rename(columns={"output_path": "path"})
        if "condition" not in other and "attack" in other:
            other["condition"] = other["attack"]
        if not {"cover_id", "condition", "path"}.issubset(other.columns):
            raise ValueError("External CSV needs cover_id, condition, path (or output_path)")
        other["external_manifest"] = str(path.resolve())
        frames.append(other)
    if not frames:
        raise ValueError("No attack/diffusion outputs available to analyze")
    result = pd.concat(frames, ignore_index=True).fillna("")
    if result.duplicated(["cover_id", "condition"]).any():
        raise ValueError("Duplicate cover/condition measurements in supplied manifests")
    return result


def rate_interval(successes: int, n: int) -> list[float]:
    return [0.0 if successes == 0 else float(beta.ppf(.025, successes, n - successes + 1)),
            1.0 if successes == n else float(beta.ppf(.975, successes + 1, n - successes))]


def summarize(frame: pd.DataFrame) -> dict:
    summaries = []
    for (condition, detector), block in frame.groupby(["condition", "detector"], sort=True):
        relevant = block.loc[((block.input_origin_label == 1) & (block.watermark_method == detector)) |
                             ((block.input_origin_label == 0) & (block.watermark_method == "unmarked"))]
        for generator in ["all", *GENERATORS]:
            selected = relevant if generator == "all" else relevant.loc[relevant.paired_generator == generator]
            for origin in (0, 1):
                subset = selected.loc[selected.input_origin_label == origin]
                if subset.empty:
                    continue
                n = len(subset)
                source_rates = subset.groupby('source_id').verified.mean().to_numpy()
                rng = np.random.default_rng(20261009)
                interval = np.percentile(source_rates[rng.integers(len(source_rates),size=(2000,len(source_rates)))].mean(axis=1),[2.5,97.5]).tolist()
                successes = int(subset.verified.sum())
                summaries.append({"condition": condition, "detector": detector, "generator": generator,
                                  "input_origin_label": origin, "n_sources": len(source_rates), "n_outputs": n, "matches": successes,
                                  "expected_payload_match_rate": successes / n,
                                  "rate_ci95_source_bootstrap": interval,
                                  "rate_ci95_exact_binomial_one_output_per_source": rate_interval(successes,n) if n==len(source_rates) else None,
                                  "mean_payload_bit_accuracy": float(subset.payload_bit_accuracy.mean()),
                                  "mean_raw_codeword_accuracy": float(subset.raw_codeword_accuracy.mean()),
                                  "mean_psnr_vs_input_cover": float(subset.psnr_vs_input_cover.mean()),
                                  "mean_ssim_vs_input_cover": float(subset.ssim_vs_input_cover.mean())})
    return {"measurements": summaries,
            "rate_ci": "2000 source bootstrap draws, payload repeats kept together; small source count",
            "interpretation": "Origin labels denote input-source provenance. A diffusion-reconstructed photograph contains generated pixels; its payload match rate is not an authenticity false-positive rate.",
            "comparison_limit": "Different embedding protocols: classical 16 bits at strength 12 versus TrustMark Q 100-bit BCH codeword / 61 data bits at default strength 1. No equal-capacity or equal-distortion claim.",
            "thresholds_retuned": False, "independent_new_dataset": False}


def analyze(args, marks: Watermarks) -> pd.DataFrame:
    covers = csv(args.output / "base_covers.csv").set_index("cover_id", verify_integrity=True)
    conditions = import_conditions(args)
    if not set(conditions.cover_id).issubset(covers.index):
        raise ValueError("External manifest contains unknown cover IDs")
    previous_path = args.output / 'measurements.csv'
    rows = pd.read_csv(previous_path,keep_default_na=False,dtype={'payload':str,'payload_bits':str,'source_id':str,'cover_id':str}).to_dict('records') if previous_path.exists() else []
    completed = {(r['cover_id'],r['condition']) for r in rows}
    for index, transformed in enumerate(conditions.to_dict("records")):
        cover = covers.loc[transformed["cover_id"]].to_dict()
        path = resolve(transformed["path"])
        pixels = read_image(path)
        expected_hash = transformed.get("output_file_sha256", "")
        if expected_hash and digest(path) != expected_hash:
            raise ValueError(f"Changed attack output: {path}")
        if (transformed['cover_id'],transformed['condition']) in completed:
            continue
        input_path = resolve(cover["path"])
        if digest(input_path) != cover["output_file_sha256"]:
            raise ValueError("Prepared cover changed after publication")
        original_path = resolve(cover["original_path"])
        if digest(original_path) != cover["original_file_sha256"]:
            raise ValueError("Original cover changed after selection")
        input_quality = quality(read_image(input_path), pixels)
        original_quality = quality(read_image(original_path), pixels)
        started = time.perf_counter()
        marks.set_payload(cover['payload'] or PAYLOAD)
        decoded = marks.detect(pixels)
        seconds = time.perf_counter() - started
        origin = int(cover["input_origin_label"])
        clean = transformed["condition"] == "identity"
        group = ("R-C" if clean else "R-T") if not origin else (
            "S-U" if cover["watermark_method"] == "unmarked" else ("S-W" if clean else "S-A"))
        for detection in decoded:
            # Scientific provenance always comes from the frozen cover table,
            # even if an external attack CSV contains similarly named columns.
            rows.append({**transformed, **cover, **detection,
                         "image_id": identity("watermark_attack", transformed["cover_id"], transformed["condition"]),
                         "input_origin_label": origin, "cover_group": cover["group"], "group": group,
                         "label": int(cover["label"]), "path": str(path), "input_cover_path": str(input_path),
                         "output_file_sha256": digest(path), "decode_both_seconds": seconds,
                         "psnr_vs_input_cover": input_quality["psnr"], "ssim_vs_input_cover": input_quality["ssim"],
                         "psnr_vs_original": original_quality["psnr"], "ssim_vs_original": original_quality["ssim"]})
        if (index + 1) % 24 == 0:
            print(f"Decoded outputs: {index + 1}/{len(conditions)}", flush=True)
        # Persist progress so an interrupted inference does not hide completed measurements.
        if (index + 1) % 96 == 0:
            pd.DataFrame(rows).to_csv(args.output / "measurements.partial.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "measurements.csv", index=False)
    write_json(args.output / "summary.json", summarize(frame))
    write_json(args.output / "analysis_provenance.json", {**marks.provenance(),
               "external_manifests": {str(p.resolve()): digest(p) for p in args.diffusion_manifest},
               "covers_sha256": digest(args.output / "base_covers.csv"), "measured_outputs": len(conditions)})
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "attacks", "analyze", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--pairs-per-generator", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--diffusion-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    if args.pairs_per_generator < 1 or args.threads < 1:
        raise ValueError("Positive pair count and thread count required")
    args.output.mkdir(parents=True, exist_ok=True)
    marks = None
    if args.stage != "attacks":
        marks = Watermarks(args.device, args.threads)
    if args.stage in ("prepare", "all"):
        prepare(args, marks)
    if args.stage in ("attacks", "all"):
        attacks(args)
    if args.stage in ("analyze", "all"):
        analyze(args, marks)


if __name__ == "__main__":
    main()
