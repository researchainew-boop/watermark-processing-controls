"""Frozen 204-source, two-watermark regeneration comparison and figure example.

This is a larger matched-payload robustness experiment, not a new independent
test collection. It preserves the original extension's outputs. The generated
illustration is kept out of all benchmark aggregates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiwstudy.attacks import apply_attack
from aiwstudy.watermark import DwtDctWatermark

OUT = ROOT / "outputs/phase3/regeneration"
DATA = ROOT / "data/phase3_regeneration"
ASSETS = ROOT / "manuscript/assets"
COVERS = ROOT / "outputs/phase3/watermarks/base_covers.csv"
CACHE = Path.home() / ".cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
PAYLOAD = "1011010010110100"
SEED = 20261009
CONDITIONS = (("vae_only", None), ("diffusion_strength0p25", .25), ("diffusion_strength0p50", .50))
PROMPT = "Create a single original photorealistic image for an academic image watermark experiment illustration. A blue glazed ceramic cup, a ripe pear, and a folded woven cloth on a weathered wooden table beside a garden window. Natural side lighting, visible fine wood grain and ceramic texture, soft green foliage in the background. Balanced composition with all main objects in the central area. Square composition. No people, no lettering, no logos, no watermarks, no border, no infographic or labels. This will be processed afterward by the actual research algorithms, so provide only the clean original image."


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def example_prepare():
    """Apply the actual classical algorithm and study edits to our own image."""
    original = ASSETS / "generated-original.png"
    pixels = cv2.imread(str(original), cv2.IMREAD_COLOR)
    if pixels is None:
        raise FileNotFoundError(original)
    pixels = cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_AREA)
    watermark = DwtDctWatermark(PAYLOAD, strength=12.)
    marked = watermark.embed(pixels)
    images = [("original", pixels), ("marked", marked),
              ("jpeg50", apply_attack(marked, "jpeg", 2)),
              ("crop", apply_attack(marked, "crop", 1)),
              ("lowres32", apply_attack(marked, "lowres", 2))]
    rows = []
    for name, im in images:
        path = ASSETS / f"example_{name}.png"
        if path.exists() and not np.array_equal(cv2.imread(str(path)), im):
            raise FileExistsError(f"Refusing to replace different illustration pixels: {path}")
        if not path.exists():
            if not cv2.imwrite(str(path), im):
                raise IOError(path)
        result = watermark.detect(im, threshold=13/16)
        rows.append({"condition": name, "path": str(path), "file_sha256": digest(path),
                     "decoded_bits": result.bits, "agreement_bits": int(round(result.bit_accuracy * 16)),
                     "verified": bool(result.verified), "size": [256, 256]})
    diffusion = ASSETS / "example_diffusion0p50.png"
    if diffusion.exists():
        im = cv2.imread(str(diffusion), cv2.IMREAD_COLOR)
        result = watermark.detect(im, threshold=13/16)
        rows.append({"condition": "diffusion0p50", "path": str(diffusion), "file_sha256": digest(diffusion),
                     "decoded_bits": result.bits, "agreement_bits": int(round(result.bit_accuracy * 16)),
                     "verified": bool(result.verified), "size": [256, 256]})
    labels = {"original": "Original", "marked": "Marked", "jpeg50": "JPEG 50",
              "crop": "Crop (12 px)", "lowres32": "Low-res 32", "diffusion0p50": "Diffusion 0.50"}
    measurements = ASSETS / "example_measurements.json"
    write_json(measurements, {"payload16": PAYLOAD, "method": "dwt_dct", "strength": 12.,
        "figure_example_only": True, "excluded_from_quantitative_benchmark": True,
        "diffusion_parameters": {"cached_revision": CACHE.name, "scheduler": "DDIM",
            "num_inference_steps": 20, "strength": .50, "guidance_scale": 1., "prompt": "", "seed": SEED},
        "images": [{"condition": row["condition"], "label": labels[row["condition"]],
            "path": str(Path(row["path"]).relative_to(ROOT)).replace("\\", "/"),
            "bits": row["decoded_bits"], "agreement_count": row["agreement_bits"],
            "verified": row["verified"]} for row in rows]})
    write_json(ASSETS / "generated-asset-provenance.json", {
        "project_generated": True, "dataset_images_used": False,
        "purpose": "Paper pipeline illustration only; excluded from all benchmark measurements",
        "creation_tool": "OpenAI built-in image generation", "mode": "new original image",
        "creation_prompt": PROMPT, "original_image": str(original.relative_to(ROOT)).replace("\\", "/"),
        "original_file_sha256": digest(original), "payload": PAYLOAD,
        "watermark": "Project DWT-DCT implementation, strength12, verification>=13/16",
        "standardization": "OpenCV resize INTER_AREA to256x256",
        "jpeg50": "Project JPEG severity2: OpenCV quality50",
        "crop": "Project crop severity1:12pixels from each256pixel side, INTER_LINEAR restore",
        "lowres32": "Project lowres severity2: INTER_AREA to32x32, INTER_LINEAR restore",
        "diffusion": "Same cached SD1.5-family img2img protocol as Experiment3, strength0.50",
        "figure_images": [{**row, "path": str(Path(row["path"]).relative_to(ROOT)).replace("\\", "/")} for row in rows],
        "assets": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path)}
            for path in [original, *[Path(row["path"]) for row in rows], measurements]],
        "figure_example_only": True, "excluded_from_quantitative_benchmark": True,
        "driver_sha256": digest(Path(__file__))})
    print("Prepared original illustration transforms and actual recovered bits", flush=True)


def selected_covers():
    frame = pd.read_csv(COVERS, keep_default_na=False, dtype=str)
    frame = frame.loc[(frame.input_origin_label == "1") & (frame.payload == PAYLOAD)
                      & frame.watermark_method.isin(["dwt_dct", "trustmark"])].copy()
    frame = frame.sort_values(["source_id", "watermark_method"]).reset_index(drop=True)
    assert len(frame) == 408 and frame.source_id.nunique() == 204
    assert frame.groupby("source_id").watermark_method.nunique().eq(2).all()
    assert frame.groupby("generator").source_id.nunique().eq(68).all()
    for row in frame.to_dict("records"):
        if digest(row["path"]) != row["output_file_sha256"]:
            raise ValueError(f"Input cover hash mismatch: {row['cover_id']}")
    return frame


def load_pipeline(mode):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(CACHE), torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.float()
    original_encode, original_decode = pipe.vae.encode, pipe.vae.decode
    pipe.vae.encode = lambda x, *a, **kw: original_encode(x.float(), *a, **kw)
    pipe.vae.decode = lambda x, *a, **kw: original_decode(x.float(), *a, **kw)
    original_prepare = pipe.prepare_latents
    pipe.prepare_latents = lambda *a, **kw: original_prepare(*a, **kw).to(dtype=torch.float16)
    # Default torch2 SDPA attention preserves the model operators while avoiding
    # the original run's one-head slicing overhead. Runtime details are recorded.
    pipe.enable_vae_slicing()
    if mode == "resident":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def release_pipeline(pipe):
    """Release resident weights before changing memory placement."""
    import torch
    if pipe is not None:
        pipe.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def generate(pipe, image, strength):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.inference_mode():
        if strength is None:
            x = pipe.image_processor.preprocess(image).to("cuda", dtype=torch.float32)
            latent = pipe.vae.encode(x).latent_dist.mode()
            decoded = pipe.vae.decode(latent).sample
            result = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            pipe.maybe_free_model_hooks()
        else:
            result = pipe(prompt="", image=image, strength=strength, num_inference_steps=20,
                          guidance_scale=1., generator=generator).images[0]
    return result


def run(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    example_prepare()
    covers = selected_covers()
    selection = OUT / "selected_covers.csv"
    if not selection.exists():
        covers.to_csv(selection, index=False)
    elif pd.read_csv(selection, dtype=str, keep_default_na=False).cover_id.tolist() != covers.cover_id.tolist():
        raise ValueError("Frozen regeneration cover selection changed")
    freeze = OUT / "protocol.json"
    protocol = {
        "frozen_at_utc": utc(), "model": str(CACHE), "cached_revision": CACHE.name,
        "conditions": [c for c, _ in CONDITIONS], "scheduler": "DDIM from cached scheduler config",
        "steps": 20, "effective_denoising_steps": {"0.25": 5, "0.50": 10},
        "strengths": [.25, .50], "prompt": "", "guidance_scale": 1., "seed": SEED,
        "batch_size": 1, "working_size": [256, 256],
        "unet_text_dtype": "float16", "vae_dtype": "float32",
        "planned_covers": 408, "planned_outputs": 1224, "source_count": 204,
        "sources_per_generator": 68, "payload": PAYLOAD,
        "source_selection": "Frozen phase3 watermark pairs, default payload, both watermarks, AI only",
        "reuses_previously_examined_sources": True, "new_independent_dataset": False,
        "label_interpretation": "Input-source provenance; outputs are regenerated AI-source covers",
        "no_larger_run_real_controls": True, "exact_zhao_reproduction": False,
        "selected_covers_sha256": digest(selection), "all_covers_sha256": digest(COVERS),
        "checkpoint_sha256": {str(p.relative_to(CACHE)): digest(p) for p in CACHE.rglob("*") if p.is_file()},
        "runtime_mode": args.mode,
        "auto_memory_policy": "Try all models on CUDA; on CUDA out-of-memory, resume with model CPU offload using the same model/operators/seeds",
        "attention": "Default torch2 SDPA attention processor",
        "vae_slicing": True, "driver_sha256": digest(Path(__file__))}
    if not freeze.exists():
        write_json(freeze, protocol)
    else:
        existing = json.loads(freeze.read_text(encoding="utf-8"))
        for key in ("selected_covers_sha256", "all_covers_sha256", "cached_revision", "seed", "payload", "runtime_mode"):
            if existing[key] != protocol[key]:
                raise ValueError(f"Frozen protocol mismatch: {key}")
    manifest = OUT / "manifest.csv"
    rows = pd.read_csv(manifest, keep_default_na=False, dtype=str).to_dict("records") if manifest.exists() else []
    done = {(row["cover_id"], row["condition"]): row for row in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate output keys in regeneration manifest")
    for row in rows:
        if digest(row["path"]) != row["output_file_sha256"]:
            raise ValueError("Completed output changed")
    mode = "resident" if args.mode == "auto" else args.mode
    pipe = None
    try:
        pipe = load_pipeline(mode)
    except torch.cuda.OutOfMemoryError:
        if args.mode != "auto":
            raise
        release_pipeline(pipe)
        mode = "offload"
        pipe = load_pipeline(mode)
    runtime = {"started_at_utc": utc(), "torch": importlib.metadata.version("torch"),
        "diffusers": importlib.metadata.version("diffusers"), "device": torch.cuda.get_device_name(0),
        "requested_mode": args.mode, "mode": mode,
        "attention_processors": sorted({type(p).__name__ for p in pipe.unet.attn_processors.values()}),
        "generator_device": "cpu", "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "driver_sha256": digest(Path(__file__))}
    write_json(OUT / "runtime.json", runtime)
    def generate_with_fallback(image, strength):
        nonlocal pipe, mode
        try:
            return generate(pipe, image, strength)
        except torch.cuda.OutOfMemoryError:
            if args.mode != "auto" or mode != "resident":
                raise
            release_pipeline(pipe)
            pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            mode = "offload"
            pipe = load_pipeline(mode)
            runtime.update({"mode": mode, "fallback_at_utc": utc(),
                "fallback_reason": "CUDA out-of-memory with all models resident"})
            write_json(OUT / "runtime.json", runtime)
            return generate(pipe, image, strength)
    example = ASSETS / "example_diffusion0p50.png"
    if not example.exists():
        started = time.perf_counter()
        result = generate_with_fallback(Image.open(ASSETS / "example_marked.png").convert("RGB"), .50)
        result.save(example)
        write_json(OUT / "example_regeneration.json", {"excluded_from_benchmark": True,
            "input_file_sha256": digest(ASSETS / "example_marked.png"), "output_file_sha256": digest(example),
            "seconds": time.perf_counter()-started, "strength": .50, "seed": SEED, "steps": 20})
        example_prepare()
    count = 0
    for cover in covers.to_dict("records"):
        for condition, strength in CONDITIONS:
            key = (cover["cover_id"], condition)
            if key in done:
                continue
            image = Image.open(cover["path"]).convert("RGB")
            if image.size != (256, 256):
                raise ValueError("Cover size is not256x256")
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            result = generate_with_fallback(image, strength)
            target = DATA / condition / f"{cover['cover_id']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            result.save(target)
            pixels = np.asarray(result)
            if pixels.max() == pixels.min():
                raise ValueError("Degenerate output")
            row = {"cover_id": cover["cover_id"], "source_id": cover["source_id"],
                "watermark_method": cover["watermark_method"], "generator": cover["generator"],
                "payload": PAYLOAD, "input_origin_label": 1, "condition": condition,
                "attack": "vae_roundtrip" if strength is None else "diffusion",
                "severity": 0 if strength is None else strength,
                "strength": "" if strength is None else strength,
                "steps": 0 if strength is None else 20, "seed": SEED, "path": str(target),
                "runtime_mode": mode,
                "input_cover_sha256": cover["output_file_sha256"], "output_file_sha256": digest(target),
                "seconds": time.perf_counter()-started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            rows.append(row)
            pd.DataFrame(rows).to_csv(manifest, index=False)
            done[key] = row
            count += 1
            if len(rows) % 12 == 0 or count == 1:
                print(f"Regeneration {len(rows)}/1224; {condition}; {row['seconds']:.2f}s", flush=True)
            if args.limit and count >= args.limit:
                return
    assert len(rows) == 1224
    write_json(OUT / "complete.json", {"completed_at_utc": utc(), "outputs": len(rows),
        "sources": 204, "covers": 408, "manifest_sha256": digest(manifest),
        "protocol_sha256": digest(freeze), "generated_example_excluded": True})


def analyze(args):
    """Decode matching marks with the official TrustMark API/ECC, source unit."""
    import torch
    from scipy.stats import beta
    import benchmark_watermarks as base
    manifest = OUT / "manifest.csv"
    conditions = pd.read_csv(manifest, keep_default_na=False, dtype=str)
    covers = selected_covers().set_index("cover_id")
    assert len(conditions) == 1224 and not conditions.duplicated(["cover_id", "condition"]).any()
    marks = base.Watermarks(args.device, 4)
    rows, anchors = [], []
    for start in range(0, len(conditions), args.batch_size):
        block = conditions.iloc[start:start+args.batch_size].to_dict("records")
        pixels, tensors = [], []
        for row in block:
            assert digest(row["path"]) == row["output_file_sha256"]
            image = cv2.imread(row["path"], cv2.IMREAD_COLOR)
            pixels.append(image)
            patch = marks.tm.get_the_image_for_processing(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            patch = patch.resize((marks.tm.model_resolution_dec, marks.tm.model_resolution_dec), Image.Resampling.BILINEAR)
            tensors.append(marks.transforms.ToTensor()(patch)*2-1)
        with torch.inference_mode():
            raw = (marks.tm.decoder.decoder(torch.stack(tensors).to(marks.tm.decoder.device))>0).cpu().numpy().astype(np.uint8)
        messages = marks.tm.ecc.decode_bitstream(raw, MODE="binary")
        for i, (row, image) in enumerate(zip(block, pixels)):
            cover = covers.loc[row["cover_id"]]
            dwt = marks.dwt.detect(image, threshold=13/16)
            message, valid, schema = messages[i]
            tm_verified = bool(valid and int(schema)==int(marks.tm.enctyp) and message==marks.expected)
            if len(anchors)<32:
                official = marks.tm.decode(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), MODE="binary", ROTATION=False, DETECTFIRST=False)
                assert bool(official[1]) == bool(valid)
                if valid:
                    assert official[0] == message and int(official[2]) == int(schema)
                anchors.append({"cover_id": row["cover_id"], "condition": row["condition"], "batch_matches_public_decode": True})
            input_image = cv2.imread(cover.path, cv2.IMREAD_COLOR)
            rows.append({**row, "input_origin_label": 1,
                "verified": int(dwt.verified if row["watermark_method"]=="dwt_dct" else tm_verified),
                "dwt_verified": int(dwt.verified), "trustmark_verified": int(tm_verified),
                "dwt_bits": dwt.bits, "dwt_bit_accuracy": dwt.bit_accuracy,
                "trustmark_first16_rawbits": "".join(map(str, raw[i, :16].tolist())),
                "trustmark_raw_codeword_accuracy": float(np.mean(raw[i]==marks.codeword)),
                "trustmark_ECC_valid": int(valid), "trustmark_schema": int(schema),
                "trustmark_full61bit_message_match": int(tm_verified),
                "psnr_vs_input_cover": float(cv2.PSNR(input_image, image)),
                "ssim_vs_input_cover": float(base.structural_similarity(input_image, image, channel_axis=2, data_range=255))})
        if start%128==0:
            print(f"Regeneration decoding {start+len(block)}/1224", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "measurements.csv", index=False)
    measurements = []
    for (condition, method), block in frame.groupby(["condition", "watermark_method"]):
        assert len(block)==204 and block.source_id.nunique()==204
        success = int(block.verified.sum()); n = len(block)
        ci = [float(beta.ppf(.025, success, n-success+1)) if success else 0.,
              float(beta.ppf(.975, success+1, n-success)) if success<n else 1.]
        source_rates = block.sort_values("source_id").verified.to_numpy(dtype=float)
        rng = np.random.default_rng(SEED)
        bootstrap_ci = np.percentile(source_rates[rng.integers(n, size=(2000, n))].mean(axis=1), [2.5, 97.5]).tolist()
        per_collection = []
        for generator, group in block.groupby("generator"):
            assert len(group)==68
            per_collection.append({"generator": generator, "n_sources": len(group),
                "verified": int(group.verified.sum()), "verification_rate": float(group.verified.mean())})
        measurements.append({"condition": condition, "watermark_method": method,
            "detector": method, "generator": "all", "input_origin_label": 1,
            "n_sources": n, "n_payloads": 1, "verified": success, "verification_rate": success/n,
            "n_outputs": n, "matches": success, "expected_payload_match_rate": success/n,
            "rate_ci95_source_bootstrap": bootstrap_ci,
            "rate_ci95_exact_binomial_one_output_per_source": ci,
            "ci95": ci, "mean_psnr": float(block.psnr_vs_input_cover.mean()),
            "mean_psnr_vs_input_cover": float(block.psnr_vs_input_cover.mean()),
            "mean_ssim_vs_input_cover": float(block.ssim_vs_input_cover.mean()),
            "median_psnr": float(block.psnr_vs_input_cover.median()),
            "mean_ssim": float(block.ssim_vs_input_cover.mean()), "per_collection": per_collection})
    write_json(OUT / "summary.json", {"source_count": 204, "covers": 408, "outputs": 1224,
        "matching_decoder_only": True, "default_payload_only": True,
        "real_controls_in_larger_regeneration_run": False,
        "new_independent_dataset": False, "generated_illustration_excluded": True,
        "measurements": measurements, "manifest_sha256": digest(manifest),
        "measurement_file_sha256": digest(OUT / "measurements.csv"), "provenance": marks.provenance(),
        "public_decode_anchors": anchors, "ci": "Clopper-Pearson descriptive source-binomial intervals",
        "analysis_completed_at_utc": utc()})
    print(json.dumps(measurements, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["example_prepare", "run", "analyze"], required=True)
    ap.add_argument("--mode", choices=["auto", "resident", "offload"], default="auto")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.stage == "example_prepare":
        example_prepare()
    elif args.stage == "analyze":
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
