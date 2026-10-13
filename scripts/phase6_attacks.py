"""Published attack ports and development-only quality pilot.

The public interface accepts BGR uint8 images and source-derived seeds. Every
input origin and marking arm receives identical operations. This script never
fits a detector, selects an attack using detector scores, or evaluates a fresh
held-out sample. Final use is controlled by the caller's frozen protocol.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import sys
import time
import typing
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6/attack_development"
ASSETS = OUT / "assets"
WAVES_REVISION = "1477635d306d6f8c77a588e6441bd4ba301a1f7a"
COMPRESSAI_REVISION = "1c9e3eee83a8f4c8162d38b52f78f5b0c40de175"
CHECKPOINT = ASSETS / "bmshj2018-factorized-prior-3-5c6f152b.pth.tar"
RN18_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
SD_CACHE = Path.home() / ".cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
PAYLOAD = "1011010010110100"
SEED = 20261018

# Only public immutable reference files and one explicitly sized model are
# acquired. The full diffusion model remains a separately disclosed prerequisite.
PUBLIC_ASSETS = {
    "waves_embedding.py": (f"https://raw.githubusercontent.com/umd-huang-lab/WAVES/{WAVES_REVISION}/adversarial/embedding.py", "0e94ca4de813920c81a5e35fe4fb294c6aca089dbf6d55557d415dbf864026c0"),
    "waves_regen.py": (f"https://raw.githubusercontent.com/umd-huang-lab/WAVES/{WAVES_REVISION}/regeneration/regen.py", "d77e552a5bd66a708b7fac566050d20a0e2fc7814b575ef29bc759577f34d039"),
    "waves_resnet18.py": (f"https://raw.githubusercontent.com/umd-huang-lab/WAVES/{WAVES_REVISION}/adversarial/feature_extractors/resnet18.py", "71d6bb38446faf94aefed34dd34aa30828d5fa1f8969bae114af55ff4db803e4"),
    "compressai_google.py": (f"https://raw.githubusercontent.com/InterDigitalInc/CompressAI/{COMPRESSAI_REVISION}/compressai/models/google.py", "69caf4979f0957e80e4b95c1dcf3dbf3abb089831c81c5ec98546a46654fe910"),
    "compressai_gdn.py": (f"https://raw.githubusercontent.com/InterDigitalInc/CompressAI/{COMPRESSAI_REVISION}/compressai/layers/gdn.py", "d23a5be004216a2f00222dc4bdd5b060d37ed4b12d465461f72277f8a4c13263"),
    "compressai_parametrizers.py": (f"https://raw.githubusercontent.com/InterDigitalInc/CompressAI/{COMPRESSAI_REVISION}/compressai/ops/parametrizers.py", "2d2589f15624531f7ad381f81e9b68fc594c26f3adcec7db6fde23d6c555cef3"),
    "compressai_entropy_models.py": (f"https://raw.githubusercontent.com/InterDigitalInc/CompressAI/{COMPRESSAI_REVISION}/compressai/entropy_models/entropy_models.py", "02e6729ad34ee3ed67138ef5c703a59fac20f00ff064d4103820dc64cd6fa610"),
    "compressai_image.py": (f"https://raw.githubusercontent.com/InterDigitalInc/CompressAI/{COMPRESSAI_REVISION}/compressai/zoo/image.py", "b5d9a923648db15fcecc775cd041b7131c578a16e2f8056c75370fdaff3e7b27"),
    "bmshj2018-factorized-prior-3-5c6f152b.pth.tar": ("https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-3-5c6f152b.pth.tar", "5c6f152b2597de5e0f9143e8173d8c0529310aec523d07e9e9ced96825621d82"),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def acquire_assets():
    """Reproduce public reference assets; existing files must match exactly."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (url, expected) in PUBLIC_ASSETS.items():
        target = ASSETS / name
        if not target.resolve().is_relative_to(OUT.resolve()):
            raise ValueError("Attack asset escaped the development directory")
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "watermark-processing-controls-reproduction/1.0"})
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
            if digest(temporary) != expected:
                raise ValueError(f"Downloaded public attack asset mismatch: {name}")
            temporary.replace(target)
        if digest(target) != expected:
            raise ValueError(f"Existing public attack asset mismatch: {name}")
        records.append({"name": name, "url": url, "sha256": expected, "bytes": target.stat().st_size})
    notice = "\n".join((ASSETS / "compressai_google.py").read_text(encoding="utf-8").splitlines()[:26]) + "\n"
    (ASSETS / "COMPRESSAI-NOTICE.txt").write_text(notice, encoding="utf-8")
    write_json(OUT / "public_assets.json", {"verified_at_utc": utc(), "waves_revision": WAVES_REVISION,
        "compressai_revision": COMPRESSAI_REVISION, "assets": records,
        "diffusion_prerequisite": str(SD_CACHE), "resnet_prerequisite": str(RN18_CHECKPOINT)})
    print("Verified public attack reference assets:", len(records), flush=True)


def source_seed(source_id, replicate=0):
    """Use the same source seed across unmarked/marked and real/AI arms."""
    return int(hashlib.sha256(f"{SEED}|{source_id}|{replicate}".encode()).hexdigest()[:8], 16)


def validate_image(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR uint8 image")


def image_tensor(image, device):
    import torch
    validate_image(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255


def tensor_image(tensor):
    """Match torchvision save_image's add(0.5), clamp, uint8 quantization."""
    rgb = tensor.detach().squeeze(0).permute(1, 2, 0).mul(255).add(.5).clamp(0, 255).to("cpu").to(dtype=__import__("torch").uint8).numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class WavesResNetPGD:
    """WAVES AdvEmbB-RN18 objective, update, preprocessing, and 200 steps.

    Port adaptations: explicit per-source RNG seed; one image per call; local
    weights; CPU-capable normalization buffers. The current torchvision resize
    implementation is pinned in runtime metadata, rather than assumed identical
    to the paper's historical software environment.
    """
    def __init__(self, device="cuda", steps=200):
        import torch
        from torchvision.models import resnet18
        from torchvision.transforms import functional as TF
        if not digest(RN18_CHECKPOINT).startswith("f37072fd"):
            raise ValueError("Official torchvision RN18 checkpoint hash-prefix mismatch")
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(RN18_CHECKPOINT, map_location="cpu", weights_only=True))
        self.features = torch.nn.Sequential(*list(net.children())[:-1]).eval().to(device)
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        self.TF, self.device, self.steps = TF, device, steps

    def embedding(self, images):
        return self.features((self.TF.resize(images, [224, 224]) - self.mean) / self.std)

    def tensor_attack(self, images, strength, seed):
        import torch
        if strength not in (2, 4, 6, 8):
            raise ValueError("Published WAVES strengths are 2, 4, 6, 8")
        eps, alpha = strength / 255, .05 * strength / 255
        # fork_rng prevents the attack from changing the caller's random stream.
        cuda_devices = [torch.device(self.device).index or 0] if str(self.device).startswith("cuda") else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            images = images.clone().detach().to(self.device)
            original = self.embedding(images).detach()
            adv = (images + torch.empty_like(images).uniform_(-eps, eps)).clamp(0, 1).detach()
            for _ in range(self.steps):
                adv.requires_grad_(True)
                loss = torch.nn.functional.mse_loss(self.embedding(adv), original)
                gradient = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv.detach() + alpha * gradient.sign()
                delta = (adv - images).clamp(-eps, eps)
                adv = (images + delta).clamp(0, 1).detach()
        return adv

    def apply(self, image, strength=4, seed=0):
        return tensor_image(self.tensor_attack(image_tensor(image, self.device), strength, seed))

    def close(self):
        self.features.to("cpu")


class WavesFactorizedReconstruction:
    """WAVES Regen-VAE bmshj2018-factorized q3 inference-only port.

    The official checkpoint's analysis/synthesis transforms, trained GDN
    parameters, and entropy-bottleneck medians are used. In eval mode the
    bottleneck's image reconstruction is exactly round(y-median)+median. The
    likelihood/CDF calculation and entropy coding do not affect x_hat and are
    omitted to avoid CompressAI's unavailable Windows entropy-code extension.
    A round trip through an entropy bitstream is not claimed.

    WAVES first resizes to512. Its native512 output is available; the caller's
   256 experiment uses a separately labeled INTER_AREA output restoration.
    """
    def __init__(self, device="cuda", checkpoint=CHECKPOINT):
        import torch
        from torch import nn
        if not digest(checkpoint).startswith("5c6f152b"):
            raise ValueError("Official CompressAI checkpoint hash-prefix mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.state, self.device = state, device

        class GDNInference(nn.Module):
            def __init__(self, prefix, inverse):
                super().__init__()
                self.inverse = inverse
                beta = state[prefix + ".beta"]
                gamma = state[prefix + ".gamma"]
                # These trained parametrizer buffers are in the official state.
                beta = torch.maximum(beta, state[prefix + ".beta_reparam.lower_bound.bound"]).square() - state[prefix + ".beta_reparam.pedestal"]
                gamma = torch.maximum(gamma, state[prefix + ".gamma_reparam.lower_bound.bound"]).square() - state[prefix + ".gamma_reparam.pedestal"]
                self.register_buffer("beta", beta)
                self.register_buffer("gamma", gamma.reshape(gamma.shape[0], gamma.shape[1], 1, 1))

            def forward(self, x):
                norm = torch.nn.functional.conv2d(x.square(), self.gamma, self.beta)
                return x * (norm.sqrt() if self.inverse else norm.rsqrt())

        def transform(name, inverse):
            modules = []
            for idx in range(7):
                prefix = f"{name}.{idx}"
                if idx % 2:
                    modules.append(GDNInference(prefix, inverse))
                else:
                    w, b = state[prefix + ".weight"], state[prefix + ".bias"]
                    layer = nn.ConvTranspose2d(w.shape[0], w.shape[1], 5, stride=2, padding=2, output_padding=1) if inverse else nn.Conv2d(w.shape[1], w.shape[0], 5, stride=2, padding=2)
                    layer.load_state_dict({"weight": w, "bias": b})
                    modules.append(layer)
            return nn.Sequential(*modules).eval().to(device)

        self.encoder, self.decoder = transform("g_a", False), transform("g_s", True)
        self.medians = state["entropy_bottleneck.quantiles"][:, :, 1:2].reshape(1, -1, 1, 1).to(device)

    def tensor_reconstruct(self, x):
        import torch
        with torch.inference_mode():
            y = self.encoder(x)
            return self.decoder((y - self.medians).round() + self.medians).clamp(0, 1)

    def apply_native(self, image, seed=0):
        import torch
        from torchvision.transforms import functional as TF
        validate_image(image)
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((512, 512))
        x = TF.to_tensor(rgb).unsqueeze(0).to(self.device)
        # ToPILImage truncation is the official WAVES reconstruction path.
        with torch.inference_mode():
            native = TF.to_pil_image(self.tensor_reconstruct(x).squeeze(0).cpu())
        return cv2.cvtColor(np.asarray(native), cv2.COLOR_RGB2BGR)

    def apply(self, image, seed=0):
        native = self.apply_native(image, seed)
        return cv2.resize(native, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    def close(self):
        self.encoder.to("cpu")
        self.decoder.to("cpu")


class StableDiffusionRegeneration:
    """Cached RV5.1/SD1.5 img2img; adaptation, not exact WAVES Diff attack."""
    def __init__(self, device="cuda", mode="resident"):
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        self.device = device
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(SD_CACHE), torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.float()
        encode, decode = self.pipe.vae.encode, self.pipe.vae.decode
        self.pipe.vae.encode = lambda x, *a, **kw: encode(x.float(), *a, **kw)
        self.pipe.vae.decode = lambda x, *a, **kw: decode(x.float(), *a, **kw)
        prepare = self.pipe.prepare_latents
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe.prepare_latents = lambda *a, **kw: prepare(*a, **kw).to(dtype=dtype)
        self.pipe.enable_vae_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        if mode == "offload" and device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def apply(self, image, strength=.1, seed=0):
        import torch
        validate_image(image)
        if not .05 <= strength <= 1:
            raise ValueError("DDIM20 needs strength>=.05 for at least one effective step")
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            output = self.pipe(prompt="", image=rgb, strength=strength, num_inference_steps=20, guidance_scale=1,
                generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
        pixels = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
        if pixels.shape != image.shape:
            raise ValueError("Diffusion altered image dimensions")
        return pixels

    def close(self):
        self.pipe.to("cpu")


def release(attack):
    import torch
    attack.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quality(reference, image):
    return {"PSNR_dB": float(cv2.PSNR(reference, image)),
            "SSIM": float(structural_similarity(reference, image, channel_axis=2, data_range=255))}


def residual_budget(reference, raw_attack, target_psnr=35., tolerance=.1, seed=0):
    """Explicit pixel-budget adaptation; no decoder or origin score is used.

    Scale an existing attack residual toward the input to reach a fixed PSNR.
    If the raw attack is less distorted than the requested budget, return it
    unchanged and disclose that the target is unreachable. This function never
    strengthens a published attack beyond its original perturbation budget.
    Fixed pixel PSNR does not imply matched perceptual/semantic quality.
    """
    validate_image(reference)
    validate_image(raw_attack)
    if reference.shape != raw_attack.shape:
        raise ValueError("Pixel budget requires same image dimensions")
    raw_psnr = float(cv2.PSNR(reference, raw_attack))
    if raw_psnr > target_psnr:
        return raw_attack.copy(), {"gain": 1., "target_PSNR_dB": target_psnr, "actual_PSNR_dB": raw_psnr,
            "target_reachable": False, "within_tolerance": abs(raw_psnr-target_psnr) <= tolerance,
            "raw_PSNR_dB": raw_psnr, "adaptation": "fixed_PSNR_residual_blend", "rounding": "source_seeded_stochastic_v1", "rounding_seed": seed}
    delta = raw_attack.astype(np.float64)-reference.astype(np.float64)
    # Fixed per-source uniforms couple all marking arms and produce monotone
    # residual magnitudes as gain increases. No watermark or detector outcome
    # enters gain selection. Stochastic rounding avoids synchronous integer
    # jumps of thousands of saturated PGD pixels under nearest rounding.
    uniform = np.random.default_rng(seed).random(reference.shape)
    def at(gain):
        return np.clip(np.floor(reference.astype(np.float64)+gain*delta+uniform), 0, 255).astype(np.uint8)
    lo, hi = 0., 1.
    candidates = []
    for _ in range(40):
        gain = (lo+hi)/2
        image = at(gain)
        value = float(cv2.PSNR(reference, image))
        candidates.append((abs(value-target_psnr), gain, image, value))
        if value > target_psnr:
            lo = gain
        else:
            hi = gain
    _, gain, image, value = min(candidates, key=lambda r: r[0])
    return image, {"gain": gain, "target_PSNR_dB": target_psnr, "actual_PSNR_dB": value,
        "target_reachable": True, "within_tolerance": abs(value-target_psnr) <= tolerance,
        "raw_PSNR_dB": raw_psnr, "adaptation": "fixed_PSNR_residual_blend", "rounding": "source_seeded_stochastic_v1", "rounding_seed": seed}


def budget_pilot():
    """Pixel-only quality anchors derived from the existing raw pilot outputs."""
    inputs = pd.read_csv(OUT / "pilot_outputs.csv", keep_default_na=False)
    raw = inputs.loc[((inputs.attack == "waves_resnet_pgd") & (inputs.strength == 8)) |
        (inputs.attack == "waves_factorized_reconstruction") |
        ((inputs.attack == "sd15_ddim_regeneration") & (inputs.strength == .05))]
    protocol = OUT / "budget_stochastic_protocol.json"
    obj = {"frozen_at_utc": utc(), "development_only": True, "detector_or_watermark_scores_observed": False,
        "raw_manifest_sha256": digest(OUT / "pilot_outputs.csv"), "targets_PSNR_dB": [35., 38., 40.],
        "maximum_error_dB": .1, "raw_attacks": {"PGD": "published_strength8", "compression": "published_q3", "diffusion": "minimum_strength.05"},
        "rounding": "source_seeded_stochastic_v1", "rounding_seed": "source_seed(source_id)",
        "adaptation": "Rescale pixel residual toward input only. Not the complete original reconstruction attack; more watermark may survive.",
        "quality_limit": "Shared PSNR constrains pixels, not semantic or perceptual equivalence.",
        "driver_sha256": digest(Path(__file__))}
    if protocol.exists():
        existing = json.loads(protocol.read_text())
        if existing["raw_manifest_sha256"] != obj["raw_manifest_sha256"]:
            raise ValueError("Raw quality pilot manifest changed")
    else:
        write_json(protocol, obj)
    rows = []
    for row in raw.to_dict("records"):
        if digest(row["path"]) != row["output_file_sha256"]:
            raise ValueError("Raw pilot output hash mismatch")
        reference, candidate = cv2.imread(row["input_path"]), cv2.imread(row["path"])
        for target in [35., 38., 40.]:
            image, settings = residual_budget(reference, candidate, target, seed=source_seed(row["source_id"]))
            name = hashlib.sha256(f"{row['output_file_sha256']}|stochastic_budget_v1|{target}".encode()).hexdigest()[:24]
            output = OUT / "budget_stochastic_images" / f"{name}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and not np.array_equal(cv2.imread(str(output)), image):
                raise FileExistsError(output)
            assert cv2.imwrite(str(output), image)
            rows.append({"cover_id": row["cover_id"], "source_id": row["source_id"], "attack": row["attack"],
                "watermark_method": row["watermark_method"], "label": row["label"], "collection": row["collection"],
                "raw_path": row["path"], "path": str(output), "output_file_sha256": digest(output), **settings, **quality(reference, image)})
    target = OUT / "budget_stochastic_outputs.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    summary = {"completed_at_utc": utc(), "development_only": True, "images": len(rows),
        "maximum_PSNR_error_dB": max(abs(r["actual_PSNR_dB"]-r["target_PSNR_dB"]) for r in rows),
        "all_targets_reached": all(r["target_reachable"] and r["within_tolerance"] for r in rows),
        "outputs_sha256": digest(target), "no_decoder_or_detector_scores": True}
    write_json(OUT / "budget_stochastic_summary.json", summary)
    print(json.dumps(summary), flush=True)


def prepare_pilot():
    """Use already examined Phase5 covers; never a new test selection."""
    frame = pd.read_csv(ROOT / "outputs/phase5/fair_watermarks/covers.csv", keep_default_na=False)
    ai = frame.loc[(frame.role == "test") & (frame.label == 1)]
    source = sorted(ai.source_id.unique(), key=lambda s: hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest())[0]
    rows = ai.loc[(ai.source_id == source) & ((ai.payload == PAYLOAD) | (ai.watermark_method == "unmarked"))]
    real = frame.loc[(frame.role == "test") & (frame.label == 0)]
    for collection in ["coco", "openimages"]:
        candidates = real.loc[real.collection == collection]
        if len(candidates):
            selected = sorted(candidates.source_id.unique(), key=lambda s: hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest())[0]
            rows = pd.concat([rows, candidates.loc[candidates.source_id == selected].head(1)], ignore_index=True)
    rows = rows.sort_values(["source_id", "watermark_method"]).reset_index(drop=True)
    target = OUT / "pilot_inputs.csv"
    if target.exists():
        saved = pd.read_csv(target, keep_default_na=False)
        if saved.cover_id.tolist() != rows.cover_id.tolist():
            raise ValueError("Development pilot selection changed")
        return saved
    for row in rows.to_dict("records"):
        if digest(row["path"]) != row["output_file_sha256"]:
            raise ValueError("Development cover hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(target, index=False)
    references = {p.name: digest(p) for p in ASSETS.glob("*") if p.is_file()}
    protocol = {"frozen_at_utc": utc(), "development_only": True, "already_examined_sources": True,
        "watermark_or_detector_scores_used_to_select_attacks": False,
        "input_manifest_sha256": digest(target), "source_count": int(rows.source_id.nunique()), "covers": len(rows),
        "waves_commit": WAVES_REVISION, "asset_sha256": references,
        "seed_rule": "first32 SHA256(20261018|source_id|replicate)",
        "attacks": {"waves_resnet_pgd": {"strengths": [2, 4, 8], "steps": 200, "eps": "strength/255", "alpha": ".05*strength/255"},
                    "waves_factorized_reconstruction": {"quality": 3, "metric": "mse", "native_size": [512, 512], "restore": "INTER_AREA to input size"},
                    "sd15_ddim_regeneration": {"strengths": [.05, .1, .2], "steps": 20, "prompt": "", "guidance": 1}},
        "quality_selection": "Report full PSNR/SSIM curves; quality bins will be fixed on development only, before fresh inference. No decoder performance in this pilot.",
        "generator_families": "One diffusion generator family; learned-compression reconstruction is a distinct second regeneration mechanism, not another diffusion family.",
        "driver_sha256_at_prepare": digest(Path(__file__))}
    write_json(OUT / "pilot_protocol.json", protocol)
    return rows


def parity(args):
    """Numerical implementation check against pinned official PGD class."""
    import torch
    torch.set_num_threads(args.threads)
    attack = WavesResNetPGD(args.device, steps=8)
    tree = ast.parse((ASSETS / "waves_embedding.py").read_text(encoding="utf-8"))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WarmupPGDEmbedding")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[original], type_ignores=[]), "pinned_waves_pgd_class", "exec"), namespace)

    class Adapter(torch.nn.Module):
        def forward(self, x):
            return attack.embedding(x)

    data = np.random.default_rng(123).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    x = image_tensor(data, args.device)
    seed = source_seed("development_parity")
    model = Adapter().to(args.device)
    reference = namespace["WarmupPGDEmbedding"](model, args.device, eps=4/255, alpha=.05*4/255, steps=8)
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    expected = reference.forward(x)
    actual = attack.tensor_attack(x, 4, seed)
    maximum = float((expected - actual).abs().max().item())
    if maximum != 0:
        raise ValueError(f"Published PGD parity mismatch: {maximum}")
    result = {"checked_at_utc": utc(), "reference": "Pinned official WAVES WarmupPGDEmbedding class executed unchanged",
        "steps": 8, "production_steps": 200, "maximum_tensor_absolute_error": maximum,
        "uint8_output_identical": bool(np.array_equal(tensor_image(expected), tensor_image(actual))),
        "reference_sha256": digest(ASSETS / "waves_embedding.py"), "device": args.device}
    write_json(OUT / "pgd_parity.json", result)
    release(attack)
    print(json.dumps(result), flush=True)


def reconstruction_parity(args):
    """Compare the port to official PyTorch GDN/quantizer/FactorizedPrior code.

    Native entropy coding is not used by WAVES forward reconstruction. The
    reference's unused coder object is replaced with a stub, while its complete
    EntropyBottleneck.forward (including likelihoods) executes unchanged.
    """
    import torch
    import warnings
    from torch import nn
    torch.set_num_threads(args.threads)

    class LowerBound(nn.Module):
        def __init__(self, bound):
            super().__init__()
            self.register_buffer("bound", torch.Tensor([bound]))
        def forward(self, x):
            return torch.maximum(x, self.bound)

    class CoderStub:
        def __init__(self, name):
            self.name = name

    namespace = {"torch": torch, "nn": nn, "F": torch.nn.functional, "np": np,
        "Tensor": torch.Tensor, "LowerBound": LowerBound, "warnings": warnings,
        "_EntropyCoder": CoderStub, "default_entropy_coder": lambda: "unused",
        "_forward": lambda *a, **kw: None, **vars(typing)}
    def classes(path, names):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        for node in selected:
            node.decorator_list = []
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    classes(ASSETS / "compressai_parametrizers.py", ["NonNegativeParametrizer"])
    classes(ASSETS / "compressai_gdn.py", ["GDN"])
    classes(ASSETS / "compressai_entropy_models.py", ["EntropyModel", "EntropyBottleneck"])
    namespace.update({"CompressionModel": nn.Module,
        "conv": lambda in_channels, out_channels, kernel_size=5, stride=2: nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2),
        "deconv": lambda in_channels, out_channels, kernel_size=5, stride=2: nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, output_padding=stride-1)})
    classes(ASSETS / "compressai_google.py", ["FactorizedPrior"])
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    # Published checkpoint predates the modern public ParameterList names.
    mapped = {k.replace("entropy_bottleneck._matrices.", "entropy_bottleneck.matrices.")
                .replace("entropy_bottleneck._biases.", "entropy_bottleneck.biases.")
                .replace("entropy_bottleneck._factors.", "entropy_bottleneck.factors."): v for k, v in state.items()}
    reference = namespace["FactorizedPrior"](state["g_a.0.weight"].shape[0], state["g_a.6.weight"].shape[0])
    for key in ["_offset", "_quantized_cdf", "_cdf_length"]:
        setattr(reference.entropy_bottleneck, key, state[f"entropy_bottleneck.{key}"].clone())
    reference.load_state_dict(mapped, strict=True)
    reference.eval().to(args.device)
    port = WavesFactorizedReconstruction(args.device)
    image = np.random.default_rng(987).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    x = image_tensor(image, args.device)
    with torch.inference_mode():
        expected = reference(x)["x_hat"].clamp(0, 1)
        actual = port.tensor_reconstruct(x)
    error = float((expected-actual).abs().max().item())
    if error != 0:
        raise ValueError(f"Official reconstruction forward mismatch: {error}")
    result = {"checked_at_utc": utc(), "maximum_tensor_absolute_error": error, "device": args.device,
        "input_size": [256, 256], "official_forward_and_likelihoods_executed": True,
        "reference_environment_adaptations": ["Unused entropy coder stub", "Inference-only LowerBound torch.maximum", "Historical checkpoint ParameterList key migration"],
        "reference_source_sha256": {name: digest(ASSETS / name) for name in ["compressai_google.py", "compressai_gdn.py", "compressai_parametrizers.py", "compressai_entropy_models.py"]},
        "checkpoint_sha256": digest(CHECKPOINT), "bitstream_entropy_coding_tested": False}
    write_json(OUT / "reconstruction_parity.json", result)
    release(port)
    print(json.dumps(result), flush=True)


def pilot(args):
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    covers = prepare_pilot()
    target = OUT / "pilot_outputs.csv"
    rows = pd.read_csv(target, keep_default_na=False).to_dict("records") if target.exists() else []
    done = {(r["cover_id"], r["attack"], float(r["strength"])): r for r in rows}
    for row in rows:
        if digest(row["path"]) != row["output_file_sha256"]:
            raise ValueError("Saved development output changed")
    factories = {"waves_resnet_pgd": lambda: WavesResNetPGD(args.device),
        "waves_factorized_reconstruction": lambda: WavesFactorizedReconstruction(args.device),
        "sd15_ddim_regeneration": lambda: StableDiffusionRegeneration(args.device, args.mode)}
    strengths = {"waves_resnet_pgd": [2, 4, 8], "waves_factorized_reconstruction": [3], "sd15_ddim_regeneration": [.05, .1, .2]}
    for name in args.attacks.split(","):
        if name not in factories:
            raise ValueError(f"Unknown attack: {name}")
        attack = factories[name]()
        for row in covers.to_dict("records"):
            reference = cv2.imread(row["path"])
            for strength in strengths[name]:
                key = (row["cover_id"], name, float(strength))
                if key in done:
                    continue
                seed = source_seed(row["source_id"])
                before = time.perf_counter()
                kwargs = {} if name == "waves_factorized_reconstruction" else {"strength": strength}
                result = attack.apply(reference, seed=seed, **kwargs)
                output_id = hashlib.sha256(f"{key}|{seed}".encode()).hexdigest()[:24]
                output = OUT / "images" / f"{output_id}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists() and not np.array_equal(cv2.imread(str(output)), result):
                    raise FileExistsError(output)
                assert cv2.imwrite(str(output), result)
                saved = {"cover_id": row["cover_id"], "source_id": row["source_id"], "collection": row["collection"],
                    "label": row["label"], "watermark_method": row["watermark_method"], "payload": row["payload"],
                    "attack": name, "strength": strength, "seed": seed, "input_path": row["path"],
                    "input_file_sha256": row["output_file_sha256"], "path": str(output), "output_file_sha256": digest(output),
                    "seconds": time.perf_counter()-before, **quality(reference, result)}
                rows.append(saved)
                pd.DataFrame(rows).to_csv(target, index=False)
                print(f"Development {name} s{strength} {row['watermark_method']}: {saved['seconds']:.2f}s, PSNR{saved['PSNR_dB']:.2f}, SSIM{saved['SSIM']:.3f}", flush=True)
        release(attack)
        del attack
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUT / "pilot_summary.json", {"completed_at_utc": utc(), "development_only": True,
        "no_watermark_or_origin_detector_scoring": True, "outputs_manifest_sha256": digest(target),
        "rows": len(rows), "source_count": int(covers.source_id.nunique()),
        "by_attack_strength": pd.DataFrame(rows).groupby(["attack", "strength"], dropna=False).agg(
            images=("cover_id", "size"), mean_PSNR_dB=("PSNR_dB", "mean"), mean_SSIM=("SSIM", "mean"), mean_seconds=("seconds", "mean")).reset_index().to_dict("records"),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "diffusers", "numpy", "pillow"]},
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "driver_sha256": digest(Path(__file__))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["assets", "prepare", "parity", "reconstruction_parity", "pilot", "budget"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mode", choices=["resident", "offload"], default="resident")
    parser.add_argument("--attacks", default="waves_resnet_pgd,waves_factorized_reconstruction,sd15_ddim_regeneration")
    args = parser.parse_args()
    if args.stage == "assets":
        acquire_assets()
    elif args.stage == "prepare":
        print(prepare_pilot().to_json(orient="records"))
    elif args.stage == "parity":
        parity(args)
    elif args.stage == "budget":
        budget_pilot()
    elif args.stage == "reconstruction_parity":
        reconstruction_parity(args)
    else:
        pilot(args)


if __name__ == "__main__":
    main()
