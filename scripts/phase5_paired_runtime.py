"""Execute the locked caption-pair generation and one frozen evaluation.

Preparation, original detector heads and earlier measurements remain read-only.
The runtime supplement adds independently calibrated heads before any scores.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from phase5_paired_content import ROOT, OUT, DATA, CACHE, digest, utc, write_json, pixel_digest
from evaluate_encoding_controls import checkpoint_defaults, load_backbone, infer, cached_features, original_caches
from phase4_domain_generalization import systems
from phase4_fresh_replication import weighted_auc_draws, rate_draws
from aiwstudy.attacks import apply_attack
from aiwstudy.watermark import DwtDctWatermark

CAL = ROOT / "outputs/phase5/calibration"
ADAPTER = ROOT / ".tools/extension-assets/lcm/pytorch_lora_weights.safetensors"
PAYLOAD = "1011010010110100"
cv2.setNumThreads(1)


def verify_lock(obj):
    for name, value in obj["input_sha256"].items():
        if digest(ROOT / name) != value:
            raise ValueError(f"Locked paired runtime input changed: {name}")


def lock():
    target = OUT / "runtime_protocol.json"
    if target.exists():
        saved = json.loads(target.read_text())
        verify_lock(saved)
        return saved
    if (OUT / "generation_complete.json").exists() or (OUT / "predictions.csv").exists():
        raise ValueError("Runtime/head protocol must precede generation and scoring")
    selection = json.loads((OUT / "selection_complete.json").read_text())
    assert selection["pairs"] == 200 and selection["source_pairs_sha256"] == digest(OUT / "source_pairs.csv")
    assets = json.loads((OUT / "generation_assets_lock.json").read_text())
    assert digest(ADAPTER) == assets["adapter_sha256"]
    paths = [OUT / name for name in ["protocol.json", "generation_assets_lock.json", "selection_complete.json", "source_pairs.csv"]]
    paths += [CAL / name for name in ["protocol.json", "heads_locked.json", "thresholds_locked.json"]]
    paths += [CAL / "sd_family_excluded" / name for name in ["protocol.json", "heads_locked.json", "thresholds_locked.json"]]
    heads = {}
    for backbone in ["resnet", "clip"]:
        heads[f"joint_{backbone}"] = [ROOT / "outputs/phase3/controls/split_seeds/20260920" / f"{backbone}_full.npz", CAL / "20260920" / f"{backbone}_fusion.npz"]
        heads[f"SD_excluded_{backbone}"] = [CAL / "sd_family_excluded" / f"{backbone}_{kind}.npz" for kind in ["probe", "fusion"]]
        paths += sum([heads[f"joint_{backbone}"], heads[f"SD_excluded_{backbone}"]], [])
    saved = {"locked_at_utc": utc(), "generation_or_detector_scores_observed": False,
             "primary": "Identity unmarked origin AUC of the independently calibrated joint probes; additional SD-family-excluded probes report restricted DALL-E-only synthetic fitting.",
             "secondary": "JPEG50 and lowres32; separately calibrated probe/fusion/gate decisions under all three fixed policies. Caption pairs are the bootstrap unit.",
             "legacy_protocol": "Retained unchanged. This prescoring supplement makes independently calibrated Phase5 heads the primary comparison; previously frozen original heads remain archived.",
             "test_threshold_tuning": False, "generation_assets_sha256": digest(OUT / "generation_assets_lock.json"),
             "input_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
             "heads": {k: [p.relative_to(ROOT).as_posix() for p in v] for k, v in heads.items()},
             "bootstrap_replicates": 2000, "bootstrap_seed": 20261012,
             "runtime_driver_sha256": digest(Path(__file__))}
    write_json(target, saved)
    return saved


def load_pipeline(mode):
    import torch
    from diffusers import StableDiffusionPipeline, LCMScheduler
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    pipe = StableDiffusionPipeline.from_pretrained(str(CACHE), torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(ADAPTER.parent), weight_name=ADAPTER.name, local_files_only=True)
    pipe.fuse_lora(lora_scale=1.0)
    pipe.vae.float()
    original_decode = pipe.vae.decode
    pipe.vae.decode = lambda x, *a, **kw: original_decode(x.float(), *a, **kw)
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    if mode == "resident":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def generate():
    import torch
    protocol = lock()
    if (OUT / "generation_complete.json").exists():
        saved = json.loads((OUT / "generation_complete.json").read_text())
        assert saved["generated_sources_sha256"] == digest(OUT / "generated_sources.csv")
        print("Reused completed locked generation", flush=True)
        return
    pairs = pd.read_csv(OUT / "source_pairs.csv", keep_default_na=False, dtype={"public_coco_image_id": str})
    started_path = OUT / "generation_started.json"
    if not started_path.exists():
        write_json(started_path, {"started_at_utc": utc(), "runtime_protocol_sha256": digest(OUT / "runtime_protocol.json"),
                                 "runtime_driver_sha256": digest(Path(__file__)), "pairs": 200, "scores_observed": False})
    target = OUT / "generated_sources.csv"
    rows = pd.read_csv(target, keep_default_na=False).to_dict("records") if target.exists() else []
    existing = {r["pair_id"]: r for r in rows}
    for r in rows:
        assert digest(Path(r["ai_native_path"])) == r["ai_native_file_sha256"]
        assert digest(Path(r["ai_path"])) == r["ai_file_sha256"]
    mode = "resident"
    pipe = load_pipeline(mode)
    total = time.perf_counter()
    for pair in pairs.to_dict("records"):
        if pair["pair_id"] in existing:
            continue
        before = time.perf_counter()
        while True:
            try:
                rng = torch.Generator(device="cpu").manual_seed(int(pair["generation_seed"]))
                with torch.inference_mode():
                    image = pipe(prompt=pair["prompt"], negative_prompt="", num_inference_steps=8,
                                 guidance_scale=1., width=512, height=512, generator=rng).images[0]
                break
            except torch.cuda.OutOfMemoryError:
                if mode == "offload":
                    raise
                pipe.to("cpu")
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                mode = "offload"
                pipe = load_pipeline(mode)
                print("CUDA memory fallback: same image seed and pipeline with model CPU offload", flush=True)
        native = DATA / "generated_native" / f"{pair['ai_source_id']}.png"
        base = DATA / "generated_base" / f"{pair['ai_source_id']}.png"
        native.parent.mkdir(parents=True, exist_ok=True)
        base.parent.mkdir(parents=True, exist_ok=True)
        image.save(native)
        pixels = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        pixels = cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        standard = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        assert cv2.imwrite(str(base), standard)
        rows.append({"pair_id": pair["pair_id"], "ai_source_id": pair["ai_source_id"],
                     "ai_native_path": str(native), "ai_path": str(base), "generation_seed": int(pair["generation_seed"]),
                     "ai_native_file_sha256": digest(native), "ai_file_sha256": digest(base),
                     "ai_standardized_pixel_sha256": pixel_digest(standard), "runtime_mode": mode,
                     "seconds": time.perf_counter() - before})
        pd.DataFrame(rows).to_csv(target, index=False)
        if len(rows) % 5 == 0 or len(rows) == 1:
            print(f"Generated fixed caption pair {len(rows)}/200; {rows[-1]['seconds']:.2f}s", flush=True)
    assert len(rows) == 200 and set(r["pair_id"] for r in rows) == set(pairs.pair_id)
    write_json(OUT / "generation_complete.json", {"completed_at_utc": utc(), "pairs": len(rows),
               "generated_sources_sha256": digest(target), "runtime_protocol_sha256": digest(OUT / "runtime_protocol.json"),
               "total_runtime_seconds": time.perf_counter() - total, "candidate_resampling": False,
               "scores_observed_during_generation": False, "runtime_modes": {str(k): int(v) for k, v in pd.Series([r["runtime_mode"] for r in rows]).value_counts().items()},
               "packages": {p: importlib.metadata.version(p) for p in ["torch", "diffusers", "transformers", "peft", "accelerate"]}})
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def derive():
    lock()
    target = OUT / "manifest.csv"
    if target.exists():
        saved = json.loads((OUT / "derivation_complete.json").read_text())
        assert saved["manifest_sha256"] == digest(target)
        return
    gen = json.loads((OUT / "generation_complete.json").read_text())
    assert gen["generated_sources_sha256"] == digest(OUT / "generated_sources.csv")
    pairs = pd.read_csv(OUT / "source_pairs.csv", keep_default_na=False)
    generated = pd.read_csv(OUT / "generated_sources.csv", keep_default_na=False).set_index("pair_id")
    marker = DwtDctWatermark(PAYLOAD, 12.)
    rows = []
    for pair in pairs.to_dict("records"):
        ai = generated.loc[pair["pair_id"]]
        for label, source, path, expected in [(0, pair["real_source_id"], pair["real_path"], pair["standardized_file_sha256"]),
                                               (1, pair["ai_source_id"], ai.ai_path, ai.ai_file_sha256)]:
            assert digest(Path(path)) == expected
            cover = cv2.imread(path)
            for attack, severity in [("identity", 0), ("jpeg", 2), ("lowres", 2)]:
                pixels = apply_attack(cover, attack, severity)
                iid = __import__("hashlib").sha256(f"{source}|{attack}|{severity}".encode()).hexdigest()[:24]
                output = DATA / "derived" / f"{iid}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(output), pixels)
                rows.append({"image_id": iid, "pair_id": pair["pair_id"], "source_id": source, "label": label,
                             "collection": "coco" if label == 0 else "RV_LCM_SD15", "attack": attack,
                             "severity": severity, "mark": "unmarked", "path": str(output),
                             "active_score": marker.detect(pixels).bit_accuracy, "output_file_sha256": digest(output)})
    frame = pd.DataFrame(rows)
    assert len(frame) == 1200 and frame.pair_id.nunique() == 200 and frame.source_id.nunique() == 400
    frame.to_csv(target, index=False)
    write_json(OUT / "derivation_complete.json", {"completed_at_utc": utc(), "copies": 1200,
               "manifest_sha256": digest(target), "detector_inference_performed": False})


def extract():
    import torch
    torch.set_num_threads(2)
    protocol = lock()
    frame = pd.read_csv(OUT / "manifest.csv", keep_default_na=False)
    _, checkpoint = checkpoint_defaults()
    anchors = pd.read_csv(ROOT / "outputs/conference_clean/manifest_derived.csv", keep_default_na=False)
    anchors = anchors.loc[anchors.group.isin(["R-C", "S-U"])].groupby("generator").head(1)
    for r in frame.to_dict("records"):
        assert digest(Path(r["path"])) == r["output_file_sha256"]
    for name in ["resnet", "clip"]:
        target = OUT / f"{name}_embeddings.npz"
        if target.exists():
            with np.load(target, allow_pickle=True) as z:
                assert str(z["manifest_sha256"]) == digest(OUT / "manifest.csv") and np.array_equal(z["image_id"].astype(str), frame.image_id)
            continue
        model, transform, forward, preproc, identity = load_backbone(name, "cuda", checkpoint)
        current = infer(anchors, transform, forward, "cuda", 8).astype(np.float64)
        old = cached_features(original_caches()[:1] if name == "resnet" else original_caches()[1:], anchors.image_id.tolist()).astype(np.float64)
        relative = np.linalg.norm(current - old, axis=1) / np.linalg.norm(old, axis=1)
        assert relative.max() < .005
        started = {"started_at_utc": utc(), "manifest_sha256": digest(OUT / "manifest.csv"),
                   "runtime_protocol_sha256": digest(OUT / "runtime_protocol.json"), "model": identity,
                   "preprocessing": preproc, "anchor_relative_l2": relative.tolist(), "images": 1200}
        write_json(OUT / f"{name}_inference_started.json", started)
        vectors = infer(frame, transform, forward, "cuda", 8)
        np.savez(target, image_id=frame.image_id.to_numpy(), embedding=vectors, manifest_sha256=digest(OUT / "manifest.csv"))
        write_json(OUT / f"{name}_inference.json", {**started, "completed_at_utc": utc(), "embedding_sha256": digest(target)})
        del model, transform, forward, vectors
        gc.collect()
        torch.cuda.empty_cache()


def analyze():
    protocol = lock()
    frame = pd.read_csv(OUT / "manifest.csv", keep_default_na=False)
    pair_ids = sorted(frame.pair_id.unique())
    mapping = {p: i for i, p in enumerate(pair_ids)}
    idx = frame.pair_id.map(mapping).to_numpy()
    rng = np.random.default_rng(20261012)
    counts = np.stack([np.bincount(rng.integers(200, size=200), minlength=200) for _ in range(2000)]).astype(np.float32)
    rows, predictions = [], frame.copy()
    for backbone in ["resnet", "clip"]:
        with np.load(OUT / f"{backbone}_embeddings.npz", allow_pickle=True) as z:
            assert str(z["manifest_sha256"]) == digest(OUT / "manifest.csv") and np.array_equal(z["image_id"].astype(str), frame.image_id)
            x = z["embedding"].astype(np.float32)
        for regime in ["joint", "SD_excluded"]:
            head_paths = [ROOT / p for p in protocol["heads"][f"{regime}_{backbone}"]]
            score_map = systems(x, frame.active_score.to_numpy(), *head_paths)
            saved = json.loads((CAL / "thresholds_locked.json" if regime == "joint" else CAL / "sd_family_excluded/thresholds_locked.json").read_text())
            entry = saved["models"][f"20260920_{backbone}" if regime == "joint" else backbone]
            for system, score in score_map.items():
                predictions[f"{regime}_{backbone}_{system}_logit"] = score
                for attack in ["identity", "jpeg", "lowres", "overall"]:
                    mask = np.ones(len(frame), dtype=bool) if attack == "overall" else frame.attack.eq(attack).to_numpy()
                    y, s, pi = frame.label.to_numpy()[mask], score[mask], idx[mask]
                    auc = roc_auc_score(y, s)
                    draws = weighted_auc_draws(y, s, pi, counts)
                    base = {"regime": regime, "backbone": backbone, "system": system, "condition": attack,
                            "AUC": float(auc), "AUC_ci95_low": float(np.percentile(draws, 2.5)), "AUC_ci95_high": float(np.percentile(draws, 97.5)),
                            "pairs": 200, "copies": len(y)}
                    for policy_name, policy in entry["systems"][system].items():
                        threshold = policy["threshold_logit"]
                        accept = np.zeros(len(s), dtype=bool) if threshold is None else (s > threshold if policy["comparison"] == ">" else s >= threshold)
                        tpr = rate_draws(y, accept, pi, counts, 1)
                        fpr = rate_draws(y, accept, pi, counts, 0)
                        real_pred = np.bincount(pi[y == 0], weights=accept[y == 0], minlength=200) > 0
                        any_fpr = counts @ real_pred / 200
                        rows.append({**base, "policy": policy_name, "threshold_logit": threshold, "comparison": policy["comparison"],
                                     "TPR": float(accept[y == 1].mean()), "FPR": float(accept[y == 0].mean()),
                                     "TPR_ci95_low": float(np.percentile(tpr, 2.5)), "TPR_ci95_high": float(np.percentile(tpr, 97.5)),
                                     "FPR_ci95_low": float(np.percentile(fpr, 2.5)), "FPR_ci95_high": float(np.percentile(fpr, 97.5)),
                                     "source_any_FPR": float(real_pred.mean()), "source_any_FPR_ci95_low": float(np.percentile(any_fpr, 2.5)), "source_any_FPR_ci95_high": float(np.percentile(any_fpr, 97.5))})
    predictions.to_csv(OUT / "predictions.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "metrics.csv", index=False)
    summary = {"completed_at_utc": utc(), "pairs": 200, "origin_sources": 400, "copies": 1200,
               "primary": result.loc[result.condition.eq("identity") & result.system.eq("probe") & result.policy.eq("per_image_core13")].to_dict("records"),
               "all_metrics": rows, "paired_bootstrap_replicates": 2000,
               "test_refitting": False, "test_threshold_tuning": False, "candidate_resampling": False,
               "interpretation": "Prospective unused photograph/caption pairs, one accelerated SD1.5-family generated sample per caption. Pairing reduces coarse content imbalance; residual styles, encoding histories and pretrained overlap remain. SD-exclusion heads train on DALL-E-only synthetic sources and do not establish universal generator transfer."}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "complete.json", {"completed_at_utc": utc(), "pairs": 200, "sources": 400, "copies": 1200,
               "script_sha256": digest(Path(__file__)), "output_sha256": {name: digest(OUT / name) for name in ["runtime_protocol.json", "selection_complete.json", "generation_complete.json", "generated_sources.csv", "manifest.csv", "resnet_inference.json", "clip_inference.json", "metrics.csv", "summary.json"]},
               "test_refitting": False, "test_threshold_tuning": False})
    print(result.loc[result.system.eq("probe") & result.policy.eq("per_image_core13"), ["regime", "backbone", "condition", "AUC", "TPR", "FPR"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["lock", "generate", "derive", "extract", "analyze", "all"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        for stage in (["lock", "generate", "derive", "extract", "analyze"] if args.stage == "all" else [args.stage]):
            globals()[stage]()
