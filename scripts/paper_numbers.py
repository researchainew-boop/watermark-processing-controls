"""Collect the numbers the manuscript is allowed to quote.

Reads outputs/metrics.json and outputs/test_predictions.csv from the
native-resolution run. Intervals use the same source-clustered bootstrap
as the evaluator. True-positive rate at 5% false-positive rate is a point
on the test ROC, not a threshold fit on validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aiwstudy.config import load_config
from aiwstudy.evaluation import (
    positive_rate,
    stratified_cluster_bootstrap_ci,
    true_positive_rate_at_fpr,
)
from aiwstudy.models import FusionCalibrator, LinearProbe


ROOT = Path(__file__).resolve().parents[1]
DEAD = {("crop", 1), ("lowres", 2), ("regenerate", 2), ("spectral", 1)}


def _ci(labels, scores, groups, samples, seed, clusters: str) -> dict:
    lower, upper = stratified_cluster_bootstrap_ci(
        labels, scores, groups, samples=samples, seed=seed
    )
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "ci95": [lower, upper],
        "tpr_at_fpr_5pct_test_roc": float(true_positive_rate_at_fpr(labels, scores, 0.05)),
        "n": int(len(labels)),
        "clusters": clusters,
    }


def _pair(frame: pd.DataFrame, column: str, samples: int, seed: int) -> dict:
    subset = frame[frame["group"].isin(["S-A", "R-T"])].dropna(subset=[column])
    return _ci(
        subset["label"].to_numpy(),
        subset[column].to_numpy(),
        subset["source_id"].to_numpy(),
        samples,
        seed,
        "source_id",
    )


def _dead(frame: pd.DataFrame, column: str, samples: int, seed: int) -> dict:
    mask = [
        (str(attack), int(severity)) in DEAD and group in {"S-A", "R-T"}
        for attack, severity, group in zip(frame["attack"], frame["severity"], frame["group"])
    ]
    subset = frame.loc[mask].dropna(subset=[column])
    return _ci(
        subset["label"].to_numpy(),
        subset[column].to_numpy(),
        subset["source_id"].to_numpy(),
        samples,
        seed,
        "source_id",
    )


def _clip_frame(test: pd.DataFrame) -> pd.DataFrame | None:
    embeddings_path = ROOT / "outputs" / "clip_eval_embeddings.npz"
    probe_path = ROOT / "outputs" / "clip.npz"
    fusion_path = ROOT / "outputs" / "clip_fusion.npz"
    if not (embeddings_path.exists() and probe_path.exists() and fusion_path.exists()):
        return None
    values = np.load(embeddings_path, allow_pickle=True)
    vectors = pd.DataFrame(
        {"image_id": values["image_id"].astype(str), "embedding": list(values["embedding"])}
    )
    merged = test.merge(vectors, on="image_id", how="inner")
    matrix = np.stack(merged["embedding"].to_numpy())
    merged["clip_score"] = LinearProbe.load(probe_path).predict_proba(matrix)
    merged["clip_fusion_score"] = FusionCalibrator.load(fusion_path).predict_proba(
        merged["clip_score"].to_numpy(), merged["active_score"].to_numpy()
    )
    return merged


def _system(block: dict) -> dict:
    return {
        "roc_auc": block["roc_auc"],
        "ci95": block["roc_auc_ci"],
        "threshold_validation": block["threshold"],
        "recall_at_validation_threshold": block["recall"],
        "fpr_at_validation_threshold": block["false_positive_rate"],
        "tpr_at_fpr_5pct_test_roc": block["tpr_at_fpr_5pct"],
        "n": block["n"],
    }


def main() -> None:
    config, _root = load_config(ROOT / "configs" / "study.yaml")
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["seed"])
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    sources = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
    test = pd.read_csv(ROOT / "outputs" / "test_predictions.csv")
    test["image_id"] = test["image_id"].astype(str)
    quality = pd.read_csv(ROOT / "outputs" / "quality.csv")

    facts = {
        "sources_by_split_generator": {
            f"{split}|{generator}": int(count)
            for (split, generator), count in sources.groupby(["split", "generator"]).size().items()
        },
        "unique_sources": int(sources["image_id"].nunique()) if "image_id" in sources.columns else int(len(sources)),
        "test_images": int(len(test)),
        "test_sources": int(test["source_id"].nunique()),
        "mean_embed_psnr": float(quality["psnr"].mean()),
        "mean_embed_ssim": float(quality["ssim"].mean()) if quality["ssim"].notna().any() else None,
        "active": _system(metrics["active"]),
        "resnet_passive": _system(metrics["passive"]),
        "resnet_fusion": _system(metrics["fusion"]),
        "paired_fusion_minus_passive_auc": metrics["paired_fusion_minus_passive_auc"],
        "group_positive_rate": metrics.get("group_positive_rate", {}),
        "artifact_shortcut": metrics.get("artifact_shortcut", {}),
        "watermark_recovery": metrics.get("watermark_recovery", {}),
        "condition_auc": {
            key: {name: block["roc_auc"] for name, block in systems.items()}
            for key, systems in metrics.get("conditions", {}).items()
        },
        "decision": metrics.get("decision", {}),
        "humanify": metrics.get("humanify", {}),
        "degradation": {
            "roc_auc": metrics.get("degradation", {}).get("roc_auc"),
            "ci95": metrics.get("degradation", {}).get("roc_auc_ci"),
            "threshold": metrics.get("degradation", {}).get("threshold"),
            "by_attack": metrics.get("degradation", {}).get("by_attack", {}),
            "heldout_lowres_positive_rate": metrics.get("degradation", {}).get(
                "heldout_lowres_positive_rate"
            ),
        },
        "heldout_generator": metrics.get("heldout_generator", {}),
        "clip": {
            "passive": _system(metrics["clip"]["passive"]) if "clip" in metrics else None,
            "fusion": _system(metrics["clip"]["fusion"]) if "clip" in metrics else None,
            "humanify": metrics.get("clip", {}).get("humanify"),
        },
        "matched_sa_vs_rt": {
            "resnet_passive": _pair(test, "passive_score", samples, seed),
            "resnet_fusion": _pair(test, "fusion_score", samples, seed),
            "resnet_humanify": _pair(test, "humanify_score", samples, seed),
        },
        "payload_dead_sa_vs_rt": {
            "resnet_passive": _dead(test, "passive_score", samples, seed),
            "resnet_fusion": _dead(test, "fusion_score", samples, seed),
        },
    }
    threshold = float(metrics["active"]["threshold"])
    facts["payload_accept_by_group"] = {
        str(group): {
            "n": int(len(subset)),
            "accept_rate": positive_rate(subset["active_score"].to_numpy(), threshold),
        }
        for group, subset in test.groupby("group")
    }
    clip = _clip_frame(test)
    if clip is not None:
        facts["matched_sa_vs_rt"]["clip_passive"] = _pair(clip, "clip_score", samples, seed)
        facts["matched_sa_vs_rt"]["clip_fusion"] = _pair(clip, "clip_fusion_score", samples, seed)
        facts["payload_dead_sa_vs_rt"]["clip_passive"] = _dead(clip, "clip_score", samples, seed)
        facts["payload_dead_sa_vs_rt"]["clip_fusion"] = _dead(clip, "clip_fusion_score", samples, seed)

    destination = ROOT / "outputs" / "paper_facts.json"
    destination.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
