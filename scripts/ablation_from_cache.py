"""Refit probe ablations from embeddings saved by the full run.

Training variants:
  no-attacks: drop R-T and S-A from training
  no-matched-controls: drop R-T from training

Held-out attacks stay out of every training fit, matching the full run.
Fusion is recalibrated on validation active scores. Test ROC points at 5%
FPR are read from the test curve and are not validation thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiwstudy.attacks import is_held_out_row
from aiwstudy.config import load_config
from aiwstudy.evaluation import (
    classification_metrics,
    paired_auc_difference_ci,
    positive_rate,
    select_threshold,
    stratified_cluster_bootstrap_ci,
)
from aiwstudy.models import FusionCalibrator, LinearProbe
from aiwstudy.pipeline import add_active_scores


ROOT = Path(__file__).resolve().parents[1]


def _load_vectors(path: Path) -> pd.DataFrame:
    values = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        {
            "image_id": values["image_id"].astype(str),
            "embedding": list(values["embedding"]),
        }
    )


def _variant_train(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    train = frame[frame["split"] == "train"]
    if variant == "no-attacks":
        train = train[~train["group"].isin(["R-T", "S-A"])]
    elif variant == "no-matched-controls":
        train = train[train["group"] != "R-T"]
    return train


def _score_variant(
    config: dict,
    frame: pd.DataFrame,
    vectors: pd.DataFrame,
    variant: str,
    samples: int,
    seed: int,
) -> dict:
    merged = frame.merge(vectors, on="image_id", how="inner")
    train = _variant_train(merged, variant)
    probe = LinearProbe.fit(np.stack(train["embedding"].to_numpy()), train["label"].to_numpy())
    scored = merged[merged["split"].isin(["val", "test"])].copy()
    scored["passive_score"] = probe.predict_proba(np.stack(scored["embedding"].to_numpy()))
    val = scored[scored["split"] == "val"]
    val_fit = val.loc[~is_held_out_row(val, config)]
    if val_fit["label"].nunique() != 2:
        val_fit = val
    test = scored[scored["split"] == "test"]
    fusion = FusionCalibrator.fit(
        val_fit["passive_score"].to_numpy(),
        val_fit["active_score"].to_numpy(),
        val_fit["label"].to_numpy(),
    )
    test = test.copy()
    test["fusion_score"] = fusion.predict_proba(
        test["passive_score"].to_numpy(), test["active_score"].to_numpy()
    )
    report = {"train_rows": int(len(train)), "test_rows": int(len(test))}
    for name, column in (("passive", "passive_score"), ("fusion", "fusion_score")):
        labels = test["label"].to_numpy()
        scores = test[column].to_numpy()
        threshold = select_threshold(val_fit["label"].to_numpy(), val_fit[column].to_numpy()) if name == "passive" else select_threshold(
            val_fit["label"].to_numpy(),
            fusion.predict_proba(val_fit["passive_score"].to_numpy(), val_fit["active_score"].to_numpy()),
        )
        metrics = classification_metrics(labels, scores, threshold)
        metrics["roc_auc_ci"] = stratified_cluster_bootstrap_ci(
            labels, scores, test["source_id"].to_numpy(), samples=samples, seed=seed
        )
        metrics["threshold_fit"] = "validation"
        metrics["tpr_at_fpr_5pct_source"] = "test_roc"
        report[name] = metrics
    estimate, lower, upper = paired_auc_difference_ci(
        test["label"].to_numpy(),
        test["fusion_score"].to_numpy(),
        test["passive_score"].to_numpy(),
        test["source_id"].to_numpy(),
        samples=samples,
        seed=seed,
    )
    report["paired_fusion_minus_passive_auc"] = {"estimate": estimate, "ci95": [lower, upper]}
    real = test[test["label"] == 0]
    report["passive"]["test_fpr_at_val_threshold"] = positive_rate(
        real["passive_score"].to_numpy(), report["passive"]["threshold"]
    )
    return report


def _active_scores(config: dict, frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    cache = output / "active_scores.csv"
    if cache.exists():
        cached = pd.read_csv(cache)
        cached["image_id"] = cached["image_id"].astype(str)
        return cached
    predictions = output / "test_predictions.csv"
    known = pd.DataFrame(columns=["image_id", "active_score"])
    if predictions.exists():
        known = pd.read_csv(predictions, usecols=["image_id", "active_score"])
        known["image_id"] = known["image_id"].astype(str)
    missing = frame.loc[~frame["image_id"].isin(set(known["image_id"])), ["image_id", "path"]].copy()
    if missing.empty:
        known.to_csv(cache, index=False)
        return known
    print(f"detect payload on {len(missing)} images not in test_predictions.csv", flush=True)
    detected = add_active_scores(config, missing)[["image_id", "active_score"]]
    combined = pd.concat([known, detected], ignore_index=True).drop_duplicates("image_id")
    combined.to_csv(cache, index=False)
    return combined


def main() -> None:
    config, root = load_config(ROOT / "configs" / "study.yaml")
    output = root / "outputs"
    manifest = pd.read_csv(root / "data" / "manifest_derived.csv", keep_default_na=False)
    manifest["image_id"] = manifest["image_id"].astype(str)
    manifest = manifest.loc[~((manifest["split"] == "train") & is_held_out_row(manifest, config))].copy()
    active = _active_scores(config, manifest, output)
    manifest = manifest.merge(active, on="image_id", how="left")
    if manifest["active_score"].isna().any():
        raise SystemExit("Active scores are missing for some rows.")

    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["seed"])
    report = {}
    resnet_path = output / "resnet_embeddings.npz"
    if resnet_path.exists():
        vectors = _load_vectors(resnet_path)
        report["resnet"] = {
            variant: _score_variant(config, manifest, vectors, variant, samples, seed)
            for variant in ("no-attacks", "no-matched-controls")
        }
    clip_train = output / "clip_train_embeddings.npz"
    clip_eval = output / "clip_eval_embeddings.npz"
    if clip_train.exists() and clip_eval.exists():
        vectors = pd.concat([_load_vectors(clip_train), _load_vectors(clip_eval)], ignore_index=True)
        vectors = vectors.drop_duplicates("image_id")
        report["clip"] = {
            variant: _score_variant(config, manifest, vectors, variant, samples, seed)
            for variant in ("no-attacks", "no-matched-controls")
        }
    destination = output / "ablations.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
