from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attacks import apply_attack, is_held_out_row
from .config import resolve
from .decision import decision_metrics, fit_decision_policy
from .evaluation import (
    classification_metrics,
    paired_auc_difference_ci,
    positive_rate,
    select_threshold,
    source_cluster_bootstrap_ci,
    stratified_cluster_bootstrap_ci,
)
from .io import image_quality, read_image, standardize, write_image
from .manifest import MANIFEST_COLUMNS, stable_id, validate_manifest
from .models import FusionCalibrator, LinearProbe, extract_embeddings
from .quality import degradation_features, degraded_indicator
from .watermark import build_watermarker


def preprocess_manifest(config: dict, root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    width = int(config["image"]["width"])
    height = int(config["image"]["height"])
    output_root = resolve(root, config["data"]["processed_dir"]) / "base"
    records = []
    for row in frame.to_dict("records"):
        output = output_root / f"{row['image_id']}.png"
        write_image(output, standardize(read_image(row["path"]), width, height))
        row["path"] = str(output.resolve())
        records.append(row)
    result = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    validate_manifest(result)
    return result


def _derived_row(
    parent: dict,
    path: Path,
    group: str,
    attack: str,
    severity: int,
    payload: str,
    method: str = "",
):
    signature = f"{parent['image_id']}:{group}:{attack}:{severity}:{payload}"
    row = parent.copy()
    row.update(
        {
            "image_id": stable_id(signature),
            "path": str(path.resolve()),
            "group": group,
            "watermark_method": method if payload else "",
            "payload": payload,
            "attack": attack,
            "severity": severity,
            "parent_id": parent["image_id"],
        }
    )
    return row


def _derived_specs(
    parent: dict,
    output_root: Path,
    config: dict,
    payload: str,
    wrong_payload: str,
    method: str,
) -> list[tuple[Path, str, str, int, str, str]]:
    folder = output_root / parent["image_id"]
    if int(parent["label"]) == 1:
        specs = [(folder / "S-W__identity__0.png", "S-W", "identity", 0, payload, method)]
        group, row_payload, row_method = "S-A", payload, method
    else:
        specs = [(folder / "R-S__identity__0.png", "R-S", "identity", 0, wrong_payload, method)]
        group, row_payload, row_method = "R-T", "", ""
    for attack in config["attacks"]:
        name, severity = str(attack["name"]), int(attack["severity"])
        if name == "identity":
            continue
        specs.append(
            (
                folder / f"{group}__{name}__{severity}.png",
                group,
                name,
                severity,
                row_payload,
                row_method,
            )
        )
    return specs


def derive_dataset(config: dict, root: Path, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    watermark_cfg = config["watermark"]
    payload = str(watermark_cfg["payload"])
    wrong_payload = "".join("1" if bit == "0" else "0" for bit in payload)
    method = str(watermark_cfg["method"])
    marker = build_watermarker(method, payload, float(watermark_cfg["strength"]))
    spoof_marker = build_watermarker(method, wrong_payload, float(watermark_cfg["strength"]))
    output_root = resolve(root, config["data"]["processed_dir"]) / "derived"
    records = base.to_dict("records")
    quality_records: list[dict] = []

    parents = base.to_dict("records")
    for index, parent in enumerate(parents):
        if index % 100 == 0:
            print(f"derive {index}/{len(parents)}", flush=True)
        specs = _derived_specs(parent, output_root, config, payload, wrong_payload, method)
        if all(path.exists() for path, *_rest in specs):
            for path, group, name, severity, row_payload, row_method in specs:
                records.append(
                    _derived_row(parent, path, group, name, severity, row_payload, row_method)
                )
            if int(parent["label"]) == 1:
                marked_path = output_root / parent["image_id"] / "S-W__identity__0.png"
                quality_records.append(
                    {
                        "image_id": parent["image_id"],
                        "group": "S-W",
                        **image_quality(read_image(parent["path"]), read_image(marked_path)),
                    }
                )
            continue
        original = read_image(parent["path"])
        if int(parent["label"]) == 1:
            marked = marker.embed(original)
            clean_path = output_root / parent["image_id"] / "S-W__identity__0.png"
            write_image(clean_path, marked)
            records.append(_derived_row(parent, clean_path, "S-W", "identity", 0, payload, method))
            quality_records.append(
                {"image_id": parent["image_id"], "group": "S-W", **image_quality(original, marked)}
            )
            for attack in config["attacks"]:
                name, severity = str(attack["name"]), int(attack["severity"])
                if name == "identity":
                    continue
                path = output_root / parent["image_id"] / f"S-A__{name}__{severity}.png"
                write_image(path, apply_attack(marked, name, severity))
                records.append(_derived_row(parent, path, "S-A", name, severity, payload, method))
        else:
            spoof_path = output_root / parent["image_id"] / "R-S__identity__0.png"
            write_image(spoof_path, spoof_marker.embed(original))
            records.append(
                _derived_row(parent, spoof_path, "R-S", "identity", 0, wrong_payload, method)
            )
            for attack in config["attacks"]:
                name, severity = str(attack["name"]), int(attack["severity"])
                if name == "identity":
                    continue
                path = output_root / parent["image_id"] / f"R-T__{name}__{severity}.png"
                write_image(path, apply_attack(original, name, severity))
                records.append(_derived_row(parent, path, "R-T", name, severity, ""))

    result = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    validate_manifest(result)
    return result, pd.DataFrame(quality_records)


def add_active_scores(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    watermark_cfg = config["watermark"]
    marker = build_watermarker(
        str(watermark_cfg["method"]),
        str(watermark_cfg["payload"]),
        float(watermark_cfg["strength"]),
    )
    records = []
    rows = manifest.to_dict("records")
    for index, row in enumerate(rows):
        if index % 5000 == 0:
            print(f"active {index}/{len(rows)}", flush=True)
        detection = marker.detect(read_image(row["path"]))
        records.append(
            {
                "image_id": row["image_id"],
                "active_score": detection.bit_accuracy,
                "watermark_confidence": detection.confidence,
                "watermark_verified": detection.verified,
            }
        )
    return manifest.merge(pd.DataFrame(records), on="image_id", how="left")


def humanify_indicator(groups: pd.Series) -> np.ndarray:
    """Attack-head label: edited watermarked AI, without asserting removal."""
    return (groups.astype(str) == "S-A").to_numpy(dtype=int)


def _probe_name(kind: str, variant: str) -> str:
    return f"{kind}.npz" if variant == "full" else f"{kind}_{variant}.npz"


def _training_rows(config: dict, manifest: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Apply the held-out protocol before any ablation-specific exclusions."""
    train = manifest.loc[manifest["split"] == "train"].copy()
    train = train.loc[~is_held_out_row(train, config)]
    if variant == "no-attacks":
        train = train.loc[~train["group"].isin(["R-T", "S-A"])]
    elif variant == "no-matched-controls":
        train = train.loc[train["group"] != "R-T"]
    elif variant != "full":
        raise ValueError(f"Unknown training variant: {variant}")
    return train


def train_passive(
    config: dict, root: Path, manifest: pd.DataFrame, variant: str = "full"
) -> Path:
    train = _training_rows(config, manifest, variant)
    if train.empty:
        raise ValueError("Training split is empty.")
    embeddings = extract_embeddings(
        train,
        str(config["model"]["backbone"]),
        bool(config["model"]["pretrained"]),
        str(config["model"]["device"]),
        int(config["model"]["batch_size"]),
    )
    vectors = np.stack(embeddings["embedding"].to_numpy())
    probe = LinearProbe.fit(vectors, embeddings["label"].to_numpy())
    output = resolve(root, config["evaluation"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / _probe_name("passive", variant)
    probe.save(path)

    grouped = embeddings.merge(train[["image_id", "group"]], on="image_id", how="left")
    humanify = humanify_indicator(grouped["group"])
    if humanify.min() != humanify.max():
        LinearProbe.fit(vectors, humanify).save(output / _probe_name("humanify", variant))
    _fit_quality_probe(train, output, variant)
    if bool(config["model"].get("compare_clip", False)):
        _fit_backbone_probe(config, train, "clip", output, variant)
    return path


def _fit_quality_probe(train: pd.DataFrame, output: Path, variant: str) -> None:
    rows = train.loc[~((train["attack"].astype(str) == "lowres") & (train["severity"].astype(int) == 2))]
    labels = degraded_indicator(rows["attack"])
    if len(labels) == 0 or labels.min() == labels.max():
        return
    features = np.stack([degradation_features(read_image(path)) for path in rows["path"]])
    LinearProbe.fit(features, labels).save(output / _probe_name("quality", variant))


def _fit_backbone_probe(
    config: dict, train: pd.DataFrame, backbone: str, output: Path, variant: str
) -> None:
    embeddings = extract_embeddings(
        train,
        backbone,
        bool(config["model"]["pretrained"]),
        str(config["model"]["device"]),
        int(config["model"]["batch_size"]),
    )
    vectors = np.stack(embeddings["embedding"].to_numpy()).astype(np.float32)
    if backbone == "clip" and variant == "full":
        np.savez(output / "clip_train_embeddings.npz", image_id=embeddings["image_id"].to_numpy(), embedding=vectors)
    LinearProbe.fit(vectors, embeddings["label"].to_numpy()).save(output / _probe_name(backbone, variant))
    grouped = embeddings.merge(train[["image_id", "group"]], on="image_id", how="left")
    humanify = humanify_indicator(grouped["group"])
    if humanify.min() != humanify.max():
        LinearProbe.fit(vectors, humanify).save(output / _probe_name(f"{backbone}_humanify", variant))


def _load_cached_embeddings(cache: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray] | None:
    if not cache.exists():
        return None
    values = np.load(cache, allow_pickle=True)
    table = pd.DataFrame(
        {"image_id": values["image_id"].astype(str), "embedding": list(values["embedding"])}
    )
    aligned = manifest[["image_id"]].astype({"image_id": str}).merge(table, on="image_id", how="left")
    if len(aligned) != len(manifest) or aligned["embedding"].isna().any():
        return None
    vectors = np.stack(aligned["embedding"].to_numpy()).astype(np.float32)
    print(f"loaded cached embeddings from {cache.name}", flush=True)
    return aligned, vectors


def add_passive_scores(
    config: dict, root: Path, manifest: pd.DataFrame, variant: str = "full"
) -> pd.DataFrame:
    output = resolve(root, config["evaluation"]["output_dir"])
    cached = _load_cached_embeddings(output / "resnet_embeddings.npz", manifest) if variant == "full" else None
    if cached is None:
        embeddings = extract_embeddings(
            manifest,
            str(config["model"]["backbone"]),
            bool(config["model"]["pretrained"]),
            str(config["model"]["device"]),
            int(config["model"]["batch_size"]),
        )
        vectors = np.stack(embeddings["embedding"].to_numpy()).astype(np.float32)
        if variant == "full":
            np.savez(
                output / "resnet_embeddings.npz",
                image_id=embeddings["image_id"].astype(str).to_numpy(),
                embedding=vectors,
            )
    else:
        embeddings, vectors = cached
    probe = LinearProbe.load(output / _probe_name("passive", variant))
    embeddings["passive_score"] = probe.predict_proba(vectors)
    humanify_path = output / _probe_name("humanify", variant)
    if humanify_path.exists():
        embeddings["humanify_score"] = LinearProbe.load(humanify_path).predict_proba(vectors)
    columns = ["image_id", "passive_score"]
    if "humanify_score" in embeddings.columns:
        columns.append("humanify_score")
    return manifest.merge(embeddings[columns], on="image_id", how="left")


def _score_degradation(config: dict, validation: pd.DataFrame, test: pd.DataFrame, path: Path) -> dict:
    probe = LinearProbe.load(path)
    for frame in (validation, test):
        features = np.stack([degradation_features(read_image(row)) for row in frame["path"]])
        frame["degraded_score"] = probe.predict_proba(features)
        frame["degraded"] = degraded_indicator(frame["attack"])
    if validation["degraded"].nunique() != 2 or test["degraded"].nunique() != 2:
        return {"skipped": True}
    threshold = select_threshold(validation["degraded"].to_numpy(), validation["degraded_score"].to_numpy())
    report = classification_metrics(test["degraded"].to_numpy(), test["degraded_score"].to_numpy(), threshold)
    print("interval degradation", flush=True)
    report["roc_auc_ci"] = source_cluster_bootstrap_ci(
        test["degraded"].to_numpy(),
        test["degraded_score"].to_numpy(),
        test["source_id"].to_numpy(),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    report["by_attack"] = {}
    for attack, subset in test.groupby("attack"):
        report["by_attack"][str(attack)] = {
            "n": int(len(subset)),
            "positive_rate": positive_rate(subset["degraded_score"].to_numpy(), threshold),
            "mean_score": float(subset["degraded_score"].mean()),
        }
    held = test[(test["attack"].astype(str) == "lowres") & (test["severity"].astype(int) == 2)]
    if not held.empty:
        report["heldout_lowres_positive_rate"] = positive_rate(held["degraded_score"].to_numpy(), threshold)
    return report


def _score_clip(
    config: dict, validation: pd.DataFrame, test: pd.DataFrame, output_dir: Path, variant: str
) -> dict:
    scored = pd.concat([validation, test], ignore_index=True)
    cached = (
        _load_cached_embeddings(output_dir / "clip_eval_embeddings.npz", scored) if variant == "full" else None
    )
    if cached is None:
        embeddings = extract_embeddings(
            scored,
            "clip",
            bool(config["model"]["pretrained"]),
            str(config["model"]["device"]),
            int(config["model"]["batch_size"]),
        )
        vectors = np.stack(embeddings["embedding"].to_numpy()).astype(np.float32)
        if variant == "full":
            np.savez(
                output_dir / "clip_eval_embeddings.npz",
                image_id=embeddings["image_id"].astype(str).to_numpy(),
                embedding=vectors,
            )
    else:
        embeddings, vectors = cached
    probe = LinearProbe.load(output_dir / _probe_name("clip", variant))
    embeddings["clip_score"] = probe.predict_proba(vectors)
    humanify_path = output_dir / _probe_name("clip_humanify", variant)
    if humanify_path.exists():
        embeddings["clip_humanify_score"] = LinearProbe.load(humanify_path).predict_proba(vectors)
    merged = scored.merge(
        embeddings[[column for column in ("image_id", "clip_score", "clip_humanify_score") if column in embeddings]],
        on="image_id",
        how="left",
    )
    val = merged[merged["split"] == "val"].copy()
    held = merged[merged["split"] == "test"].copy()
    fusion = FusionCalibrator.fit(val["clip_score"].to_numpy(), val["active_score"].to_numpy(), val["label"].to_numpy())
    fusion.save(output_dir / _probe_name("clip_fusion", variant))
    for frame in (val, held):
        frame["clip_fusion_score"] = fusion.predict_proba(
            frame["clip_score"].to_numpy(), frame["active_score"].to_numpy()
        )
    report = {
        "passive": _fitted_metrics(config, val, held, "label", "clip_score"),
        "fusion": _fitted_metrics(config, val, held, "label", "clip_fusion_score"),
    }
    if "clip_humanify_score" in held.columns:
        val = val.copy()
        held = held.copy()
        val["humanify"] = humanify_indicator(val["group"])
        held["humanify"] = humanify_indicator(held["group"])
        if val["humanify"].nunique() == 2 and held["humanify"].nunique() == 2:
            threshold = select_threshold(val["humanify"].to_numpy(), val["clip_humanify_score"].to_numpy())
            metrics = classification_metrics(
                held["humanify"].to_numpy(), held["clip_humanify_score"].to_numpy(), threshold
            )
            metrics["roc_auc_ci"] = source_cluster_bootstrap_ci(
                held["humanify"].to_numpy(),
                held["clip_humanify_score"].to_numpy(),
                held["source_id"].to_numpy(),
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]),
            )
            metrics["true_positive_on_stripped_ai"] = positive_rate(
                held.loc[held["group"] == "S-A", "clip_humanify_score"].to_numpy(), threshold
            )
            metrics["false_positive_on_matched_real"] = positive_rate(
                held.loc[held["group"] == "R-T", "clip_humanify_score"].to_numpy(), threshold
            )
            report["humanify"] = metrics
    return report


def _fitted_metrics(
    config: dict, validation: pd.DataFrame, test: pd.DataFrame, label: str, column: str
) -> dict:
    threshold = select_threshold(validation[label].to_numpy(), validation[column].to_numpy())
    metrics = classification_metrics(test[label].to_numpy(), test[column].to_numpy(), threshold)
    print(f"interval {column}", flush=True)
    metrics["roc_auc_ci"] = stratified_cluster_bootstrap_ci(
        test[label].to_numpy(),
        test[column].to_numpy(),
        test["source_id"].to_numpy(),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    return metrics


def _removal_attack_mask(frame: pd.DataFrame, config: dict) -> pd.Series:
    """Select declared stress conditions without inferring removal per image."""
    default = [
        {"name": "crop", "severity": 1},
        {"name": "lowres", "severity": 2},
        {"name": "regenerate", "severity": 2},
        {"name": "spectral", "severity": 1},
    ]
    specifications = config.get("evaluation", {}).get("removal_attacks", default)
    keys = {(str(item["name"]), int(item["severity"])) for item in specifications}
    return pd.Series(
        [(str(attack), int(severity)) in keys for attack, severity in zip(frame["attack"], frame["severity"])],
        index=frame.index,
    )


def evaluate_scores(
    config: dict, root: Path, scored: pd.DataFrame, variant: str = "full"
) -> dict:
    output_dir = resolve(root, config["evaluation"]["output_dir"])
    if variant != "full":
        output_dir = output_dir / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = scored[scored["split"] == "val"].copy()
    test = scored[scored["split"] == "test"].copy()
    if validation["label"].nunique() != 2 or test["label"].nunique() != 2:
        raise ValueError("Validation and test sets must each contain both classes.")

    val_fit = validation.loc[~is_held_out_row(validation, config)].copy()
    if val_fit["label"].nunique() != 2:
        val_fit = validation.copy()
    fusion = FusionCalibrator.fit(
        val_fit["passive_score"].to_numpy(),
        val_fit["active_score"].to_numpy(),
        val_fit["label"].to_numpy(),
    )
    fusion.save(output_dir / "fusion.npz")
    for frame in (validation, val_fit, test):
        frame["fusion_score"] = fusion.predict_proba(
            frame["passive_score"].to_numpy(), frame["active_score"].to_numpy()
        )

    systems = {
        "active": "active_score",
        "passive": "passive_score",
        "fusion": "fusion_score",
    }
    report: dict = {"contribution": "generate_humanify_detect"}
    for system, column in systems.items():
        print(f"interval {system}", flush=True)
        threshold = select_threshold(val_fit["label"].to_numpy(), val_fit[column].to_numpy())
        labels, scores = test["label"].to_numpy(), test[column].to_numpy()
        metrics = classification_metrics(labels, scores, threshold)
        metrics["roc_auc_ci"] = stratified_cluster_bootstrap_ci(
            labels,
            scores,
            test["source_id"].to_numpy(),
            samples=int(config["evaluation"]["bootstrap_samples"]),
            seed=int(config["seed"]),
        )
        report[system] = metrics

    print("interval fusion-minus-passive", flush=True)
    estimate, lower, upper = paired_auc_difference_ci(
        test["label"].to_numpy(),
        test["fusion_score"].to_numpy(),
        test["passive_score"].to_numpy(),
        test["source_id"].to_numpy(),
        samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    report["paired_fusion_minus_passive_auc"] = {"estimate": estimate, "ci95": [lower, upper]}

    report["conditions"] = {}
    for (attack, severity), subset in test.groupby(["attack", "severity"]):
        if subset["label"].nunique() != 2:
            continue
        key = f"{attack}:severity={severity}"
        report["conditions"][key] = {}
        for system, column in systems.items():
            report["conditions"][key][system] = classification_metrics(
                subset["label"].to_numpy(),
                subset[column].to_numpy(),
                float(report[system]["threshold"]),
            )

    report["group_positive_rate"] = {}
    report["artifact_shortcut"] = {}
    for system, column in systems.items():
        threshold = float(report[system]["threshold"])
        report["group_positive_rate"][system] = {}
        for group, subset in test.groupby("group"):
            report["group_positive_rate"][system][str(group)] = {
                "n": int(len(subset)),
                "positive_rate": positive_rate(subset[column].to_numpy(), threshold),
            }
        groups = report["group_positive_rate"][system]
        rc = groups.get("R-C", {}).get("positive_rate", float("nan"))
        rt = groups.get("R-T", {}).get("positive_rate", float("nan"))
        rs = groups.get("R-S", {}).get("positive_rate", float("nan"))
        report["artifact_shortcut"][system] = {
            "delta_fpr_rt_minus_rc": float(rt - rc) if np.isfinite(rc) and np.isfinite(rt) else float("nan"),
            "spoof_accept_rate": rs,
        }

    report["watermark_recovery"] = {}
    positives = test[test["group"].isin(["S-W", "S-A"])]
    for (attack, severity), subset in positives.groupby(["attack", "severity"]):
        key = f"{attack}:severity={severity}"
        report["watermark_recovery"][key] = {
            "mean_bit_accuracy": float(subset["active_score"].mean()),
            "verification_rate": positive_rate(subset["active_score"].to_numpy(), report["active"]["threshold"]),
            "verification_threshold": float(report["active"]["threshold"]),
            "n": int(len(subset)),
        }

    decision_cfg = config.get("decision", {})
    policy = fit_decision_policy(
        val_fit,
        max_real_fpr=float(decision_cfg.get("max_real_fpr", 0.05)),
        min_coverage=float(decision_cfg.get("min_coverage", 0.70)),
        shortcut_slack=float(decision_cfg.get("shortcut_slack", 0.02)),
    )
    test["decision"] = policy.decide_many(
        test["active_score"].to_numpy(), test["fusion_score"].to_numpy()
    )
    report["decision"] = decision_metrics(test, policy)

    killed = test.loc[_removal_attack_mask(test, config)]
    if not killed.empty and killed["label"].nunique() == 2:
        report["residual_after_payload_death"] = {
            "n": int(len(killed)),
            "mean_bit_accuracy": float(killed["active_score"].mean()),
            "systems": {},
        }
        for system, column in systems.items():
            report["residual_after_payload_death"]["systems"][system] = classification_metrics(
                killed["label"].to_numpy(),
                killed[column].to_numpy(),
                float(report[system]["threshold"]),
            )
        report["residual_after_payload_death"]["decision"] = decision_metrics(killed, policy)

    held_attacks = test.loc[is_held_out_row(test, config)]
    if not held_attacks.empty and held_attacks["label"].nunique() == 2:
        report["heldout_attacks"] = {"systems": {}, "decision": decision_metrics(held_attacks, policy)}
        for system, column in systems.items():
            report["heldout_attacks"]["systems"][system] = classification_metrics(
                held_attacks["label"].to_numpy(),
                held_attacks[column].to_numpy(),
                float(report[system]["threshold"]),
            )

    heldout = config["data"].get("heldout_generator")
    if heldout:
        cross_generator = test[
            (test["label"] == 0) | (test["generator"].astype(str) == str(heldout))
        ]
        if cross_generator["label"].nunique() == 2:
            report["heldout_generator"] = {"name": str(heldout), "systems": {}}
            for system, column in systems.items():
                report["heldout_generator"]["systems"][system] = classification_metrics(
                    cross_generator["label"].to_numpy(),
                    cross_generator[column].to_numpy(),
                    float(report[system]["threshold"]),
                )

    if "humanify_score" in test.columns and test["humanify_score"].notna().any():
        test["humanify"] = humanify_indicator(test["group"])
        val_human = val_fit.copy()
        val_human["humanify"] = humanify_indicator(val_human["group"])
        if val_human["humanify"].nunique() == 2 and test["humanify"].nunique() == 2:
            threshold = select_threshold(
                val_human["humanify"].to_numpy(), val_human["humanify_score"].to_numpy()
            )
            report["humanify"] = classification_metrics(
                test["humanify"].to_numpy(),
                test["humanify_score"].to_numpy(),
                threshold,
            )
            print("interval humanify", flush=True)
            report["humanify"]["roc_auc_ci"] = source_cluster_bootstrap_ci(
                test["humanify"].to_numpy(),
                test["humanify_score"].to_numpy(),
                test["source_id"].to_numpy(),
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["seed"]),
            )
            report["humanify"]["false_positive_on_matched_real"] = positive_rate(
                test.loc[test["group"] == "R-T", "humanify_score"].to_numpy(),
                threshold,
            )
            report["humanify"]["true_positive_on_stripped_ai"] = positive_rate(
                test.loc[test["group"] == "S-A", "humanify_score"].to_numpy(),
                threshold,
            )
            held_h = test.loc[is_held_out_row(test, config)]
            if not held_h.empty and held_h["humanify"].nunique() == 2:
                report["humanify_heldout_attacks"] = classification_metrics(
                    held_h["humanify"].to_numpy(),
                    held_h["humanify_score"].to_numpy(),
                    threshold,
                )

    quality_path = output_dir / _probe_name("quality", variant)
    if quality_path.exists():
        print("degradation probe", flush=True)
        report["degradation"] = _score_degradation(config, val_fit, test, quality_path)

    clip_path = output_dir / _probe_name("clip", variant)
    if clip_path.exists():
        print("clip probe", flush=True)
        report["clip"] = _score_clip(config, val_fit, test, output_dir, variant)

    test.to_csv(output_dir / "test_predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
