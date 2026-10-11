"""Corrected conference experiment using the original frozen feature caches.

Never overwrites original manifests, feature caches, or pilot results. Exact
pixel hashes and conservative average-hash clusters select representatives
before fitting any model. This is a retrospective correction, not a new
preregistered experiment. Run with the project's analysis dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aiwstudy.attacks import is_held_out_row
from aiwstudy.decision import decision_metrics, fit_decision_policy
from aiwstudy.evaluation import classification_metrics, select_threshold

DEST = ROOT / "outputs" / "conference_clean"
DEAD = {("crop", 1), ("lowres", 2), ("regenerate", 2), ("spectral", 1)}
SEED = 20260920
WARNINGS: list[str] = []


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_hash(row: dict) -> dict:
    raw = cv2.imread(row["path"], cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Missing source image: {row['path']}")
    base_path = ROOT / "data" / "processed" / "base" / f"{row['source_id']}.png"
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"Missing standardized image: {base_path}")
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    packed = np.packbits((small >= small.mean()).reshape(-1))
    return {
        **row,
        "raw_pixel_sha256": hashlib.sha256(str(raw.shape).encode() + raw.tobytes()).hexdigest(),
        "base_pixel_sha256": hashlib.sha256(str(base.shape).encode() + base.tobytes()).hexdigest(),
        "ahash": int.from_bytes(packed.tobytes(), "big"),
    }


def clean_sources() -> tuple[pd.DataFrame, dict]:
    sources = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
    print(f"Audit {len(sources)} sources", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashed = pd.DataFrame(pool.map(image_hash, sources.to_dict("records")))
    parents = list(range(len(hashed)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i: int, j: int) -> None:
        left, right = find(i), find(j)
        if left != right:
            parents[max(left, right)] = min(left, right)

    pairs = []
    hashes = hashed["ahash"].tolist()
    raw = hashed["raw_pixel_sha256"].tolist()
    base = hashed["base_pixel_sha256"].tolist()
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            exact = raw[i] == raw[j] or base[i] == base[j]
            if exact or distance <= 2:
                union(i, j)
                pairs.append({
                    "left_source_id": hashed.iloc[i]["source_id"],
                    "right_source_id": hashed.iloc[j]["source_id"],
                    "exact": exact,
                    "ahash_hamming": distance,
                    "cross_split": hashed.iloc[i]["split"] != hashed.iloc[j]["split"],
                    "cross_generator": hashed.iloc[i]["generator"] != hashed.iloc[j]["generator"],
                })
    hashed["duplicate_cluster"] = [str(hashed.iloc[find(i)]["source_id"]) for i in range(len(hashed))]
    priority = {"train": 0, "val": 1, "test": 2}
    hashed["priority"] = hashed["split"].map(priority)
    selected = hashed.sort_values(["priority", "source_id"]).drop_duplicates("duplicate_cluster")
    kept = set(selected["source_id"])
    hashed["retained"] = hashed["source_id"].isin(kept)
    hashed.drop(columns="priority").to_csv(DEST / "source_audit.csv", index=False)
    pd.DataFrame(pairs).to_csv(DEST / "duplicate_pairs.csv", index=False)
    clean = sources.loc[sources["source_id"].isin(kept)].copy()
    clean.to_csv(DEST / "manifest_sources.csv", index=False)
    report = {
        "original_files": len(sources), "retained_sources": len(clean),
        "excluded_sources": len(sources) - len(clean),
        "pairs": len(pairs), "exact_pairs": sum(p["exact"] for p in pairs),
        "cross_split_pairs_before": sum(p["cross_split"] for p in pairs),
        "cross_generator_pairs_before": sum(p["cross_generator"] for p in pairs),
        "cross_split_pairs_after": 0,
        "rule": "Connected components of raw/standardized exact pixel equality or 8x8 mean hash Hamming distance <=2; one representative, priority train then validation then test then lexical source_id. All original splits retained; no score-dependent filtering.",
        "caution": "Average hash similarity is conservative screening; it can cluster distinct photographs with similar layouts. This audit does not establish that every candidate was a semantic duplicate.",
        "sources_by_split_generator": {
            f"{split}|{gen}": int(n)
            for (split, gen), n in clean.groupby(["split", "generator"]).size().items()
        },
    }
    (DEST / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return clean, report


def load_matrix(paths: list[Path], ids: pd.Series) -> np.ndarray:
    lookup = {}
    blocks = []
    offset = 0
    for path in paths:
        values = np.load(path, allow_pickle=True)
        vectors = values["embedding"].astype(np.float32)
        blocks.append(vectors)
        for idx, value in enumerate(values["image_id"].astype(str)):
            if value not in lookup:
                lookup[value] = offset + idx
        offset += len(vectors)
    missing = set(ids.astype(str)) - lookup.keys()
    if missing:
        raise ValueError(f"{len(missing)} missing embeddings; first {sorted(missing)[:3]}")
    all_vectors = np.concatenate(blocks)
    return all_vectors[np.array([lookup[str(i)] for i in ids])]


def fit_model(features: np.ndarray, labels: np.ndarray, name: str) -> LogisticRegression:
    print(f"Fit {name}: {len(labels)} rows", flush=True)
    model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels)
    WARNINGS.extend(f"{name}: {w.message}" for w in found)
    if any(issubclass(w.category, ConvergenceWarning) for w in found):
        raise RuntimeError(f"{name} did not converge; cannot report result")
    np.savez(DEST / f"{name}.npz", coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)
    return model


def summaries(frame: pd.DataFrame, score_columns: dict[str, str], thresholds: dict[str, float], samples: int, outcome: str = "label") -> dict:
    """Joint source bootstrap, stratified by source collection, for all scores.

    Rows from one source are always resampled together, including tasks with
    different outcome labels within a source. ROC points are descriptive.
    Model columns with a corresponding ``*_logit`` column use that linear
    decision score for ranking, avoiding sigmoid saturation. Probabilities
    still determine fixed-threshold errors and fusion inputs.
    """
    frame = frame.reset_index(drop=True)
    labels = frame[outcome].to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Both outcomes required")
    scores = {name: frame[col].to_numpy(dtype=float) for name, col in score_columns.items()}
    ranking = {name: frame[col + "_logit"].to_numpy(dtype=float) if col + "_logit" in frame else scores[name] for name, col in score_columns.items()}
    report = {name: classification_metrics(labels, score, thresholds[name], ranking_scores=ranking[name]) for name, score in scores.items()}
    buckets = []
    for _gen, subset in frame.groupby("generator", sort=True):
        buckets.append([g.index.to_numpy() for _src, g in subset.groupby("source_id", sort=True)])
    draws = {name: [] for name in scores}
    rng = np.random.default_rng(SEED)
    for _ in range(samples):
        indices = np.concatenate([np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for rows in buckets])
        y = labels[indices]
        if np.unique(y).size != 2:
            continue
        for name, score in scores.items():
            s = score[indices]
            fpr, tpr, _ = roc_curve(y, ranking[name][indices], drop_intermediate=False)
            pred = s >= thresholds[name]
            draws[name].append([
                np.trapezoid(tpr, fpr), np.max(tpr[fpr <= .05]),
                pred[y == 0].mean(), pred[y == 1].mean(),
            ])
    for name in report:
        values = np.array(draws[name])
        interval = np.percentile(values, [2.5, 97.5], axis=0)
        for j, key in enumerate(["roc_auc", "tpr_at_fpr_5pct", "false_positive_rate", "recall"]):
            report[name][f"{key}_ci"] = interval[:, j].tolist()
        report[name]["sources"] = int(frame["source_id"].nunique())
    for first, second in [("fusion", "probe"), ("clip_fusion", "clip_probe")]:
        if first in draws and second in draws:
            differences = np.array(draws[first])[:, 0] - np.array(draws[second])[:, 0]
            report[f"{first}_minus_{second}"] = {
                "estimate": report[first]["roc_auc"] - report[second]["roc_auc"],
                "ci95": np.percentile(differences, [2.5, 97.5]).tolist(),
            }
    return report


def summarize_full(test: pd.DataFrame, validation: pd.DataFrame, block: dict,
                   columns: dict[str, str], thresholds: dict[str, float],
                   samples: int, config: dict) -> None:
    """Common analysis for a fresh refit and a saved-fit statistics refresh."""
    active_threshold = thresholds["active"]
    block["conditions"] = {}
    for attack, severity in [("jpeg", 1), ("jpeg", 2), ("blur", 2), ("lowres", 1), ("crop", 1), ("lowres", 2), ("regenerate", 2), ("spectral", 1)]:
        print(f"Bootstrap {attack} {severity}", flush=True)
        sub = test.loc[test["attack"].eq(attack) & test["severity"].eq(severity)]
        marked = sub[sub["group"].eq("S-A")]
        block["conditions"][f"{attack}:{severity}"] = {
            "bit_accuracy": float(marked["active_score"].mean()),
            "verification_rate_at_validation_threshold": float((marked["active_score"] >= active_threshold).mean()),
            "systems": summaries(sub, {k: columns[k] for k in ["active", "probe", "fusion"]}, thresholds, samples),
        }
    dead_mask = np.array([(a, int(s)) in DEAD for a, s in zip(test["attack"], test["severity"])])
    block["removal_slice"] = summaries(test.loc[dead_mask], columns, thresholds, samples)
    for gen in ["stable-diffusion", "dalle-3", "midjourney"]:
        block[f"generator_{gen}"] = summaries(test.loc[test["generator"].isin(["real", gen])], columns, thresholds, samples)
    block["matched_attacks"] = summaries(test.loc[test["group"].isin(["S-A", "R-T"])], columns, thresholds, samples)
    block["clean_unmarked"] = summaries(test.loc[test["group"].isin(["S-U", "R-C"])], {k: columns[k] for k in ["probe", "clip_probe"]}, thresholds, samples)
    attack_thresholds = {name: select_threshold(validation["attack_label"].to_numpy(), validation[name].to_numpy()) for name in ["attack_head", "clip_attack_head"]}
    block["attack_heads"] = summaries(test, {k: k for k in attack_thresholds}, attack_thresholds, samples, "attack_label")
    block["attack_heads_matched"] = summaries(test.loc[test["group"].isin(["S-A", "R-T"])], {k: k for k in attack_thresholds}, attack_thresholds, samples, "attack_label")
    block["payload_by_group"] = {g: {"rows": len(sub), "sources": int(sub["source_id"].nunique()), "accept_rate": float((sub["active_score"] >= active_threshold).mean())} for g, sub in test.groupby("group")}
    validation = validation.rename(columns={"fusion": "fusion_score"})
    policy = fit_decision_policy(validation, **config["decision"])
    test["fusion_score"] = test["fusion"]
    block["decision_validation"] = decision_metrics(validation, policy)
    block["decision"] = decision_metrics(test, policy)
    block["decision_removal_slice"] = decision_metrics(test.loc[dead_mask], policy)
    test["decision"] = policy.decide_many(test["active_score"].to_numpy(), test["fusion"].to_numpy())


def refresh_statistics(samples: int) -> None:
    """Recompute all rankings from saved fits without changing their inputs.

    Linear scores are evaluated in float64 from the exact saved coefficients.
    Re-reading float32 probability columns as float32 preserves fusion inputs.
    """
    import shutil
    report = json.loads((DEST / "metrics.json").read_text(encoding="utf-8"))
    backup = DEST / "metrics_probability_ranking.json"
    if not backup.exists():
        shutil.copyfile(DEST / "metrics.json", backup)
    probability_dtypes = {k: np.float32 for k in ["probe", "clip_probe", "attack_head", "clip_attack_head"]}
    test_full = pd.read_csv(DEST / "test_predictions.csv", keep_default_na=False, dtype=probability_dtypes)
    matrices = {
        "resnet": load_matrix([ROOT / "outputs/resnet_embeddings.npz"], test_full["image_id"]).astype(np.float64),
        "clip": load_matrix([ROOT / "outputs/clip_train_embeddings.npz", ROOT / "outputs/clip_eval_embeddings.npz"], test_full["image_id"]).astype(np.float64),
    }
    validation = pd.read_csv(DEST / "validation_predictions.csv", keep_default_na=False, dtype=probability_dtypes)
    config = yaml.safe_load((DEST / "config_snapshot.yaml").read_text(encoding="utf-8"))
    columns = {"active": "active_score", **{k: k for k in ["probe", "fusion", "clip_probe", "clip_fusion"]}}
    for variant in ("full", "no_attacks", "no_matched_reals"):
        path = DEST / ("test_predictions.csv" if variant == "full" else f"test_predictions_{variant}.csv")
        test = pd.read_csv(path, keep_default_na=False, dtype=probability_dtypes)
        if not test["image_id"].equals(test_full["image_id"]):
            raise ValueError("Saved variants must have identical row ordering")
        for backbone, matrix in matrices.items():
            prefix = "" if backbone == "resnet" else "clip_"
            names = [(prefix + "probe", f"{backbone}_{variant}", matrix),
                     (prefix + "fusion", f"{backbone}_fusion_{variant}", test[[prefix + "probe", "active_score"]].to_numpy(dtype=np.float64))]
            if variant == "full":
                names.append((prefix + "attack_head", f"{backbone}_attack", matrix))
            for name, fit, inputs in names:
                coef = np.load(DEST / f"{fit}.npz")
                test[name + "_logit"] = (inputs @ coef["coefficients"].astype(np.float64).T + coef["intercept"].astype(np.float64)).ravel()
        block = report["variants"][variant]
        thresholds = {name: block["systems"][name]["threshold"] for name in columns}
        print(f"Refresh all statistics: {variant}", flush=True)
        block["systems"] = summaries(test, columns, thresholds, samples)
        if variant == "full":
            summarize_full(test, validation, block, columns, thresholds, samples, config)
        test.to_csv(path, index=False)
        write_json(DEST / "metrics.json", report)
    report["bootstrap_samples"] = samples
    report["ranking_scores"] = "Float64 linear decision scores for every logistic model; raw payload agreement for active. Probabilities, fits, fusion inputs and validation thresholds unchanged."
    write_json(DEST / "metrics.json", report)
    provenance = json.loads((DEST / "provenance.json").read_text(encoding="utf-8"))
    key = str(Path(__file__).relative_to(ROOT))
    old_key = next(k for k in provenance["input_sha256"] if k.replace("\\", "/") == key.replace("\\", "/"))
    provenance.setdefault("original_fit_script_sha256", provenance["input_sha256"][old_key])
    if old_key != key:
        del provenance["input_sha256"][old_key]
    provenance["input_sha256"][key] = digest(Path(__file__))
    for name in ["evaluation.py", "decision.py", "attacks.py"]:
        path = ROOT / "src/aiwstudy" / name
        provenance["input_sha256"][str(path.relative_to(ROOT))] = digest(path)
    provenance["statistics_refresh"] = report["ranking_scores"]
    write_json(DEST / "provenance.json", provenance)
    print("All statistics refreshed from unchanged saved fits", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-audit", action="store_true", help="Reuse the audit produced by this script in the current workspace")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--refresh-statistics", action="store_true", help="Recompute all ROC statistics using saved coefficients and existing predictions; no refitting")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.refresh_statistics:
        refresh_statistics(args.samples)
        return
    if args.reuse_audit:
        clean = pd.read_csv(DEST / "manifest_sources.csv", keep_default_na=False)
        audit = json.loads((DEST / "audit.json").read_text(encoding="utf-8"))
    else:
        clean, audit = clean_sources()
    if args.audit_only:
        return
    config = yaml.safe_load((ROOT / "configs" / "study.yaml").read_text(encoding="utf-8"))
    (DEST / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    frame = pd.read_csv(ROOT / "data" / "manifest_derived.csv", keep_default_na=False, dtype={"image_id": str, "source_id": str, "payload": str})
    frame = frame.loc[frame["source_id"].isin(set(clean["source_id"]))].copy()
    # Original active/CLIP caches do not contain training held-out attacks.
    frame = frame.loc[~((frame["split"].isin(["train", "val"])) & is_held_out_row(frame, config))].reset_index(drop=True)
    active = pd.read_csv(ROOT / "outputs" / "active_scores.csv", dtype={"image_id": str})
    frame = frame.merge(active, on="image_id", how="left", validate="one_to_one")
    if frame["active_score"].isna().any():
        raise ValueError("Missing active scores")
    # Held-out rows omitted for ALL fits, including ablations.
    frame.to_csv(DEST / "manifest_derived.csv", index=False)
    train = frame["split"].eq("train").to_numpy()
    val = frame["split"].eq("val").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    frame["attack_label"] = frame["group"].eq("S-A").astype(int)
    Xr = load_matrix([ROOT / "outputs" / "resnet_embeddings.npz"], frame["image_id"])
    Xc = load_matrix([ROOT / "outputs" / "clip_train_embeddings.npz", ROOT / "outputs" / "clip_eval_embeddings.npz"], frame["image_id"])
    report = {"audit": audit, "bootstrap_samples": args.samples, "bootstrap_unit": "source_id", "bootstrap_strata": "source collection", "exploratory": True, "variants": {}}
    active_threshold = select_threshold(frame.loc[val, "label"].to_numpy(), frame.loc[val, "active_score"].to_numpy())
    for variant in ("full", "no_attacks", "no_matched_reals"):
        selected = train.copy()
        if variant == "no_attacks":
            selected &= ~frame["group"].isin(["R-T", "S-A"]).to_numpy()
        elif variant == "no_matched_reals":
            selected &= ~frame["group"].eq("R-T").to_numpy()
        thresholds = {"active": active_threshold}
        columns = {"active": "active_score"}
        for backbone, matrix in [("resnet", Xr), ("clip", Xc)]:
            prefix = "" if backbone == "resnet" else "clip_"
            name = prefix + "probe"
            p = fit_model(matrix[selected], frame.loc[selected, "label"].to_numpy(), f"{backbone}_{variant}")
            frame[name] = p.predict_proba(matrix)[:, 1]
            frame[name + "_logit"] = p.decision_function(matrix.astype(np.float64))
            fused = prefix + "fusion"
            feats = frame[[name, "active_score"]].to_numpy()
            f = fit_model(feats[val], frame.loc[val, "label"].to_numpy(), f"{backbone}_fusion_{variant}")
            frame[fused] = f.predict_proba(feats)[:, 1]
            frame[fused + "_logit"] = f.decision_function(feats.astype(np.float64))
            for n in [name, fused]:
                thresholds[n] = select_threshold(frame.loc[val, "label"].to_numpy(), frame.loc[val, n].to_numpy())
                columns[n] = n
            if variant == "full":
                hname = prefix + "attack_head"
                h = fit_model(matrix[selected], frame.loc[selected, "attack_label"].to_numpy(), f"{backbone}_attack")
                frame[hname] = h.predict_proba(matrix)[:, 1]
                frame[hname + "_logit"] = h.decision_function(matrix.astype(np.float64))
        test = frame.loc[test_mask].copy()
        print(f"Bootstrap {variant} full test", flush=True)
        block = {"train_rows": int(selected.sum()), "train_sources": int(frame.loc[selected, "source_id"].nunique()), "systems": summaries(test, columns, thresholds, args.samples)}
        if variant == "full":
            summarize_full(test, frame.loc[val], block, columns, thresholds, args.samples, config)
            test.to_csv(DEST / "test_predictions.csv", index=False)
            frame.loc[val].to_csv(DEST / "validation_predictions.csv", index=False)
        else:
            test.to_csv(DEST / f"test_predictions_{variant}.csv", index=False)
        report["variants"][variant] = block
        write_json(DEST / "metrics.json", report)
    quality = pd.read_csv(ROOT / "outputs" / "quality.csv")
    quality = quality[quality["image_id"].isin(set(clean["source_id"]))]
    report["embedding_quality"] = {"mean_psnr": float(quality["psnr"].mean()), "n":len(quality)}
    report["fit_warnings"] = WARNINGS
    report["null_16bit_accept_probability_13bits"] = sum(__import__("math").comb(16, k) for k in range(13, 17)) / 2**16
    report["ranking_scores"] = "Float64 linear decision scores for every logistic model; raw payload agreement for active. Probabilities determine fusion inputs and validation thresholds."
    write_json(DEST / "metrics.json", report)
    files = [ROOT / "data" / "manifest.csv", ROOT / "data" / "manifest_derived.csv", ROOT / "configs" / "study.yaml", Path(__file__), *[ROOT / "src/aiwstudy" / name for name in ["evaluation.py", "decision.py", "attacks.py"]], *[ROOT / "outputs" / name for name in ["resnet_embeddings.npz", "clip_train_embeddings.npz", "clip_eval_embeddings.npz", "active_scores.csv"]]]
    provenance = {"python": sys.version, "platform":platform.platform(), "packages":{name:importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "opencv-python-headless", "PyWavelets", "PyYAML"]}, "input_sha256":{str(p.relative_to(ROOT)):digest(p) for p in files}, "notes":"Frozen embeddings and derived images reused from original run; all logistic probes and fusion refit after source filtering. No backbone fine-tuning. Selection and analyses are retrospective."}
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Done: {DEST}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
