from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = [
    "image_id",
    "source_id",
    "path",
    "label",
    "generator",
    "split",
    "group",
    "watermark_method",
    "payload",
    "attack",
    "severity",
    "parent_id",
]


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _records(
    directory: Path,
    label: int,
    extensions: set[str],
    default_generator: str,
):
    if not directory.exists():
        return
    for path in sorted(p for p in directory.rglob("*") if p.suffix.lower() in extensions):
        relative = path.relative_to(directory).as_posix()
        stem_parts = path.stem.rsplit("__", 1)
        generator = default_generator
        if label == 1 and len(stem_parts) == 2:
            generator = stem_parts[1]
        image_id = stable_id(f"{label}:{relative}")
        yield {
            "image_id": image_id,
            "source_id": image_id,
            "path": str(path.resolve()),
            "label": label,
            "generator": "real" if label == 0 else generator,
            "split": "",
            "group": "R-C" if label == 0 else "S-U",
            "watermark_method": "",
            "payload": "",
            "attack": "identity",
            "severity": 0,
            "parent_id": "",
        }


def scan_manifest(
    real_dir: Path,
    synthetic_dir: Path,
    extensions: list[str],
    default_generator: str = "unknown",
) -> pd.DataFrame:
    allowed = {ext.lower() for ext in extensions}
    records = list(_records(real_dir, 0, allowed, default_generator))
    records.extend(_records(synthetic_dir, 1, allowed, default_generator))
    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def assign_splits(
    frame: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    heldout_generator: str | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = ""
    heldout = pd.Series(False, index=result.index)
    if heldout_generator:
        heldout = (result["label"] == 1) & (result["generator"] == heldout_generator)
        result.loc[heldout, "split"] = "test"

    rng = np.random.default_rng(seed)
    for label in sorted(result.loc[~heldout, "label"].unique()):
        candidates = result.loc[(~heldout) & (result["label"] == label)]
        groups = candidates["source_id"].unique()
        has_forced_test = bool((heldout & (result["label"] == label)).any())
        minimum = 2 if has_forced_test else 3
        if len(groups) < minimum:
            raise ValueError(
                f"Class {label} requires at least {minimum} independent source images."
            )
        groups = rng.permutation(groups)
        reserve_test = 0 if has_forced_test else 1
        train_count = max(1, round(train_fraction * len(groups)))
        val_count = max(1, round(val_fraction * len(groups)))
        while train_count + val_count > len(groups) - reserve_test:
            if train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break
        assignments = {
            "train": groups[:train_count],
            "val": groups[train_count : train_count + val_count],
            "test": groups[train_count + val_count :],
        }
        for split, split_groups in assignments.items():
            selected = candidates["source_id"].isin(split_groups)
            result.loc[candidates.index[selected], "split"] = split
    return result


def validate_manifest(frame: pd.DataFrame) -> None:
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    leakage = frame.groupby("source_id")["split"].nunique()
    if (leakage > 1).any():
        bad = leakage[leakage > 1].index.tolist()[:5]
        raise ValueError(f"Source leakage across splits: {bad}")
