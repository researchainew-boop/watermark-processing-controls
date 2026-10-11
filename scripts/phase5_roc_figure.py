"""Plot the locked caption-paired identity test; no fitting or inference.

Run after the paired-content analysis has written its completion lock. The
previous manuscript ROC assets are archived before the approved filenames are
replaced. Raw predictions remain private; only hashes and aggregate provenance
are suitable for the reviewer package.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase5/paired_content"
ASSETS = ROOT / "manuscript/assets"
ARCHIVE = ROOT / "manuscript/archive"
SCOPE = "phase5_caption_paired_identity"
LEFT_EDGE = 0.001
SYSTEMS = [
    ("joint_resnet_probe_logit", "resnet", "probe", "ResNet probe", "#222222", "-", 1.8),
    ("joint_resnet_fusion_logit", "resnet", "fusion", "ResNet fusion", "#d55e00", "--", 1.3),
    ("joint_resnet_gated_logit", "resnet", "gated", "ResNet gate", "#009e73", ":", 1.2),
    ("joint_clip_probe_logit", "clip", "probe", "CLIP probe", "#0072b2", "-.", 1.5),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_locked_scores() -> tuple[list[dict], dict, dict]:
    complete = read_json(OUT / "complete.json")
    if (complete.get("pairs"), complete.get("sources"), complete.get("copies")) != (200, 400, 1200):
        raise ValueError("Expected the completed 200-pair / 400-source / 1200-copy test")
    if complete.get("test_refitting") is not False or complete.get("test_threshold_tuning") is not False:
        raise ValueError("Paired test completion does not assert frozen fitting and thresholds")
    for name, expected in complete["output_sha256"].items():
        if digest(OUT / name) != expected:
            raise ValueError(f"Completed paired output changed: {name}")
    runtime = read_json(OUT / "runtime_protocol.json")
    if runtime.get("test_threshold_tuning") is not False:
        raise ValueError("Runtime protocol does not lock test thresholds")
    for name, expected in runtime["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Runtime-locked input changed: {name}")
    manifest = read_csv(OUT / "manifest.csv")
    predictions = read_csv(OUT / "predictions.csv")
    manifest_by_id = {row["image_id"]: row for row in manifest}
    prediction_by_id = {row["image_id"]: row for row in predictions}
    if len(manifest_by_id) != 1200 or len(prediction_by_id) != 1200 or len(predictions) != 1200:
        raise ValueError("Expected one saved prediction per locked image")
    if set(manifest_by_id) != set(prediction_by_id):
        raise ValueError("Saved prediction IDs differ from the locked manifest")
    for iid, expected in manifest_by_id.items():
        if any(prediction_by_id[iid].get(name) != value for name, value in expected.items()):
            raise ValueError(f"Saved prediction metadata differs from the locked manifest: {iid}")
    rows = [row for row in predictions if row["attack"] == "identity"]
    if len(rows) != 400 or len({row["source_id"] for row in rows}) != 400:
        raise ValueError("Identity scope is not 400 unique sources")
    pairs = {row["pair_id"] for row in rows}
    if len(pairs) != 200 or any(sorted(int(row["label"]) for row in rows if row["pair_id"] == pair) != [0, 1]
                               for pair in pairs):
        raise ValueError("Identity scope is not 200 complete real/generated pairs")
    if any(row["mark"] != "unmarked" or row["severity"] != "0" for row in rows):
        raise ValueError("Identity figure must contain only unmarked, unedited test copies")
    for row in rows:
        for column, *_ in SYSTEMS:
            if not np.isfinite(float(row[column])):
                raise ValueError("Saved score is not finite")
        expected = row["joint_resnet_fusion_logit"] if float(row["active_score"]) >= 13 / 16 else row["joint_resnet_probe_logit"]
        if float(row["joint_resnet_gated_logit"]) != float(expected):
            raise ValueError("Saved gate does not follow the locked 13-of-16 rule")
    summary = read_json(OUT / "summary.json")
    input_paths = [OUT / name for name in
                   ("predictions.csv", "manifest.csv", "summary.json", "metrics.csv", "complete.json",
                    "runtime_protocol.json", "resnet_inference.json", "clip_inference.json")]
    input_hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in input_paths}
    head_hashes = {name: runtime["input_sha256"][name]
                   for key in ("joint_resnet", "joint_clip") for name in runtime["heads"][key]}
    return rows, summary, {"input_sha256": input_hashes, "locked_head_sha256": head_hashes}


def preserve_old_assets() -> dict:
    names = ("resnet_roc.pdf", "resnet_roc.png", "roc_provenance.json")
    previous = read_json(ASSETS / names[2])
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if previous.get("evaluation_scope") != SCOPE:
        for name in names:
            source = ASSETS / name
            target = ARCHIVE / ("pre_phase5_" + name)
            if target.exists() and digest(target) != digest(source):
                raise ValueError(f"An incompatible previous asset already occupies {target.name}")
            if not target.exists():
                shutil.copyfile(source, target)
    archived = {}
    for name in names:
        path = ARCHIVE / ("pre_phase5_" + name)
        if not path.is_file():
            raise ValueError("Previous ROC assets must be preserved before replacement")
        archived[path.relative_to(ROOT).as_posix()] = digest(path)
    return archived


def main() -> None:
    rows, summary, locks = verify_locked_scores()
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    curves, aucs, score_columns = {}, {}, {}
    for column, backbone, system, label, *_ in SYSTEMS:
        scores = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        measured_auc = float(roc_auc_score(labels, scores))
        records = [record for record in summary["all_metrics"]
                   if record["regime"] == "joint" and record["backbone"] == backbone
                   and record["system"] == system and record["condition"] == "identity"
                   and record["policy"] == "per_image_core13"]
        if len(records) != 1 or abs(measured_auc - records[0]["AUC"]) > 1e-12:
            raise ValueError(f"Saved-score AUC disagrees with the completed aggregate: {column}")
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        curves[label] = (fpr, tpr)
        aucs[label] = measured_auc
        score_columns[label] = column
    archived = preserve_old_assets()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    # Fixed page dimensions and a reserved note row preserve legibility at IEEE column width.
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.25, top=0.985)
    ax.set_xscale("log")
    ax.set_xlim(LEFT_EDGE, 1)
    ax.set_ylim(0, 1.02)
    ax.axvline(0.05, color="0.6", linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    ax.text(0.05, 0.545, "5%", fontsize=8, color="0.35", ha="center", va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    for _, _, _, label, color, linestyle, width in SYSTEMS:
        fpr, tpr = curves[label]
        ax.step(np.maximum(fpr, LEFT_EDGE), tpr, where="post", color=color,
                linestyle=linestyle, linewidth=width, label=f"{label}: AUC {aucs[label]:.3f}", zorder=3)
    ax.set_xlabel("False-positive rate (log scale)", labelpad=2)
    ax.set_ylabel("True-positive rate", labelpad=2)
    ticks = [0.001, 0.01, 0.1, 1.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.001", "0.01", "0.1", "1"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0", "0.25", "0.50", "0.75", "1.0"]))
    ax.tick_params(axis="both", which="major", length=3, width=0.7, pad=2)
    ax.grid(axis="y", linewidth=0.5, color="0.9", zorder=0)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="0.8",
              framealpha=0.98, handlelength=2.5, handletextpad=0.5, borderpad=0.45,
              labelspacing=0.35)
    fig.text(0.57, 0.022, "FPR = 0 is drawn at the left edge.", ha="center", va="bottom", fontsize=8)
    pdf, png = ASSETS / "resnet_roc.pdf", ASSETS / "resnet_roc.png"
    fig.savefig(pdf, metadata={"Title": "Caption-paired identity test ROC", "Author": "",
                              "Creator": "Study plotting script", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=320)
    plt.close(fig)
    identical_curves = [[first, second] for index, first in enumerate(curves)
                        for second in list(curves)[index + 1:]
                        if all(np.array_equal(a, b) for a, b in zip(curves[first], curves[second]))]
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": SCOPE, "condition": "identity", "split": "test",
        "training_regime": "joint", "rows": len(rows), "sources": len(rows), "pairs": 200,
        "real_sources": int(np.sum(labels == 0)), "generated_sources": int(np.sum(labels == 1)),
        "unmarked_only": True, "saved_scores_only": True, "refits": False, "inference": False,
        "test_threshold_tuning": False, "curve_kind": "empirical step ROC, all thresholds retained",
        "auc_computation": "ROC-AUC on raw saved logits, checked against locked summary",
        "auc": aucs, "score_columns": score_columns,
        "display": {"width_inches": 3.5, "height_inches": 2.4, "minimum_font_points": 8,
                    "FPR_scale": "log", "FPR_left_edge": LEFT_EDGE,
                    "zero_FPR_note": "FPR = 0 is drawn at the left edge.",
                    "smallest_nonzero_empirical_FPR": 1 / int(np.sum(labels == 0)),
                    "reference_FPR": 0.05, "threshold_selection": False},
        "identical_empirical_ROC_curves": identical_curves,
        "script_sha256": digest(Path(__file__)), **locks,
        "archived_asset_sha256": archived,
        "output_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in (pdf, png)},
    }
    (ASSETS / "roc_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"scope": SCOPE, "rows": len(rows), "pairs": 200, "auc": aucs,
                      "identical_curves": identical_curves}, indent=2))


if __name__ == "__main__":
    main()
