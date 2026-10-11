"""Extract only the prospectively locked missing calibration CLIP vectors."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_encoding_controls import (
    ROOT, checkpoint_defaults, load_backbone, infer, digest,
    cached_features, original_caches,
)


def main():
    import torch
    torch.set_num_threads(2)
    out = ROOT / "outputs/phase5/calibration"
    manifest = out / "calibration_extra_manifest.csv"
    frame = pd.read_csv(manifest, keep_default_na=False)
    saved = out / "clip_calibration_extra_embeddings.npz"
    if saved.exists():
        with np.load(saved, allow_pickle=True) as z:
            assert str(z["manifest_sha256"]) == digest(manifest)
            assert np.array_equal(z["image_id"].astype(str), frame.image_id)
        print("Reused locked calibration embeddings", flush=True)
        return
    for row in frame.to_dict("records"):
        assert digest(Path(row["path"])) == row["output_file_sha256"]
    anchors = pd.read_csv(ROOT / "outputs/conference_clean/manifest_derived.csv", keep_default_na=False)
    anchors = anchors.loc[anchors.group.isin(["R-C", "S-U"])].groupby("generator").head(1)
    _, checkpoint = checkpoint_defaults()
    model, transform, forward, preproc, identity = load_backbone("clip", "cuda", checkpoint)
    current = infer(anchors, transform, forward, "cuda", 8).astype(np.float64)
    old = cached_features(original_caches()[1:], anchors.image_id.tolist()).astype(np.float64)
    cosine = np.sum(current * old, axis=1) / (np.linalg.norm(current, axis=1) * np.linalg.norm(old, axis=1))
    relative = np.linalg.norm(current - old, axis=1) / np.linalg.norm(old, axis=1)
    assert cosine.min() > .99999 and relative.max() < .005
    started = datetime.now(timezone.utc).isoformat()
    info = {"started_at_utc": started, "manifest_sha256": digest(manifest),
            "protocol_sha256": digest(out / "protocol.json"), "images": len(frame),
            "device": "cuda", "checkpoint_sha256": digest(checkpoint),
            "model": identity, "preprocessing": preproc,
            "anchor_cosine": cosine.tolist(), "anchor_relative_l2": relative.tolist()}
    (out / "clip_calibration_extra_inference_started.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    vectors = infer(frame, transform, forward, "cuda", 8)
    np.savez(saved, image_id=frame.image_id.to_numpy(), embedding=vectors, manifest_sha256=digest(manifest))
    info.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), embedding_sha256=digest(saved), script_sha256=digest(Path(__file__)))
    (out / "clip_calibration_extra_inference.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    del model, forward, transform, vectors
    gc.collect()
    torch.cuda.empty_cache()
    print("Completed locked calibration CLIP extraction", flush=True)


if __name__ == "__main__":
    main()
