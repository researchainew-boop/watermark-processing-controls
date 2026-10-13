"""Synthetic receipt, query-family and paired-cover guard failures only."""
import json
from pathlib import Path

import pandas as pd
import pytest

import phase6_attack_reporting as reporting

QUERIES = ["0100101101001011", "1011010010110100", "1100101000110101"]


def setup_inputs(tmp_path, monkeypatch, role="test"):
    root, out = tmp_path, tmp_path / "outputs" / "phase6"
    directory = out / f"{role}_attacks"
    directory.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts" / "phase6_features.py").write_text("frozen synthetic feature driver")
    (out / "frozen_protocol.json").write_text(json.dumps({"registered_payloads": QUERIES}))
    inventory = pd.DataFrame([
        {"source_id": "r", "role": role, "collection": "coco", "label": 0},
        {"source_id": "a", "role": role, "collection": "BigGAN" if role == "test" else "openimages", "label": int(role == "test")},
    ])
    (out / "data_inventory").mkdir()
    inventory.to_csv(out / "data_inventory" / "primary_source_manifest.csv", index=False)
    monkeypatch.setattr(reporting, "ROOT", root)
    monkeypatch.setattr(reporting, "OUT", out)
    monkeypatch.setattr(reporting, "DEST", out / "attack_analysis")
    monkeypatch.setattr(reporting, "attack_subset", lambda stage: {"r", "a"})
    manifest, features = [], []
    for source in inventory.to_dict("records"):
        covers = [("unmarked", "")]
        if source["label"]:
            covers += [(method, query) for method in reporting.f.METHODS for query in QUERIES]
        for method, payload in covers:
            for condition in reporting.CONDITIONS:
                image_id = f"{source['source_id']}|{method}|{payload}|{condition}"
                record = {**source, "image_id": image_id, "watermark_method": method,
                          "embedded_payload": payload, "condition": condition}
                manifest.append(record)
                for detector in reporting.f.METHODS:
                    if method not in ("unmarked", detector):
                        continue
                    for query in QUERIES:
                        features.append({**record, "detector": detector, "query_payload": query,
                            "resnet_logit": float(2 * source["label"] - 1), "clip_logit": 0.0,
                            **{name: 0.5 for name in reporting.f.FEATURES}})
    manifest, features = pd.DataFrame(manifest), pd.DataFrame(features)

    def save(m=manifest, f=features):
        m.to_csv(directory / "manifest.csv", index=False)
        f.to_csv(directory / "features.csv", index=False)
        receipt = {"features_sha256": reporting.digest(directory / "features.csv"),
            "manifest_sha256": reporting.digest(directory / "manifest.csv"),
            "protocol_sha256_before_scores": reporting.digest(out / "frozen_protocol.json"),
            "driver_sha256": reporting.digest(root / "scripts" / "phase6_features.py")}
        (directory / "features_complete.json").write_text(json.dumps(receipt))
    save()
    return directory, manifest, features, save


def setup_cut_receipt(tmp_path, monkeypatch):
    directory, _, _, _ = setup_inputs(tmp_path, monkeypatch, "calibration")
    folder = reporting.OUT / "fusion_v2"
    folder.mkdir()
    (folder / "fit_complete.json").write_text("frozen synthetic fit receipt")
    reporting.DEST.mkdir()
    dependencies = reporting.calibration_dependencies(folder, directory)
    cuts = {"inputs": dependencies, "policies": {"synthetic": "unchanged"}}
    path = reporting.DEST / "secondary_cuts_locked.json"
    path.write_text(json.dumps(cuts))
    receipt = {"inputs": dependencies, "secondary_cuts_sha256": reporting.digest(path)}
    (reporting.DEST / "calibration_complete.json").write_text(json.dumps(receipt))
    return folder, cuts, path


def test_fixed_attack_family_input_accepts_complete_paired_covers(tmp_path, monkeypatch):
    directory, manifest, _, _ = setup_inputs(tmp_path, monkeypatch)
    frame, queries, found = reporting.inputs("test")
    assert found == directory and queries == QUERIES
    assert len(manifest) == 72 and len(frame) == 270


@pytest.mark.parametrize("key", ["features_sha256", "manifest_sha256", "protocol_sha256_before_scores", "driver_sha256"])
def test_attack_input_refuses_changed_receipts(tmp_path, monkeypatch, key):
    directory, _, _, _ = setup_inputs(tmp_path, monkeypatch)
    path = directory / "features_complete.json"
    receipt = json.loads(path.read_text())
    receipt[key] = "changed"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="receipt mismatch|different frozen inference dependencies"):
        reporting.inputs("test")


def test_attack_input_refuses_incomplete_cover_even_if_global_conditions_remain(tmp_path, monkeypatch):
    _, manifest, features, save = setup_inputs(tmp_path, monkeypatch)
    image_id = manifest.iloc[0].image_id
    save(manifest.loc[manifest.image_id.ne(image_id)], features.loc[features.image_id.ne(image_id)])
    with pytest.raises(RuntimeError, match="Every secondary cover"):
        reporting.inputs("test")


def test_attack_input_refuses_missing_decoder_for_unmarked_cover(tmp_path, monkeypatch):
    _, manifest, features, save = setup_inputs(tmp_path, monkeypatch)
    image_id = manifest.iloc[0].image_id
    bad = features.loc[~(features.image_id.eq(image_id) & features.detector.eq("trustmark"))]
    save(manifest, bad)
    with pytest.raises(RuntimeError, match="decoder coverage"):
        reporting.inputs("test")


def test_attack_input_refuses_changed_registered_query_family(tmp_path, monkeypatch):
    _, manifest, features, save = setup_inputs(tmp_path, monkeypatch)
    bad = features.copy()
    bad["query_payload"] = bad.query_payload.replace({QUERIES[0]: "1111111111111111"})
    bad["embedded_payload"] = bad.embedded_payload.replace({QUERIES[0]: "1111111111111111"})
    save(manifest, bad)
    with pytest.raises(ValueError, match="Registered query set changed"):
        reporting.inputs("test")


def test_attack_input_refuses_registered_source_metadata_change(tmp_path, monkeypatch):
    _, manifest, features, save = setup_inputs(tmp_path, monkeypatch)
    bad = features.copy()
    bad.loc[bad.source_id.eq("r"), "collection"] = "foreign_photographs"
    save(manifest, bad)
    with pytest.raises(RuntimeError, match="metadata differ"):
        reporting.inputs("test")


def test_secondary_cut_receipt_accepts_unchanged_policies(tmp_path, monkeypatch):
    folder, cuts, _ = setup_cut_receipt(tmp_path, monkeypatch)
    reporting.verify_cut_receipt(cuts, folder)


def test_secondary_cut_receipt_refuses_policy_changes(tmp_path, monkeypatch):
    folder, cuts, path = setup_cut_receipt(tmp_path, monkeypatch)
    cuts["policies"]["synthetic"] = "changed"
    path.write_text(json.dumps(cuts))
    with pytest.raises(RuntimeError, match="calibrated policy changed"):
        reporting.verify_cut_receipt(cuts, folder)


def test_secondary_cut_receipt_refuses_dependency_changes(tmp_path, monkeypatch):
    folder, cuts, _ = setup_cut_receipt(tmp_path, monkeypatch)
    (folder / "fit_complete.json").write_text("different fit")
    with pytest.raises(RuntimeError, match="calibration dependencies changed"):
        reporting.verify_cut_receipt(cuts, folder)


def test_secondary_calibration_resume_never_rescores(tmp_path, monkeypatch):
    folder, _, path = setup_cut_receipt(tmp_path, monkeypatch)
    original = path.read_bytes()
    monkeypatch.setattr(reporting, "check_final_lock", lambda role: None)
    monkeypatch.setattr(reporting, "learner", lambda: (folder, {}, {}))

    def forbidden(*args):
        raise AssertionError("Completed secondary calibration must not score again.")

    monkeypatch.setattr(reporting.f, "score", forbidden)
    reporting.calibrate()
    assert path.read_bytes() == original
