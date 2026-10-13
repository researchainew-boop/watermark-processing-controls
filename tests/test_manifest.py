import pandas as pd
import pytest

from aiwstudy.manifest import MANIFEST_COLUMNS, assign_splits, validate_manifest
from aiwstudy.pipeline import humanify_indicator


def test_humanify_label_is_stripped_ai_only():
    groups = pd.Series(["R-C", "R-T", "R-S", "S-U", "S-W", "S-A"])
    assert humanify_indicator(groups).tolist() == [0, 0, 0, 0, 0, 1]


def _manifest(count=20):
    records = []
    for index in range(count):
        label = index % 2
        records.append(
            {
                "image_id": f"i{index}",
                "source_id": f"s{index}",
                "path": f"{index}.png",
                "label": label,
                "generator": "real" if not label else ("genA" if index < 10 else "genB"),
                "split": "",
                "group": "R-C" if not label else "S-U",
                "watermark_method": "",
                "payload": "",
                "attack": "identity",
                "severity": 0,
                "parent_id": "",
            }
        )
    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def test_group_split_has_no_source_leakage():
    result = assign_splits(_manifest(), 0.7, 0.15, 7)
    validate_manifest(result)
    assert set(result["split"]) == {"train", "val", "test"}


def test_heldout_generator_is_test_only():
    result = assign_splits(_manifest(), 0.6, 0.2, 7, heldout_generator="genB")
    heldout = result[(result["label"] == 1) & (result["generator"] == "genB")]
    assert not heldout.empty
    assert set(heldout["split"]) == {"test"}
    validate_manifest(result)


def test_validate_rejects_derivative_leakage():
    frame = _manifest(4)
    frame.loc[1, "source_id"] = frame.loc[0, "source_id"]
    frame.loc[0, "split"] = "train"
    frame.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(frame)
