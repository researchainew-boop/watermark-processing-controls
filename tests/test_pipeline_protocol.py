import pandas as pd

from aiwstudy.pipeline import _removal_attack_mask, _training_rows


def test_all_training_variants_exclude_held_out_conditions():
    frame = pd.DataFrame(
        [
            {"split": "train", "group": "S-A", "attack": "lowres", "severity": 2},
            {"split": "train", "group": "R-T", "attack": "lowres", "severity": 2},
            {"split": "train", "group": "S-A", "attack": "jpeg", "severity": 1},
            {"split": "train", "group": "R-T", "attack": "jpeg", "severity": 1},
            {"split": "train", "group": "S-U", "attack": "identity", "severity": 0},
            {"split": "val", "group": "S-U", "attack": "identity", "severity": 0},
        ]
    )
    config = {"attacks": [{"name": "lowres", "severity": 2, "hold_out": True}]}
    assert set(_training_rows(config, frame, "full").index) == {2, 3, 4}
    assert set(_training_rows(config, frame, "no-matched-controls").index) == {2, 4}
    assert set(_training_rows(config, frame, "no-attacks").index) == {4}


def test_removal_conditions_include_crop_and_only_lowres_severity_two():
    frame = pd.DataFrame(
        {"attack": ["crop", "lowres", "lowres", "regenerate", "spectral", "jpeg"],
         "severity": [1, 1, 2, 2, 1, 1]},
        index=[10, 11, 12, 13, 14, 15],
    )
    assert list(frame.index[_removal_attack_mask(frame, {})]) == [10, 12, 13, 14]
    config = {"evaluation": {"removal_attacks": [{"name": "crop", "severity": 1}]}}
    assert list(frame.index[_removal_attack_mask(frame, config)]) == [10]
