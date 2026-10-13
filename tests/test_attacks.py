import numpy as np
import pandas as pd

from aiwstudy.attacks import apply_attack, held_out_keys
from aiwstudy.quality import degradation_features, degraded_indicator
from aiwstudy.watermark import DwtDctWatermark


def test_attacks_preserve_shape():
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    for name in ("jpeg", "blur", "noise", "resize", "crop", "color", "inpaint", "regenerate", "spectral", "lowres"):
        attacked = apply_attack(image, name, 1)
        assert attacked.shape == image.shape
        assert attacked.dtype == np.uint8


def test_regeneration_lowers_payload_accuracy():
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    marker = DwtDctWatermark("10110100", strength=24)
    marked = marker.embed(image)
    clean = marker.detect(marked).bit_accuracy
    killed = marker.detect(apply_attack(marked, "regenerate", 2)).bit_accuracy
    assert clean >= 0.9
    assert killed < clean


def test_spectral_band_erasure_drops_payload_near_chance():
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    marker = DwtDctWatermark("10110100", strength=24)
    marked = marker.embed(image)
    killed = marker.detect(apply_attack(marked, "spectral", 1)).bit_accuracy
    assert killed <= 0.65


def test_blur_and_lowres_score_softer_than_a_sharp_image():
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    blurry = apply_attack(sharp, "blur", 2)
    small = apply_attack(sharp, "lowres", 2)
    sharp_score = degradation_features(sharp)[0]
    assert degradation_features(blurry)[0] < sharp_score
    assert degradation_features(small)[0] < sharp_score
    labels = degraded_indicator(pd.Series(["jpeg", "blur", "lowres", "identity"]))
    assert labels.tolist() == [0, 1, 1, 0]


def test_held_out_keys_read_config_flag():
    keys = held_out_keys(
        {"attacks": [{"name": "jpeg", "severity": 1}, {"name": "regenerate", "severity": 1, "hold_out": True}]}
    )
    assert keys == {("regenerate", 1)}
