import numpy as np

from aiwstudy.watermark import DwtDctWatermark, DwtSvdWatermark


def _textured() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)


def test_dwt_dct_round_trip():
    image = _textured()
    marker = DwtDctWatermark("10110100", strength=24)
    marked = marker.embed(image)
    detection = marker.detect(marked)
    assert detection.bit_accuracy >= 0.9
    assert detection.verified


def test_dwt_svd_round_trip():
    image = _textured()
    marker = DwtSvdWatermark("10110100", strength=24)
    marked = marker.embed(image)
    detection = marker.detect(marked)
    assert detection.bit_accuracy >= 0.75
