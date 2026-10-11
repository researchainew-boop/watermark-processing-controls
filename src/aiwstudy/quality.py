from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

DEGRADED_ATTACKS = {"blur", "lowres"}


def degraded_indicator(attacks: pd.Series) -> np.ndarray:
    """1 when the image was blurred or reduced to a low resolution."""
    return attacks.astype(str).isin(DEGRADED_ATTACKS).to_numpy(dtype=int)


def degradation_features(image: np.ndarray) -> np.ndarray:
    """Sharpness measures. Low values mean blur or a low-resolution upscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = float(np.mean(np.abs(gray.astype(np.float32) - smooth.astype(np.float32))))
    return np.array([np.log1p(laplacian), residual], dtype=np.float64)


def degradation_matrix(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([degradation_features(image) for image in images])
