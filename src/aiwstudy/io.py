from __future__ import annotations

import cv2
import numpy as np


def read_image(path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot open image: {path}")
    return image


def write_image(path, image: np.ndarray) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Cannot write image: {path}")


def standardize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def image_quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    result = {
        "psnr": float(cv2.PSNR(reference, candidate)),
        "mean_absolute_error": float(
            np.mean(np.abs(reference.astype(np.float32) - candidate.astype(np.float32)))
        ),
    }
    try:
        from skimage.metrics import structural_similarity

        result["ssim"] = float(
            structural_similarity(
                cv2.cvtColor(reference, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB),
                channel_axis=2,
                data_range=255,
            )
        )
    except ImportError:
        result["ssim"] = float("nan")
    return result
