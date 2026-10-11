from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pywt


def held_out_keys(config: dict) -> set[tuple[str, int]]:
    return {
        (str(item["name"]), int(item["severity"]))
        for item in config.get("attacks", [])
        if item.get("hold_out")
    }


def is_held_out_row(frame: pd.DataFrame, config: dict) -> pd.Series:
    keys = held_out_keys(config)
    if not keys:
        return pd.Series(False, index=frame.index)
    return pd.Series(
        [(str(attack), int(severity)) in keys for attack, severity in zip(frame["attack"], frame["severity"])],
        index=frame.index,
    )


def _even(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    even_h, even_w = height - height % 2, width - width % 2
    return image[:even_h, :even_w], (height, width)


def _restore_size(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG attack failed to encode an image.")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _regenerate(image: np.ndarray, severity: int) -> np.ndarray:
    """Break block alignment, then re-encode.

    A low-pass denoiser keeps the low-frequency band where DWT-DCT marks
    live, so it does not remove them. This attack shifts the image off the
    block grid and re-encodes it. It does not read the payload. It is not a
    diffusion img2img attack.
    """
    shift = 3 + 2 * severity
    shifted = np.roll(np.roll(image, shift, axis=0), shift, axis=1)
    quality = max(12, 35 - 8 * severity)
    recoded = _jpeg(shifted, quality)
    rng = np.random.default_rng(20_000 + severity)
    sigma = 8.0 * severity
    noisy = np.clip(recoded.astype(np.float32) + rng.normal(0, sigma, recoded.shape), 0, 255)
    return noisy.astype(np.uint8)


def _spectral(image: np.ndarray, severity: int) -> np.ndarray:
    """Suppress mid-band DCT coefficients inside Haar-LL tiles.

    Block DCT watermarks, including the one in this study, store the payload
    in that band. The attack zeros the band. It does not read the payload.
    """
    cropped, size = _even(image)
    ycc = cv2.cvtColor(cropped, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ll, details = pywt.dwt2(ycc[:, :, 0], "haar")
    for row in range(0, ll.shape[0] - 7, 8):
        for col in range(0, ll.shape[1] - 7, 8):
            block = cv2.dct(ll[row : row + 8, col : col + 8])
            for i in range(8):
                for j in range(8):
                    if 2 <= i + j <= 4 + severity:
                        block[i, j] = 0.0
            ll[row : row + 8, col : col + 8] = cv2.idct(block)
    restored = pywt.idwt2((ll, details), "haar")
    ycc[:, :, 0] = restored[: cropped.shape[0], : cropped.shape[1]]
    result = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return _restore_size(result, size)


def _lowres(image: np.ndarray, severity: int) -> np.ndarray:
    """Downscale and upscale, imitating a low-resolution upload."""
    height, width = image.shape[:2]
    side = 32 if severity >= 2 else 64
    side = max(8, min(side, height, width))
    small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_attack(image: np.ndarray, name: str, severity: int) -> np.ndarray:
    """Apply a deterministic, matched image transformation."""
    if name == "identity" or severity == 0:
        return image.copy()
    severity = max(1, int(severity))
    height, width = image.shape[:2]

    if name == "blur":
        kernel = 2 * severity + 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if name == "noise":
        rng = np.random.default_rng(10_000 + severity)
        sigma = 3.0 + 3.0 * severity
        noise = rng.normal(0, sigma, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == "resize":
        scale = max(0.35, 1.0 - 0.2 * severity)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if name == "crop":
        margin = min(0.4, 0.05 * severity)
        dx, dy = int(width * margin), int(height * margin)
        crop = image[dy : height - dy, dx : width - dx]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    if name == "color":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.0 + 0.08 * severity
        hsv[:, :, 2] *= 1.0 - 0.05 * severity
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if name == "denoise":
        strength = 3 + 2 * severity
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    if name == "jpeg":
        quality = max(20, 90 - 20 * severity)
        return _jpeg(image, quality)
    if name == "inpaint":
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        radius = max(4, min(height, width) // (16 // severity))
        cv2.circle(mask, (width // 2, height // 2), radius, 255, -1)
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    if name == "regenerate":
        return _regenerate(image, severity)
    if name == "spectral":
        return _spectral(image, severity)
    if name == "lowres":
        return _lowres(image, severity)
    raise ValueError(f"Unknown image attack: {name}")
