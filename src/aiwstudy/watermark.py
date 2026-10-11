from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pywt


def payload_array(payload: str) -> np.ndarray:
    if not payload or set(payload) - {"0", "1"}:
        raise ValueError("Payload must be a non-empty binary string.")
    return np.asarray([int(bit) for bit in payload], dtype=np.uint8)


@dataclass(frozen=True)
class Detection:
    bits: str
    bit_accuracy: float
    confidence: float
    verified: bool


def _luminance(image: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return ycc, ycc[:, :, 0]


def _restore(ycc: np.ndarray, luminance: np.ndarray, image: np.ndarray) -> np.ndarray:
    ycc[:, :, 0] = luminance[: image.shape[0], : image.shape[1]]
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


class DwtDctWatermark:
    """Blind image watermark using DCT coefficient ordering in DWT LL blocks."""

    def __init__(self, payload: str, strength: float = 12.0):
        self.expected = payload_array(payload)
        self.strength = float(strength)

    def embed(self, image: np.ndarray) -> np.ndarray:
        ycc, y = _luminance(image)
        ll, (lh, hl, hh) = pywt.dwt2(y, "haar")
        index = 0
        for row in range(0, ll.shape[0] - 7, 8):
            for col in range(0, ll.shape[1] - 7, 8):
                block = cv2.dct(ll[row : row + 8, col : col + 8])
                bit = self.expected[index % len(self.expected)]
                a, b = float(block[3, 2]), float(block[2, 3])
                midpoint = (a + b) / 2.0
                signed = self.strength if bit else -self.strength
                block[3, 2] = midpoint + signed / 2.0
                block[2, 3] = midpoint - signed / 2.0
                ll[row : row + 8, col : col + 8] = cv2.idct(block)
                index += 1
        restored = pywt.idwt2((ll, (lh, hl, hh)), "haar")
        return _restore(ycc, restored, image)

    def scores(self, image: np.ndarray) -> np.ndarray:
        y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        ll, _ = pywt.dwt2(y, "haar")
        scores: list[list[float]] = [[] for _ in self.expected]
        index = 0
        for row in range(0, ll.shape[0] - 7, 8):
            for col in range(0, ll.shape[1] - 7, 8):
                block = cv2.dct(ll[row : row + 8, col : col + 8])
                scores[index % len(self.expected)].append(float(block[3, 2] - block[2, 3]))
                index += 1
        return np.asarray([np.median(values) for values in scores], dtype=np.float32)

    def detect(self, image: np.ndarray, threshold: float = 0.75) -> Detection:
        scores = self.scores(image)
        decoded = (scores >= 0).astype(np.uint8)
        accuracy = float(np.mean(decoded == self.expected))
        confidence = float(np.mean(np.abs(scores)) / (np.mean(np.abs(scores)) + self.strength))
        return Detection(
            bits="".join(map(str, decoded.tolist())),
            bit_accuracy=accuracy,
            confidence=confidence,
            verified=accuracy >= threshold,
        )


class DwtSvdWatermark:
    """Blind DWT-SVD watermark that quantizes the largest singular value of LL tiles."""

    def __init__(self, payload: str, strength: float = 12.0):
        self.expected = payload_array(payload)
        self.strength = float(strength)

    def embed(self, image: np.ndarray) -> np.ndarray:
        ycc, y = _luminance(image)
        ll, details = pywt.dwt2(y, "haar")
        index = 0
        for row in range(0, ll.shape[0] - 7, 8):
            for col in range(0, ll.shape[1] - 7, 8):
                block = ll[row : row + 8, col : col + 8]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                bit = int(self.expected[index % len(self.expected)])
                step = self.strength
                bin_index = np.floor(s[0] / step)
                if int(bin_index) % 2 != bit:
                    bin_index += 1
                s[0] = (bin_index + 0.5) * step
                ll[row : row + 8, col : col + 8] = u @ np.diag(s) @ vt
                index += 1
        restored = pywt.idwt2((ll, details), "haar")
        return _restore(ycc, restored, image)

    def scores(self, image: np.ndarray) -> np.ndarray:
        y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        ll, _ = pywt.dwt2(y, "haar")
        scores: list[list[float]] = [[] for _ in self.expected]
        index = 0
        step = self.strength
        for row in range(0, ll.shape[0] - 7, 8):
            for col in range(0, ll.shape[1] - 7, 8):
                _, s, _ = np.linalg.svd(ll[row : row + 8, col : col + 8], full_matrices=False)
                decoded_bit = int(np.floor(s[0] / step)) % 2
                scores[index % len(self.expected)].append(1.0 if decoded_bit else -1.0)
                index += 1
        return np.asarray([np.median(values) for values in scores], dtype=np.float32)

    def detect(self, image: np.ndarray, threshold: float = 0.75) -> Detection:
        scores = self.scores(image)
        decoded = (scores >= 0).astype(np.uint8)
        accuracy = float(np.mean(decoded == self.expected))
        confidence = float(np.mean(np.abs(scores)) / (np.mean(np.abs(scores)) + self.strength / 4.0))
        return Detection(
            bits="".join(map(str, decoded.tolist())),
            bit_accuracy=accuracy,
            confidence=confidence,
            verified=accuracy >= threshold,
        )


def build_watermarker(method: str, payload: str, strength: float):
    if method == "dwt_dct":
        return DwtDctWatermark(payload, strength)
    if method == "dwt_svd":
        return DwtSvdWatermark(payload, strength)
    raise ValueError(f"Unknown watermark method: {method}")
