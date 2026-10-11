"""Create tiny procedural images for pipeline testing only, not paper results."""

from pathlib import Path

import cv2
import numpy as np


def make_image(seed: int, synthetic: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:128, 0:128]
    image = np.stack(
        (
            40 + (x * 0.8).astype(np.uint8),
            80 + (y * 0.6).astype(np.uint8),
            120 + ((x + y) * 0.2).astype(np.uint8),
        ),
        axis=2,
    )
    cv2.circle(image, (40 + seed % 40, 50 + seed % 20), 12, (220, 210, 40), -1)
    cv2.rectangle(image, (70, 80), (110, 115), (30, 90, 180), -1)
    if synthetic:
        image = np.round(image / 16) * 16
        image[:, seed % 128, :] = np.clip(image[:, seed % 128, :].astype(int) + 40, 0, 255)
    image = np.clip(image.astype(np.float32) + rng.normal(0, 2, image.shape), 0, 255)
    return image.astype(np.uint8)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "demo"
    for index in range(24):
        real_path = root / "real" / f"real_{index:02}.png"
        real_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(real_path), make_image(index, False))
        generator = "demoA" if index < 16 else "demoB"
        fake_path = root / "synthetic" / f"synthetic_{index:02}__{generator}.png"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fake_path), make_image(100 + index, True))
    print(f"Demo-only images written under {root}")


if __name__ == "__main__":
    main()
