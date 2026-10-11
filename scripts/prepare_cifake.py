"""Prepare a CIFAKE-style folder layout for the study configuration.

The script documents the expected layout and, when huggingface_hub is
installed, can download a small snapshot. CIFAKE is a public real/AI image
benchmark based on CIFAR-10 and Stable Diffusion. Confirm the current license
before redistribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

README = """Place real images under real/ and AI images under synthetic/.

Name AI files as `{id}__{generator}.png` so the held-out generator can be
reserved for test. Example:

  data/raw/real/0001.png
  data/raw/synthetic/0001__generatorA.png
  data/raw/synthetic/0002__generatorB.png
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--repo", default="bird-of-paradise/cifake")
    args = parser.parse_args()
    root = Path(args.root)
    (root / "real").mkdir(parents=True, exist_ok=True)
    (root / "synthetic").mkdir(parents=True, exist_ok=True)
    (root / "README.txt").write_text(README, encoding="utf-8")
    if args.download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("Install the optional [datasets] extra to download.") from error
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(root / "huggingface_snapshot"),
        )
        print(f"Downloaded snapshot under {root / 'huggingface_snapshot'}")
        print("Copy real and AI splits into data/raw/real and data/raw/synthetic.")
        return
    print(f"Created empty layout at {root.resolve()}")
    print("Add licensed public images, then run `aiw scan --config configs/study.yaml`.")


if __name__ == "__main__":
    main()
