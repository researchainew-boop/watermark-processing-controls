from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import load_config, resolve, seed_everything
from .manifest import assign_splits, scan_manifest, validate_manifest
from .pipeline import (
    add_active_scores,
    add_passive_scores,
    derive_dataset,
    evaluate_scores,
    preprocess_manifest,
    train_passive,
)


def _load(args):
    config, root = load_config(args.config)
    seed_everything(int(config["seed"]))
    return config, root


def command_scan(args) -> None:
    config, root = _load(args)
    data = config["data"]
    frame = scan_manifest(
        resolve(root, data["real_dir"]),
        resolve(root, data["synthetic_dir"]),
        list(data["extensions"]),
        str(data["default_generator"]),
    )
    output = resolve(root, data["manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} source images to {output}")


def command_split(args) -> None:
    config, root = _load(args)
    path = resolve(root, config["data"]["manifest"])
    frame = pd.read_csv(path, keep_default_na=False)
    result = assign_splits(
        frame,
        float(config["data"]["train_fraction"]),
        float(config["data"]["val_fraction"]),
        int(config["seed"]),
        config["data"].get("heldout_generator"),
    )
    validate_manifest(result)
    result.to_csv(path, index=False)
    print(result.groupby(["split", "label", "generator"]).size().to_string())


def command_preprocess(args) -> None:
    config, root = _load(args)
    path = resolve(root, config["data"]["manifest"])
    frame = pd.read_csv(path, keep_default_na=False)
    validate_manifest(frame)
    result = preprocess_manifest(config, root, frame)
    output = path.with_name("manifest_base.csv")
    result.to_csv(output, index=False)
    print(f"Wrote standardized manifest to {output}")


def command_derive(args) -> None:
    config, root = _load(args)
    input_path = resolve(root, config["data"]["manifest"]).with_name("manifest_base.csv")
    if not input_path.exists():
        input_path = resolve(root, config["data"]["manifest"])
        frame = pd.read_csv(input_path, keep_default_na=False)
        frame = preprocess_manifest(config, root, frame)
        frame.to_csv(input_path.with_name("manifest_base.csv"), index=False)
    else:
        frame = pd.read_csv(input_path, keep_default_na=False)
    result, quality = derive_dataset(config, root, frame)
    output = input_path.with_name("manifest_derived.csv")
    result.to_csv(output, index=False)
    quality_path = resolve(root, config["evaluation"]["output_dir"]) / "quality.csv"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(quality_path, index=False)
    print(f"Wrote {len(result)} controlled samples to {output}")


def command_train(args) -> None:
    config, root = _load(args)
    path = resolve(root, config["data"]["manifest"]).with_name("manifest_derived.csv")
    manifest = pd.read_csv(path, keep_default_na=False)
    output = train_passive(config, root, manifest, args.variant)
    print(f"Wrote passive probe to {output}")


def command_evaluate(args) -> None:
    config, root = _load(args)
    path = resolve(root, config["data"]["manifest"]).with_name("manifest_derived.csv")
    manifest = pd.read_csv(path, keep_default_na=False)
    scored = add_active_scores(config, manifest)
    scored = add_passive_scores(config, root, scored, args.variant)
    report = evaluate_scores(config, root, scored, args.variant)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watermark-aware AI-image detection study")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "scan": command_scan,
        "split": command_split,
        "preprocess": command_preprocess,
        "derive": command_derive,
        "train": command_train,
        "evaluate": command_evaluate,
    }
    for name, function in commands.items():
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="configs/study.yaml")
        if name in {"train", "evaluate"}:
            child.add_argument(
                "--variant",
                choices=["full", "no-attacks", "no-matched-controls"],
                default="full",
            )
        child.set_defaults(function=function)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
