"""Run evaluation with stage prints so a hang is visible."""

print("stage import", flush=True)
import pandas as pd
from aiwstudy.cli import _load
from aiwstudy.config import resolve
from aiwstudy.pipeline import add_active_scores, add_passive_scores, evaluate_scores

print("stage args", flush=True)
import argparse

args = argparse.Namespace(config="configs/study.yaml", variant="full")
config, root = _load(args)
print("stage manifest", flush=True)
path = resolve(root, config["data"]["manifest"]).with_name("manifest_derived.csv")
manifest = pd.read_csv(path, keep_default_na=False)
print(f"rows {len(manifest)}", flush=True)
scored = add_active_scores(config, manifest)
print("stage active", flush=True)
scored = add_passive_scores(config, root, scored, args.variant)
print("stage passive", flush=True)
evaluate_scores(config, root, scored, args.variant)
print("stage done", flush=True)

