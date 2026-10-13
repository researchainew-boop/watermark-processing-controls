"""Encode and derive fixed Phase 6 inputs without inspecting test scores."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import benchmark_watermarks as marks
from aiwstudy.attacks import apply_attack
from aiwstudy.watermark import DwtDctWatermark
from phase5_fair_watermarks import matched_residual, psnr
from phase6_features import OUT, ROOT, PAYLOADS, digest, utc, write, check_final_lock

DATA = ROOT / "data/phase6/experiment"
SOURCES = OUT / "data_inventory/primary_source_manifest.csv"
CORE = (("identity","identity",0), ("jpeg50","jpeg",2), ("crop","crop",1), ("lowres32","lowres",2))
NOISE_SEEDS = (20261011,20261012,20261013)
WRONG_PAYLOAD = "0011001100110011"


def identity(*values):
    return hashlib.sha256("|".join(map(str,values)).encode()).hexdigest()[:24]


def selected_sources(role):
    check_final_lock("calibration")
    frame = pd.read_csv(source_file(role),keep_default_na=False,dtype={"public_id":str})
    return frame.loc[frame.role.eq(role)].sort_values("source_id").reset_index(drop=True)


def source_file(role):
    return OUT / "data_inventory/calibration_source_manifest.csv" if role == "calibration" else SOURCES


def prepare(role,device):
    import torch
    torch.set_num_threads(4)
    destination = OUT / role
    destination.mkdir(parents=True,exist_ok=True)
    frame = selected_sources(role)
    target = destination / "covers.csv"
    previous = pd.read_csv(target,keep_default_na=False,dtype={"embedded_payload":str}).to_dict("records") if target.exists() else []
    done = {(r["source_id"],r["watermark_method"],r["embedded_payload"]) for r in previous}
    for row in previous:
        assert digest(row["path"]) == row["output_file_sha256"]
    marker = marks.Watermarks(device,4) if frame.label.astype(int).any() else None
    timer = time.perf_counter()
    for index, source in enumerate(frame.to_dict("records")):
        if digest(source["base_path"]) != source["output_sha256"]:
            raise ValueError("Common primary base changed")
        pixels = marks.read_image(Path(source["base_path"]))
        choices = [("unmarked","")]
        if int(source["label"]):
            choices += [(method,payload) for method in ["dwt_dct","trustmark"] for payload in PAYLOADS]
        for method,payload in choices:
            key = (source["source_id"],method,payload)
            if key in done:
                continue
            gain, quality = 0.,120.
            if method == "unmarked":
                result = pixels
            else:
                marker.set_payload(payload)
                encoded = marker.encode(pixels,method)
                result,gain,quality = matched_residual(pixels,encoded,source["source_id"],payload)
            cid = identity("phase6cover",source["source_id"],method,payload)
            path = DATA / role / "covers" / method / f"{cid}.png"
            marks.save_pixels(path,result)
            previous.append({"cover_id":cid,"image_id":cid,"source_id":source["source_id"],"role":role,
                "collection":source["collection"],"label":int(source["label"]),"watermark_method":method,"embedded_payload":payload,
                "path":str(path),"output_file_sha256":digest(path),"base_path":source["base_path"],"base_sha256":source["output_sha256"],
                "embedding_PSNR_dB":quality,"residual_gain":gain})
            done.add(key)
        if (index+1)%20 == 0:
            pd.DataFrame(previous).to_csv(target,index=False)
            print("Encoded",role,index+1,"/",len(frame),"sources;elapsed",round(time.perf_counter()-timer,1),flush=True)
    pd.DataFrame(previous).sort_values("cover_id").to_csv(target,index=False)
    write(destination/"encoding_complete.json",{"utc":utc(),"source_manifest_sha256":digest(source_file(role)),"covers_sha256":digest(target),
        "sources":len(frame),"covers":len(previous),"PSNR_target":40.,"tolerance":.1,
        "configuration":"same whole-residual rescaling/source-payload seeded stochastic rounding as Phase5;outside official default",
        "score_based_selection":False,"driver_sha256":digest(__file__),"elapsed_seconds":time.perf_counter()-timer})
    if marker is not None:
        del marker
        gc.collect()
        torch.cuda.empty_cache()


def derive_core(role):
    destination = OUT / role
    covers = pd.read_csv(destination/"covers.csv",keep_default_na=False,dtype={"embedded_payload":str})
    rows = []
    for cover in covers.to_dict("records"):
        if digest(cover["path"]) != cover["output_file_sha256"]:
            raise ValueError("Cover changed")
        pixels = marks.read_image(Path(cover["path"]))
        for condition,name,severity in CORE:
            iid = identity(cover["cover_id"],condition)
            path = DATA / role / "images" / condition / f"{iid}.png"
            result = pixels.copy() if name == "identity" else apply_attack(pixels.copy(),name,severity)
            if path.exists():
                if not np.array_equal(marks.read_image(path),result):
                    raise ValueError("Derived image mismatch")
            else:
                marks.save_pixels(path,result)
            rows.append({**cover,"image_id":iid,"condition":condition,"input_cover_path":cover["path"],
                "input_cover_sha256":cover["output_file_sha256"],"path":str(path),"output_file_sha256":digest(path),
                "attack_PSNR_dB":psnr(pixels,result),"scope":"primary_core"})
    manifest = destination/"manifest.csv"
    pd.DataFrame(rows).sort_values("image_id").to_csv(manifest,index=False)
    write(destination/"derivation_complete.json",{"utc":utc(),"images":len(rows),"sources":covers.source_id.nunique(),
        "manifest_sha256":digest(manifest),"covers_sha256":digest(destination/"covers.csv"),"conditions":[c[0] for c in CORE],
        "driver_sha256":digest(__file__),"decoder_or_detector_scores_used":False})
    print("Derived",role,len(rows),"core received images",flush=True)


def attack_subset(role):
    sources = selected_sources(role)
    selected = []
    for collection,block in sources.groupby("collection",sort=True):
        quota = 50 if role == "calibration" else 20
        block = block.copy()
        block["order"] = block.source_id.map(lambda s:identity("phase6attacksubset",s))
        selected.extend(block.sort_values(["order","source_id"]).head(quota).source_id)
    return set(selected)


def derive_extra(role,device):
    from phase6_attacks import (WavesResNetPGD,WavesFactorizedReconstruction,StableDiffusionRegeneration,
        residual_budget,source_seed,release,quality)
    check_final_lock("calibration")
    destination = OUT / f"{role}_attacks"
    destination.mkdir(parents=True,exist_ok=True)
    covers = pd.read_csv(OUT/role/"covers.csv",keep_default_na=False,dtype={"embedded_payload":str})
    selected = attack_subset(role)
    covers = covers.loc[covers.source_id.isin(selected)].sort_values("cover_id")
    target = destination/"manifest.csv"
    rows = pd.read_csv(target,keep_default_na=False,dtype={"embedded_payload":str}).to_dict("records") if target.exists() else []
    done = {(r["cover_id"],r["condition"]) for r in rows}
    for row in rows:
        assert digest(row["path"]) == row["output_file_sha256"]
    specifications = [("waves_pgd8",lambda:WavesResNetPGD(device,steps=200),{"strength":8}),
        ("waves_reconstruction_q3",lambda:WavesFactorizedReconstruction(device),{}),
        ("sd15_ddim_s005",lambda:StableDiffusionRegeneration(device),{"strength":.05})]
    for name,factory,kwargs in specifications:
        attack = factory()
        timer = time.perf_counter()
        for index,cover in enumerate(covers.to_dict("records")):
            names = [name+"_raw",name+"_budget35"]
            if all((cover["cover_id"],condition) in done for condition in names):
                continue
            pixels = marks.read_image(Path(cover["path"]))
            seed = source_seed(cover["source_id"])
            raw = attack.apply(pixels,seed=seed,**kwargs)
            budgeted,settings = residual_budget(pixels,raw,35.,seed=seed)
            for condition,result,meta in [(names[0],raw,{}),(names[1],budgeted,settings)]:
                iid = identity(cover["cover_id"],condition)
                path = DATA/role/"extra"/condition/f"{iid}.png"
                marks.save_pixels(path,result)
                rows.append({**cover,"image_id":iid,"condition":condition,"input_cover_path":cover["path"],
                    "input_cover_sha256":cover["output_file_sha256"],"path":str(path),"output_file_sha256":digest(path),
                    "scope":"secondary_attack","attack_seed":seed,**quality(pixels,result),
                    "budget_gain":meta.get("gain",1.),"budget_target_reachable":meta.get("target_reachable",False),
                    "budget_within_tolerance":meta.get("within_tolerance",False)})
                done.add((cover["cover_id"],condition))
            if (index+1)%10 == 0:
                pd.DataFrame(rows).sort_values("image_id").to_csv(target,index=False)
                print(name,role,index+1,"/",len(covers),"elapsed",round(time.perf_counter()-timer,1),flush=True)
        release(attack)
    for cover in covers.to_dict("records"):
        pixels = marks.read_image(Path(cover["path"]))
        for seed in NOISE_SEEDS:
            condition = f"noise6_seed{seed}"
            if (cover["cover_id"],condition) in done:
                continue
            # Identical per-source noise across marked/unmarked arms.
            rng = np.random.default_rng(int(hashlib.sha256(f'{seed}|{cover["source_id"]}'.encode()).hexdigest()[:8],16))
            result = np.clip(pixels.astype(float)+rng.normal(0,6,pixels.shape),0,255).astype(np.uint8)
            iid = identity(cover["cover_id"],condition)
            path = DATA/role/"extra"/condition/f"{iid}.png"
            marks.save_pixels(path,result)
            rows.append({**cover,"image_id":iid,"condition":condition,"input_cover_path":cover["path"],
                "input_cover_sha256":cover["output_file_sha256"],"path":str(path),"output_file_sha256":digest(path),
                "scope":"secondary_noise","attack_seed":seed,**quality(pixels,result)})
    pd.DataFrame(rows).sort_values("image_id").to_csv(target,index=False)
    write(destination/"derivation_complete.json",{"utc":utc(),"source_subset":sorted(selected),"manifest_sha256":digest(target),
        "images":len(rows),"sources":covers.source_id.nunique(),"driver_sha256":digest(__file__),
        "selection":"fixed ID hash order percollection;50perrealcalcollection,20pertestcollection",
        "not_a_second_diffusion_family":True,"budget_adaptation_not_native_published_attack":True,
        "input_source_provenance_labels":"Reconstructed photographs may contain generated pixels; analyze separately from benign real-photo FPR"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=["prepare","core","extra"])
    parser.add_argument("--role",required=True,choices=["calibration","test"])
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    if args.stage == "prepare":prepare(args.role,args.device)
    elif args.stage == "core":derive_core(args.role)
    else:derive_extra(args.role,args.device)


if __name__ == "__main__":main()
