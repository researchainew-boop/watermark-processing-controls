"""Actual cached SD1.5-family latent denoising, separate VAE-only control.

Declared subset: original payload covers plus unmarked paired controls.
Two strengths, 20 DDIM steps, empty prompt/guidance1, one fixed noise seed.
Resume skips completed outputs, verifying their hashes; never retunes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home()/'.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21'
OUT = ROOT/'outputs/extension/diffusion'
DATA = ROOT/'data/extension_diffusion'
PAYLOAD = '1011010010110100'
SEED = 20261009

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    covers=pd.read_csv(ROOT/'outputs/extension/watermarks/base_covers.csv',keep_default_na=False)
    covers=covers.loc[covers.payload.isin(['',PAYLOAD])]
    protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'model':str(CACHE),'cached_revision':CACHE.name,
        'conditions':['vae_only','diffusion_strength0p25','diffusion_strength0p50'],
        'scheduler':'DDIM from cached scheduler config','steps':20,'strengths':[.25,.50],'prompt':'','guidance_scale':1.,
        'seed':SEED,'batch_size':1,'working_size':[256,256],'unet_text_dtype':'float16','vae_dtype':'float32',
        'planned_covers':len(covers),'planned_outputs':len(covers)*3,'source_count':int(covers.source_id.nunique()),
        'source_selection':'Frozen watermark pairs; default payload and unmarked controls',
        'label_interpretation':'Input-source provenance; reconstructed photographs have generated pixels',
        'checkpoint_sha256':{str(p.relative_to(CACHE)):digest(p) for p in CACHE.rglob('*') if p.is_file()},
        'covers_sha256':digest(ROOT/'outputs/extension/watermarks/base_covers.csv'),'exact_zhao_reproduction':False}
    freeze=OUT/'protocol.json'
    if not freeze.exists(): freeze.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.set_num_threads(4); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark=False
    pipe=StableDiffusionImg2ImgPipeline.from_pretrained(str(CACHE),torch_dtype=torch.float16,
        safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=True)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.float()
    # Diffusers prepares half inputs from the prompt dtype; explicitly cast
    # VAE inputs to float32 while preserving the published VAE operations.
    original_encode,original_decode=pipe.vae.encode,pipe.vae.decode
    pipe.vae.encode=lambda x,*a,**kw: original_encode(x.float(),*a,**kw)
    pipe.vae.decode=lambda x,*a,**kw: original_decode(x.float(),*a,**kw)
    original_prepare_latents=pipe.prepare_latents
    pipe.prepare_latents=lambda *a,**kw: original_prepare_latents(*a,**kw).to(dtype=torch.float16)
    pipe.enable_attention_slicing('max'); pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    manifest=OUT/'manifest.csv'
    rows=pd.read_csv(manifest,keep_default_na=False).to_dict('records') if manifest.exists() else []
    finished={(r['cover_id'],r['condition']):r for r in rows}
    generated=0
    for cover in covers.to_dict('records'):
        for condition,strength in [('vae_only',None),('diffusion_strength0p25',.25),('diffusion_strength0p50',.50)]:
            key=(cover['cover_id'],condition)
            if key in finished:
                assert digest(finished[key]['path'])==finished[key]['output_file_sha256']
                continue
            im=Image.open(cover['path']).convert('RGB')
            assert im.size==(256,256)
            generator=torch.Generator(device='cpu').manual_seed(SEED)
            started=time.perf_counter()
            with torch.inference_mode():
                if strength is None:
                    # Posterior mean: isolate codec distortion from latent noise.
                    x=pipe.image_processor.preprocess(im).to('cuda',dtype=torch.float32)
                    latent=pipe.vae.encode(x).latent_dist.mode()
                    decoded=pipe.vae.decode(latent).sample
                    result=pipe.image_processor.postprocess(decoded,output_type='pil')[0]
                    pipe.maybe_free_model_hooks()
                else:
                    result=pipe(prompt='',image=im,strength=strength,num_inference_steps=20,
                        guidance_scale=1.,generator=generator).images[0]
            target=DATA/condition/f"{cover['cover_id']}.png"; target.parent.mkdir(parents=True,exist_ok=True)
            result.save(target)
            pixels=np.array(result)
            if pixels.max()==pixels.min(): raise RuntimeError('Degenerate diffusion output')
            row={'cover_id':cover['cover_id'],'condition':condition,'attack':'vae_roundtrip' if strength is None else 'diffusion',
                'severity':0 if strength is None else strength,'strength':strength,'steps':0 if strength is None else 20,
                'seed':SEED,'path':str(target),'output_file_sha256':digest(target),'seconds':time.perf_counter()-started}
            rows.append(row); pd.DataFrame(rows).to_csv(manifest,index=False)
            generated+=1
            print(condition,cover['cover_id'],round(row['seconds'],2),'seconds',f'{len(rows)}/{len(covers)*3}',flush=True)
            if args.limit and generated>=args.limit: return
    (OUT/'complete.json').write_text(json.dumps({'completed_at_utc':datetime.now(timezone.utc).isoformat(),'outputs':len(rows),'manifest_sha256':digest(manifest)},indent=2),encoding='utf-8')

if __name__=='__main__': main()
