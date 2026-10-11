"""Extract the unchanged study backbones for locked Phase 3 controls.

The root agent schedules this GPU task. Completed blocks are reused on resume,
with manifest and row identity hashes protecting each block.
"""
from __future__ import annotations
import argparse,gc,json,hashlib,importlib.metadata,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from evaluate_encoding_controls import load_backbone,checkpoint_defaults,infer,cached_features,digest,original_caches
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase3/controls'

def main():
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--block-size',type=int,default=1024);ap.add_argument('--which',choices=['unmarked','openimages','heldout','union','both'],default='both');args=ap.parse_args()
    manifests=[]
    if args.which in ('unmarked','both'): manifests.append(('unmarked',OUT/'unmarked_test_manifest.csv'))
    if args.which in ('openimages','both'): manifests.append(('openimages',OUT/'openimages_derived_manifest.csv'))
    if args.which in ('heldout','both'):manifests.append(('heldout',OUT/'heldout_clip_manifest.csv'))
    if args.which in ('union','both'):manifests.append(('union',OUT/'unmarked_union_extra_manifest.csv'))
    _,checkpoint=checkpoint_defaults()
    anchors=pd.read_csv(ROOT/'outputs/conference_clean/manifest_sources.csv',keep_default_na=False).groupby('generator').head(1).copy()
    anchors['path']=anchors.source_id.map(lambda x:str(ROOT/'data/processed/base'/f'{x}.png'))
    for name in ['resnet','clip']:
        model,transform,forward,preprocessing,identity=load_backbone(name,args.device,checkpoint)
        current=infer(anchors,transform,forward,args.device,args.batch_size).astype(np.float64)
        cache=cached_features(original_caches()[:1] if name=='resnet' else original_caches()[1:],anchors.source_id.tolist()).astype(np.float64)
        cosine=np.sum(current*cache,axis=1)/(np.linalg.norm(current,axis=1)*np.linalg.norm(cache,axis=1))
        relative=np.linalg.norm(current-cache,axis=1)/np.linalg.norm(cache,axis=1)
        if cosine.min()<.99999 or relative.max()>.005: raise ValueError('Cache comparability failed')
        audit={'model':identity,'preprocessing':preprocessing,'sources':anchors.source_id.tolist(),'cosine':cosine.tolist(),'relative_l2':relative.tolist(),'timestamp_utc':datetime.now(timezone.utc).isoformat()}
        (OUT/f'{name}_comparability.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        for kind,manifest in manifests:
            if kind=='heldout' and name=='resnet':continue
            frame=pd.read_csv(manifest,keep_default_na=False);mhash=digest(manifest)
            blockdir=OUT/'feature_blocks'/name/kind;blockdir.mkdir(parents=True,exist_ok=True)
            blocks=[]
            for start in range(0,len(frame),args.block_size):
                part=frame.iloc[start:start+args.block_size];target=blockdir/f'{start:06d}.npz'
                if target.exists():
                    with np.load(target,allow_pickle=True) as obj:
                        if str(obj['manifest_sha256'])!=mhash or not np.array_equal(obj['image_id'].astype(str),part.image_id.to_numpy()): raise ValueError('Changed feature block manifest')
                        values=obj['embedding'].copy()
                    print('Reused',name,kind,start+len(part),'/',len(frame),flush=True)
                else:
                    for row in part.to_dict('records'):
                        if digest(Path(row['path']))!=row['output_file_sha256']: raise ValueError('Changed locked image')
                    values=infer(part,transform,forward,args.device,args.batch_size)
                    np.savez(target,image_id=part.image_id.to_numpy(),embedding=values,manifest_sha256=mhash)
                blocks.append(values)
                print('Completed',name,kind,start+len(part),'/',len(frame),flush=True)
            np.savez(OUT/f'{name}_{kind}_embeddings.npz',image_id=frame.image_id.to_numpy(),source_id=frame.source_id.to_numpy(),embedding=np.concatenate(blocks),manifest_sha256=mhash)
            del blocks
        del model,transform,forward;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    (OUT/f'feature_extraction_{args.which}_complete.json').write_text(json.dumps({'completed_at_utc':datetime.now(timezone.utc).isoformat(),'models_refitted':False,'which':args.which,
        'manifests_sha256':{kind:digest(path) for kind,path in manifests},'script_sha256':digest(Path(__file__)),
        'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','open_clip_torch','numpy','opencv-python-headless']},
        'python':platform.python_version(),'torch_cuda':torch.version.cuda,'device':args.device,
        'CLIP_checkpoint_sha256':digest(checkpoint)},indent=2),encoding='utf-8')

if __name__=='__main__':main()
