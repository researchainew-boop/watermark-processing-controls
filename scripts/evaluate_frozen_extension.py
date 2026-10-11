"""One inference pass of unchanged study heads on a locked extension manifest."""
import argparse
import json
import math
import gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score
from evaluate_encoding_controls import load_backbone, checkpoint_defaults, infer, model_scores, digest
from aiwstudy.watermark import DwtDctWatermark
from aiwstudy.decision import DecisionPolicy, decision_metrics

ROOT=Path(__file__).resolve().parents[1]

def finite_json(value):
    """Absent subgroup metrics are JSON null, never a nonstandard NaN token."""
    if isinstance(value,dict): return {key:finite_json(item) for key,item in value.items()}
    if isinstance(value,list): return [finite_json(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def summary(frame,columns):
    result={}
    for name,col in columns.items():
        y=frame.label.to_numpy(); scores=frame[col].to_numpy()
        codes,sources=pd.factorize(frame.source_id,sort=True)
        strata=[np.unique(codes[frame.generator.to_numpy()==g]) for g in sorted(frame.generator.unique())]
        rng=np.random.default_rng(20261009); boot=[]
        for _ in range(2000):
            weights=np.zeros(len(sources))
            for group in strata: weights[group]=np.bincount(rng.integers(len(group),size=len(group)),minlength=len(group))
            boot.append(roc_auc_score(y,scores,sample_weight=weights[codes]))
        result[name]={'auc':float(roc_auc_score(y,scores)),'ci95':np.percentile(boot,[2.5,97.5]).tolist()}
    return result

def main():
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,default=ROOT/'outputs/extension/fresh/manifest.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/extension/fresh'); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    resultpath=args.output/'predictions.csv'
    if resultpath.exists(): raise RuntimeError('Existing inference is frozen; refusing another pass')
    frame=pd.read_csv(args.manifest,keep_default_na=False)
    for row in frame.to_dict('records'):
        if digest(Path(row['path']))!=row['output_file_sha256']: raise ValueError('Changed manifest pixels')
    started=datetime.now(timezone.utc).isoformat()
    _,checkpoint=checkpoint_defaults()
    columns={}
    for backbone,prefix in [('resnet',''),('clip','clip_')]:
        model,transform,forward,preprocessing,identity=load_backbone(backbone,'cuda',checkpoint)
        vectors=infer(frame,transform,forward,'cuda',8)
        p,logit=model_scores(vectors,ROOT/f'outputs/conference_clean/{backbone}_full.npz')
        if 'active_score' not in frame:
            frame['active_score']=[DwtDctWatermark(row.get('payload') or '1011010010110100',12).detect(cv2.imread(row['path'])).bit_accuracy for row in frame.to_dict('records')]
        fp,fl=model_scores(np.column_stack([p,frame.active_score]),ROOT/f'outputs/conference_clean/{backbone}_fusion_full.npz')
        enabled=frame.active_score.to_numpy()>=13/16
        frame[prefix+'probe']=p; frame[prefix+'probe_logit']=logit
        frame[prefix+'fusion']=fp; frame[prefix+'fusion_logit']=fl
        frame[prefix+'gated']=np.where(enabled,fp,p); frame[prefix+'gated_logit']=np.where(enabled,fl,logit)
        columns.update({backbone+'_'+kind:prefix+kind+'_logit' for kind in ['probe','fusion','gated']})
        del model,forward,transform,vectors; gc.collect(); torch.cuda.empty_cache()
    # Commit predictions before any outcome analysis; no refitting is possible.
    frame.to_csv(resultpath,index=False)
    report={'inference_started_at_utc':started,'inference_completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'rows':len(frame),'sources':int(frame.source_id.nunique()),'manifest_sha256':digest(args.manifest),
        'models_refitted':False,'thresholds_retuned':False,'overall':summary(frame,columns),'per_real_collection':{}}
    for real in sorted(frame.loc[frame.label==0,'generator'].unique()):
        sub=frame.loc[(frame.label==1)|(frame.generator==real)]
        report['per_real_collection'][real]=summary(sub,columns)
    policy=json.loads((ROOT/'outputs/extension/hard_gate/policy.json').read_text())
    frame['fusion_score']=frame.gated
    report['frozen_policy']=decision_metrics(frame,DecisionPolicy(13/16,policy['selected']['margin']))
    encoded=json.dumps(finite_json(report),indent=2,allow_nan=False)
    (args.output/'metrics.json').write_text(encoded,encoding='utf-8')
    print(encoded,flush=True)

if __name__=='__main__': main()
