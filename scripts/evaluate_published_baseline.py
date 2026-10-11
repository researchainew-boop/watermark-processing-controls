"""Official UniversalFakeDetect ViT-L/14 + released head, unchanged weights.

Uses official CenterCrop224 preprocessing and unnormalized CLIP features.
Evaluates clean original sources, common-encoding controls, and fresh sample.
No training and no threshold tuning (published logit0 operating point).
"""
import importlib.util
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from evaluate_encoding_controls import digest
from evaluate_frozen_extension import summary

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'.tools/extension-assets'
OUT=ROOT/'outputs/extension/published_baseline'
EXPECTED='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'

def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'predictions.csv').exists(): raise RuntimeError('Published baseline outputs already frozen')
    assert digest(ASSETS/'ViT-L-14.pt')==EXPECTED
    spec=importlib.util.spec_from_file_location('official_clip_model',ASSETS/'official_clip_model.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    state=torch.jit.load(str(ASSETS/'ViT-L-14.pt'),map_location='cpu').state_dict()
    model=module.build_model(state).float().eval()
    visual=model.visual.to('cuda').eval()
    del model,state; gc.collect()
    head=torch.load(ASSETS/'univfd_fc_weights.pth',map_location='cpu',weights_only=True)
    linear=torch.nn.Linear(768,1); linear.load_state_dict(head,strict=True); linear.to('cuda').eval()
    transform=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    core=pd.read_csv(ROOT/'outputs/conference_clean/test_predictions.csv',keep_default_na=False)
    core=core.loc[core.group.isin(['R-C','S-U'])].copy(); core['benchmark']='core_clean_unmarked'
    encoding=pd.read_csv(ROOT/'outputs/extension_controls/encoding/manifest.csv',keep_default_na=False)
    encoding=encoding.loc[encoding.split=='test'].copy(); encoding['benchmark']='encoding_'+encoding.condition
    fresh=pd.read_csv(ROOT/'outputs/extension/fresh/manifest.csv',keep_default_na=False); fresh['benchmark']='fresh'
    frames=[core,encoding,fresh]
    rows=pd.concat(frames,ignore_index=True)
    protocol={'baseline':'UniversalFakeDetect released ViT-L/14 head','head_sha256':digest(ASSETS/'univfd_fc_weights.pth'),
        'backbone_sha256':EXPECTED,'official_model_code_sha256':digest(ASSETS/'official_clip_model.py'),
        'preprocessing':repr(transform),'feature_normalization':False,'classifier_threshold_logit':0.,
        'test_tuning':False,'batch_size':4,'dtype':'float32','input_rows':len(rows)}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    logits=[]
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            batch=torch.stack([transform(Image.open(path).convert('RGB')) for path in rows.path.iloc[start:start+4]]).cuda()
            logits.extend(linear(visual(batch)).flatten().cpu().tolist())
            if start%100==0: print('Published detector',start+len(batch),'/',len(rows),flush=True)
    rows['univfd_logit']=logits
    rows.to_csv(OUT/'predictions.csv',index=False)
    results={}
    for benchmark,sub in rows.groupby('benchmark'):
        results[benchmark]=summary(sub,{'univfd':'univfd_logit'})
        y=sub.label.to_numpy(); positive=sub.univfd_logit.to_numpy()>=0
        results[benchmark]['threshold0']={'fpr':float(positive[y==0].mean()),'tpr':float(positive[y==1].mean())}
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
