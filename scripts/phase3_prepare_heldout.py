"""Lock only the missing CLIP rows needed for full-condition seed tests.

The original PNGs and all ResNet vectors already exist. Original training
held-out payload scores are computed here; no original file is rewritten.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json
import cv2
import numpy as np
import pandas as pd
from phase3_analyze_controls import split_sources,SEEDS,HELD
from phase3_prepare_controls import digest,OUT,ROOT,PAYLOAD
from aiwstudy.watermark import DwtDctWatermark
cv2.setNumThreads(1)

def main():
    target=OUT/'heldout_clip_manifest.csv'
    if target.exists():print('Held-out feature manifest already frozen',flush=True);return
    oldsrc=pd.read_csv(ROOT/'outputs/conference_clean/manifest_sources.csv',keep_default_na=False)
    oi=pd.read_csv(OUT/'openimages_sources.csv',keep_default_na=False)
    sources=pd.concat([oldsrc,oi],ignore_index=True)
    assignmentrows=[];union=set()
    for seed in SEEDS:
        assignment=split_sources(sources,seed);union.update(x for x,v in assignment.items() if v=='test')
        assignmentrows.extend({'seed':seed,'source_id':r['source_id'],'generator':r['generator'],'split':assignment[r['source_id']]} for r in sources.to_dict('records'))
    seedout=OUT/'split_seeds';seedout.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(assignmentrows).to_csv(seedout/'assignments.csv',index=False)
    allrows=pd.read_csv(ROOT/'data/manifest_derived.csv',keep_default_na=False,dtype={'payload':str})
    want=allrows.loc[allrows.source_id.isin(union&set(oldsrc.source_id)) & np.array([(a,int(s)) in HELD for a,s in zip(allrows.attack,allrows.severity)])].copy()
    cached=set()
    for path in [ROOT/'outputs/clip_train_embeddings.npz',ROOT/'outputs/clip_eval_embeddings.npz']:
        with np.load(path,allow_pickle=True) as obj:cached.update(obj['image_id'].astype(str))
    want=want.loc[~want.image_id.isin(cached)].reset_index(drop=True)
    active=pd.read_csv(ROOT/'outputs/active_scores.csv',keep_default_na=False)
    lookup=dict(zip(active.image_id,active.active_score))
    def prepare(row):
        pixels=cv2.imread(row['path'])
        if pixels is None:raise ValueError(row['path'])
        row['active_score']=lookup.get(row['image_id'])
        if row['active_score'] is None:row['active_score']=DwtDctWatermark(PAYLOAD,12).detect(pixels).bit_accuracy
        row['collection']='coco' if row['generator']=='real' else row['generator']
        row['output_file_sha256']=digest(row['path']);return row
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,row in enumerate(pool.map(prepare,want.to_dict('records'))):
            records.append(row)
            if i%1000==0:print('Missing CLIP held-out rows',i+1,'/',len(want),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(target,index=False)
    protocol={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'seeds':SEEDS,'sources':len(sources),
        'seed_assignment_sha256':digest(seedout/'assignments.csv'),'source_pool_sha256':{'old':digest(ROOT/'outputs/conference_clean/manifest_sources.csv'),'openimages':digest(OUT/'openimages_sources.csv')},
        'source_unit':'Every derivative follows its source; 13 conditions in training/validation and 16 in every test. Midjourney always test.',
        'heldout_rows_missing_CLIP':len(frame),'heldout_source_units_missing_CLIP':int(frame.source_id.nunique()),'heldout_manifest_sha256':digest(target),
        'extra_ResNet_features_required':0,'test_rows_per_seed':1843*16,'train_rows_per_seed':4102*13,'validation_rows_per_seed':878*13,
        'inference_performed_on_new_controls':False,
        'caveat':'Retrospective diagnostic splits of previously examined original sources plus separately selected Open Images; excludes the fresh192.',
        'calibration':'Fusion fitted only on validation, probe only on training; gate fixed13/16. Test-ROC TPR5 descriptive.',
        'script_sha256':digest(__file__)}
    (seedout/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    initial=json.loads((OUT/'protocol.json').read_text())
    initial['amendment_before_new_detector_inference']={'amended_at_utc':datetime.now(timezone.utc).isoformat(),
        'previous_split_scope':initial['split_study_conditions'],'reason':'Use the same full 16-condition test definition in every split as in the original experiment.',
        'new_split_scope':'13 training/validation conditions; all16testconditions. Existing ResNet vectors and images reused, missing CLIP test vectors extracted.',
        'heldout_manifest_sha256':digest(target),'inference_performed_on_new_controls':False}
    initial['split_study_conditions']='13 training/validation conditions and16testconditions; three held-out attacks never used for training or calibration.'
    (OUT/'protocol.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    print(json.dumps(protocol,indent=2),flush=True)

if __name__=='__main__':main()
