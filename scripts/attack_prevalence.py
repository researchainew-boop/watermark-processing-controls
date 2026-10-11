"""Paired marking prevalence under edits, with unmarked attacked counterfactuals."""
import argparse
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experimental_reliability import prepare_auc,weighted_auc

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/extension/attack_prevalence'
PAYLOAD='1011010010110100'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    covers=pd.read_csv(ROOT/'outputs/extension/watermarks/base_covers.csv',keep_default_na=False,dtype={'payload':str})
    covers=covers.loc[covers.watermark_method!='trustmark'].set_index('cover_id')
    local=pd.read_csv(ROOT/'outputs/extension/watermarks/attack_manifest.csv',keep_default_na=False)
    regen=pd.read_csv(ROOT/'outputs/extension/diffusion/manifest.csv',keep_default_na=False)
    conditions=pd.concat([local,regen],ignore_index=True).fillna('')
    rows=[]
    for row in conditions.to_dict('records'):
        if row['cover_id'] not in covers.index: continue
        cover=covers.loc[row['cover_id']].to_dict()
        rows.append({**cover,**row,'image_id':row['cover_id']+'_'+row['condition'],
            'group':'R-C' if int(cover['label'])==0 else ('S-U' if cover['watermark_method']=='unmarked' else 'S-W')})
    frame=pd.DataFrame(rows)
    assert len(frame)==1776 and frame.source_id.nunique()==48
    frame.to_csv(OUT/'manifest.csv',index=False)

def analyze():
    predictions_path=OUT/'predictions.csv'
    frame=pd.read_csv(predictions_path,keep_default_na=False,dtype={'payload':str})
    archived_rows=len(frame)
    # Compare marked/unmarked outcomes under the same expected payload.
    # The other two marked payloads have no corresponding unmarked/real
    # payload-specific detector outputs and remain in the recovery benchmark.
    frame=frame.loc[frame.watermark_method.eq('unmarked') |
        (frame.watermark_method.eq('dwt_dct') & frame.payload.eq(PAYLOAD))].copy()
    assert archived_rows==1776 and len(frame)==1152 and frame.condition.nunique()==16
    assert frame.source_id.nunique()==48
    for condition,sub in frame.groupby('condition'):
        assert len(sub)==72 and sub.source_id.nunique()==48
        real=sub.loc[sub.label==0]; ai=sub.loc[sub.label==1]
        assert len(real)==real.source_id.nunique()==24
        assert len(ai)==48 and ai.source_id.nunique()==24
        assert ai.groupby(['source_id','watermark_method']).size().eq(1).all()
        assert set(ai.watermark_method)=={'unmarked','dwt_dct'}
        assert ai.loc[ai.watermark_method.eq('dwt_dct'),'payload'].eq(PAYLOAD).all()
        assert sub.loc[sub.watermark_method.eq('unmarked'),'payload'].eq('').all()
    protocol={'analyzed_at_utc':datetime.now(timezone.utc).isoformat(),
        'predictions_sha256':hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        'archived_inference_rows':archived_rows,'selected_counterfactual_rows':len(frame),
        'conditions':16,'distinct_sources':48,'ai_sources':24,'real_sources':24,
        'rows_per_condition':72,'expected_payload':PAYLOAD,
        'analysis_selection':'Original-payload marked AI plus all unmarked AI and real controls; same expected string for each counterfactual',
        'selection_correction_after_inference':True,'models_or_thresholds_changed':False,
        'inference_repeated':False,'other_payload_results':'Retained in the separate multiple-payload watermark recovery benchmark'}
    (OUT/'analysis_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={}
    for condition,sub in frame.groupby('condition'):
        sub=sub.copy(); y=sub.label.to_numpy(); marked=sub.watermark_method.eq('dwt_dct').to_numpy()
        counts=sub.loc[marked].groupby('source_id').size()
        count=sub.source_id.map(counts).fillna(1).to_numpy()
        endpoints=[]
        codes,sources=pd.factorize(sub.source_id,sort=True)
        strata=[np.unique(codes[sub.generator.to_numpy()==g]) for g in sorted(sub.generator.unique())]
        for name,prefix in [('resnet',''),('clip','clip_')]:
            ranks={kind:sub[prefix+kind+'_logit'].to_numpy() for kind in ['probe','fusion','gated']}
            prepared={key:prepare_auc(y,s) for key,s in ranks.items()}
            rng=np.random.default_rng(20261009); boots={key:[[],[]] for key in ranks}; estimates={}
            baseweights=[np.where(y==0,1,np.where(marked,0,1)),np.where(y==0,1,np.where(marked,1/count,0))]
            for key in ranks: estimates[key]=[weighted_auc(prepared[key],w) for w in baseweights]
            for _ in range(2000):
                sourceweights=np.zeros(len(sources))
                for group in strata: sourceweights[group]=np.bincount(rng.integers(len(group),size=len(group)),minlength=len(group))
                for key in ranks:
                    for endpoint in range(2): boots[key][endpoint].append(weighted_auc(prepared[key],sourceweights[codes]*baseweights[endpoint]))
            for share in np.linspace(0,1,11):
                row={'backbone':name,'marked_share':float(share),'sources':len(sources)}
                for key in ranks:
                    row[key+'_auc']=(1-share)*estimates[key][0]+share*estimates[key][1]
                for key in ['fusion','gated']:
                    differences=(1-share)*(np.array(boots[key][0])-boots['probe'][0])+share*(np.array(boots[key][1])-boots['probe'][1])
                    row[key+'_gain']=row[key+'_auc']-row['probe_auc']
                    row[key+'_gain_ci95']=np.percentile(differences,[2.5,97.5]).tolist()
                endpoints.append(row)
        results[condition]=endpoints
    (OUT/'prevalence.json').write_text(json.dumps({'scope':'Matched original-payload marked/unmarked transformed outcomes; same expected string throughout, source-clustered and collection-stratified; input provenance labels for VAE/diffusion',
        'source_count':48,'selected_counterfactual_rows':len(frame),'archived_inference_rows':archived_rows,
        'expected_payload':PAYLOAD,'bootstrap_samples':2000,'conditions':results},indent=2),encoding='utf-8')
    print('Paired attack prevalence completed',len(results),'conditions',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['prepare','analyze'],required=True); args=ap.parse_args()
    prepare() if args.stage=='prepare' else analyze()
