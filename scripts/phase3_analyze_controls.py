"""Frozen-head unmarked controls and five diagnostic source split refits.

No original coefficient, score, split, or fresh-test artifact is overwritten.
The five refits are exploratory: their source pool was already examined.
"""
from __future__ import annotations
import argparse,json,sys,warnings,importlib.metadata,platform,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
from prepare_conference import load_matrix
from evaluate_encoding_controls import digest,model_scores
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase3/controls';FIT=ROOT/'outputs/conference_clean'
sys.path.insert(0,str(ROOT/'src'))
from aiwstudy.evaluation import select_threshold
HELD={('lowres',2),('regenerate',2),('spectral',1)}
SEEDS=[20260920,20260921,20260922,20260923,20260924]

def write_json(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def metrics(y,score,prob=None,threshold=None):
    fpr,tpr,_=roc_curve(y,score,drop_intermediate=False)
    result={'auc':float(roc_auc_score(y,score)),'tpr_at_test_roc_fpr_5pct':float(tpr[fpr<=.05].max())}
    if prob is not None and threshold is not None:
        pred=prob>=threshold;result.update(validation_threshold=float(threshold),false_positive_rate=float(pred[y==0].mean()),true_positive_rate=float(pred[y==1].mean()))
    return result

def matrix(name,frame):
    paths=[ROOT/'outputs/resnet_embeddings.npz'] if name=='resnet' else [ROOT/'outputs/clip_train_embeddings.npz',ROOT/'outputs/clip_eval_embeddings.npz']
    return load_matrix(paths,frame.image_id)

def locked_features(name,kind,frame):
    path=OUT/f'{name}_{kind}_embeddings.npz'
    manifest={'unmarked':'unmarked_test_manifest.csv','heldout':'heldout_clip_manifest.csv',
        'openimages':'openimages_derived_manifest.csv','union':'unmarked_union_extra_manifest.csv'}[kind]
    with np.load(path,allow_pickle=True) as o:
        if not np.array_equal(o['image_id'].astype(str),frame.image_id.to_numpy()):raise ValueError('Feature row mismatch')
        if str(o['manifest_sha256'])!=digest(OUT/manifest):raise ValueError('Feature manifest checksum mismatch')
        return o['embedding'].copy()

def score(frame,x,name,variant='full',dest=FIT):
    prefix='' if name=='resnet' else 'clip_'
    p,logit=model_scores(x,dest/f'{name}_{variant}.npz')
    fp,fl=model_scores(np.column_stack([p,frame.active_score]),dest/f'{name}_fusion_{variant}.npz')
    enabled=frame.active_score.to_numpy()>=13/16
    frame[prefix+'probe']=p;frame[prefix+'probe_logit']=logit
    frame[prefix+'fusion']=fp;frame[prefix+'fusion_logit']=fl
    frame[prefix+'gated']=np.where(enabled,fp,p);frame[prefix+'gated_logit']=np.where(enabled,fl,logit)
    assert np.array_equal(frame.loc[~enabled,prefix+'gated'].to_numpy(),p[~enabled])

def summarize(frame,thresholds=None):
    result={};y=frame.label.to_numpy()
    result['payload']=metrics(y,frame.active_score.to_numpy(),frame.active_score.to_numpy(),13/16)
    for name in ['resnet','clip']:
        prefix='' if name=='resnet' else 'clip_'
        for system in ['probe','fusion','gated']:
            col=prefix+system;threshold=None if thresholds is None else thresholds[name][system]
            result[name+'_'+system]=metrics(y,frame[col+'_logit'].to_numpy(),frame[col].to_numpy(),threshold)
    return result

def original_thresholds():
    val=pd.read_csv(FIT/'validation_predictions.csv',keep_default_na=False)
    saved=json.loads((FIT/'metrics.json').read_text())['variants']['full']['systems']
    result={}
    for name in ['resnet','clip']:
        prefix='' if name=='resnet' else 'clip_';enabled=val.active_score.to_numpy()>=13/16
        gated=np.where(enabled,val[prefix+'fusion'],val[prefix+'probe'])
        result[name]={'probe':saved[prefix+'probe']['threshold'],'fusion':saved[prefix+'fusion']['threshold'],
            'gated':select_threshold(val.label.to_numpy(),gated)}
    return result

def paired_AI_effects(marked,unmarked,thresholds):
    keys=['source_id','attack','severity'];columns=['probe','clip_probe']
    pair=marked[keys+['generator']+columns].merge(unmarked[keys+columns],on=keys,suffixes=('_marked','_unmarked'),validate='one_to_one')
    answer={}
    for name,col in [('resnet','probe'),('clip','clip_probe')]:
        threshold=thresholds[name]['probe']
        pair['delta_probability']=pair[col+'_marked']-pair[col+'_unmarked']
        pair['delta_prediction']=(pair[col+'_marked']>=threshold).astype(int)-(pair[col+'_unmarked']>=threshold).astype(int)
        source=pair.groupby(['generator','source_id'])[['delta_probability','delta_prediction']].mean().reset_index()
        groups=[g[['delta_probability','delta_prediction']].to_numpy() for _,g in source.groupby('generator',sort=True)]
        rng=np.random.default_rng(20261010);draw=[]
        for _ in range(2000):draw.append(np.concatenate([g[rng.integers(len(g),size=len(g))] for g in groups]).mean(axis=0))
        interval=np.percentile(draw,[2.5,97.5],axis=0)
        answer[name]={'mean_probability_marked_minus_unmarked':float(source.delta_probability.mean()),'mean_probability_delta_ci95':interval[:,0].tolist(),
            'TPR_marked_minus_unmarked_at_original_validation_threshold':float(source.delta_prediction.mean()),'TPR_delta_ci95':interval[:,1].tolist(),
            'paired_rows':len(pair),'source_units':len(source),'bootstrap_replicates':2000,'bootstrap':'Pairs averaged within source, stratified by source collection; original 14 conditions have equal weight.'}
    return answer

def frozen():
    locked=json.loads((OUT/'protocol.json').read_text())['original_heads_sha256']
    if any(digest(FIT/name)!=value for name,value in locked.items()):raise ValueError('Original frozen heads changed')
    target=OUT/'unmarked_frozen_predictions.csv'
    frame=pd.read_csv(OUT/'unmarked_test_manifest.csv',keep_default_na=False)
    if target.exists():frame=pd.read_csv(target,keep_default_na=False)
    else:
        for name in ['resnet','clip']:score(frame,locked_features(name,'unmarked',frame),name)
        frame.to_csv(target,index=False)
    original=pd.read_csv(FIT/'test_predictions.csv',keep_default_na=False)
    for name in ['resnet','clip']:
        prefix='' if name=='resnet' else 'clip_';enabled=original.active_score.to_numpy()>=13/16
        original[prefix+'gated']=np.where(enabled,original[prefix+'fusion'],original[prefix+'probe'])
        original[prefix+'gated_logit']=np.where(enabled,original[prefix+'fusion_logit'],original[prefix+'probe_logit'])
    full=pd.concat([original,frame],ignore_index=True)
    full['collection']=np.where(full.generator.eq('real'),'coco',full.generator)
    full.to_csv(OUT/'expanded_test_predictions.csv',index=False)
    basecols=['image_id','source_id','path','label','generator','collection','split','group','watermark_method','payload','attack','severity','parent_id','active_score']
    full[basecols].to_csv(OUT/'expanded_test_manifest.csv',index=False)
    real=original.loc[original.group.eq('R-T')]
    marked=original.loc[original.group.eq('S-A')]
    unmarked=pd.concat([real,frame],ignore_index=True);markedset=pd.concat([real,marked],ignore_index=True)
    th=original_thresholds()
    report={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'heads_refitted':False,'thresholds_retuned':False,
        'thresholds':'Probe/fusion thresholds copied from original validation report; gate classification threshold calculated on original validation only, with the 13/16 gate unchanged.',
        'expected_unmarked_edit_rows':19544,'actual_unmarked_edit_rows':len(frame),'AI_sources':int(frame.source_id.nunique()),
        'expanded_test_rows':len(full),'expanded_test_sources':int(full.source_id.nunique()),
        'paired_design':'The same 1396 AI sources, 284 real sources, and 14 attacks; marked and unmarked AI variants compared separately to identical real-edit rows.',
        'old_test_overall':summarize(original,th),'expanded_test_overall':summarize(full,th),
        'matched_marked_edits':summarize(markedset,th),'matched_unmarked_edits':summarize(unmarked,th),'paired_AI_probe_effects':paired_AI_effects(marked,frame,th),'conditions':{},
        'original_heads_sha256':{p.name:digest(p) for p in FIT.glob('*.npz')},'unmarked_manifest_sha256':digest(OUT/'unmarked_test_manifest.csv')}
    for (attack,severity),sub in frame.groupby(['attack','severity'],sort=True):
        r=real.loc[real.attack.eq(attack)&real.severity.eq(severity)]
        m=marked.loc[marked.attack.eq(attack)&marked.severity.eq(severity)]
        report['conditions'][f'{attack}:{severity}']={'unmarked':summarize(pd.concat([r,sub]),th),'marked':summarize(pd.concat([r,m]),th),
            'payload_unmarked_mean':float(sub.active_score.mean()),'payload_unmarked_acceptance':float((sub.active_score>=13/16).mean()),
            'payload_marked_mean':float(m.active_score.mean()),'rows_each_AI_variant':len(sub),'real_rows':len(r)}
    write_json(OUT/'unmarked_frozen_results.json',report)
    rows=[]
    for condition,block in report['conditions'].items():
        for treatment in ['marked','unmarked']:
            for system,values in block[treatment].items():rows.append({'condition':condition,'AI_treatment':treatment,'system':system,**values})
    pd.DataFrame(rows).to_csv(OUT/'unmarked_condition_results.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k in ['expanded_test_rows','matched_marked_edits','matched_unmarked_edits']},indent=2),flush=True)

def split_sources(sources,seed):
    assigned={};rng=np.random.default_rng(seed)
    for gen,part in sources.groupby('generator',sort=True):
        ids=np.array(sorted(part.source_id.astype(str)))
        if gen=='midjourney':
            assigned.update({x:'test' for x in ids});continue
        ids=ids[rng.permutation(len(ids))];a=int(np.floor(.7*len(ids)));b=int(np.floor(.15*len(ids)))
        assigned.update({x:'train' for x in ids[:a]});assigned.update({x:'val' for x in ids[a:a+b]});assigned.update({x:'test' for x in ids[a+b:]})
    return assigned

def fit(x,y,path):
    if path.exists():return
    model=LogisticRegression(max_iter=3000,class_weight='balanced',random_state=0)
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
    if any(issubclass(w.category,ConvergenceWarning) for w in found):raise RuntimeError('Nonconverged split probe')
    np.savez(path,coefficients=model.coef_,intercept=model.intercept_,classes=model.classes_)

def splits(backbones=('resnet','clip')):
    frozen_heads=json.loads((OUT/'protocol.json').read_text())['original_heads_sha256']
    if any(digest(FIT/name)!=value for name,value in frozen_heads.items()):raise ValueError('Original frozen heads changed')
    old=pd.read_csv(FIT/'manifest_derived.csv',keep_default_na=False,dtype={'payload':str})
    extra=pd.read_csv(OUT/'heldout_clip_manifest.csv',keep_default_na=False,dtype={'payload':str})
    assert not set(extra.image_id)&set(old.image_id)
    oldcombined=pd.concat([old,extra],ignore_index=True)
    oi=pd.read_csv(OUT/'openimages_derived_manifest.csv',keep_default_na=False,dtype={'payload':str})
    unmarked_prior=pd.read_csv(OUT/'unmarked_test_manifest.csv',keep_default_na=False,dtype={'payload':str})
    unmarked_extra=pd.read_csv(OUT/'unmarked_union_extra_manifest.csv',keep_default_na=False,dtype={'payload':str})
    frame=pd.concat([oldcombined,oi,unmarked_prior,unmarked_extra],ignore_index=True);frame['collection']=np.where(frame.generator.eq('real'),'coco',frame.generator)
    oldsrc=pd.read_csv(FIT/'manifest_sources.csv',keep_default_na=False);oisrc=pd.read_csv(OUT/'openimages_sources.csv',keep_default_na=False)
    sources=pd.concat([oldsrc,oisrc],ignore_index=True)
    eligible=np.array([(a,int(s)) not in HELD for a,s in zip(frame.attack,frame.severity)]) & ~frame.group.eq('S-U-A').to_numpy()
    assert int(eligible.sum())==13*len(sources) and frame.loc[eligible].groupby('source_id').size().eq(13).all()
    root=OUT/'split_seeds';root.mkdir(parents=True,exist_ok=True)
    archive=root/'marked_only_seed_archive';archive.mkdir(parents=True,exist_ok=True)
    if not (archive/'pre_expansion_resnet_metrics.csv').exists() and (root/'per_seed_metrics_resnet.csv').exists():
        shutil.copy2(root/'per_seed_metrics_resnet.csv',archive/'pre_expansion_resnet_metrics.csv')
        shutil.copy2(root/'mean_std_metrics.csv',archive/'pre_expansion_mean_std_metrics.csv')
        for path in root.glob('*/resnet_*_test_predictions.csv'):
            dest=archive/path.parent.name;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
    assignments={seed:split_sources(sources,seed) for seed in SEEDS}
    assignmentrows=[]
    for seed,assignment in assignments.items():
        for row in sources.to_dict('records'):assignmentrows.append({'seed':seed,'source_id':row['source_id'],'generator':row['generator'],'split':assignment[row['source_id']]})
    saved=pd.read_csv(root/'assignments.csv',keep_default_na=False)
    assert saved.equals(pd.DataFrame(assignmentrows))
    protocol=json.loads((root/'protocol.json').read_text());assert protocol['seed_assignment_sha256']==digest(root/'assignments.csv')
    results=[];markedresults=[]
    for seed,assignment in assignments.items():
        split_values=frame.source_id.map(assignment).to_numpy();testframe=frame.loc[split_values=='test'];train=(split_values=='train')&eligible
        expected=testframe.groupby('source_id').label.first().map({0:16,1:30})
        assert testframe.groupby('source_id').size().equals(expected)
        for subset,part in [('overall',testframe),('coco',testframe.loc[testframe.label.eq(1)|testframe.collection.eq('coco')]),('openimages',testframe.loc[testframe.label.eq(1)|testframe.collection.eq('openimages')])]:
            result=metrics(part.label.to_numpy(),part.active_score.to_numpy(),part.active_score.to_numpy(),13/16)
            results.append({'seed':seed,'backbone':'active','variant':'full','system':'payload','real_source':subset,
                'test_rows':len(part),'test_sources':int(part.source_id.nunique()),'train_rows':int(train.sum()),
                'train_sources':int(frame.loc[train,'source_id'].nunique()),**result})
            marked=part.loc[~part.group.eq('S-U-A')];measure=metrics(marked.label.to_numpy(),marked.active_score.to_numpy(),marked.active_score.to_numpy(),13/16)
            markedresults.append({'seed':seed,'backbone':'active','variant':'full','system':'payload','real_source':subset,
                'test_rows':len(marked),'test_sources':int(marked.source_id.nunique()),'train_rows':int(train.sum()),'train_sources':int(frame.loc[train,'source_id'].nunique()),**measure})
    for name in backbones:
        h=matrix(name,extra) if name=='resnet' else locked_features(name,'heldout',extra)
        x=np.concatenate([matrix(name,old),h,locked_features(name,'openimages',oi),locked_features(name,'unmarked',unmarked_prior),locked_features(name,'union',unmarked_extra)])
        print('Loaded',name,x.shape,flush=True)
        for seed,assignment in assignments.items():
            dest=root/str(seed);dest.mkdir(parents=True,exist_ok=True)
            splits_=frame.source_id.map(assignment).to_numpy();train=(splits_=='train')&eligible;val=(splits_=='val')&eligible;test=splits_=='test'
            expected=frame.loc[test].groupby('source_id').label.first().map({0:16,1:30})
            assert frame.loc[train|val].groupby('source_id').size().eq(13).all() and frame.loc[test].groupby('source_id').size().equals(expected)
            testframe=frame.loc[test].copy();validation=frame.loc[val].copy()
            for variant in ['full','no_attacks','no_matched_reals']:
                selected=train.copy()
                if variant=='no_attacks':selected &= ~frame.group.isin(['S-A','R-T']).to_numpy()
                if variant=='no_matched_reals':selected &= ~frame.group.eq('R-T').to_numpy()
                path=dest/f'{name}_{variant}.npz';print('Fit',seed,name,variant,int(selected.sum()),flush=True);fit(x[selected],frame.loc[selected,'label'].to_numpy(),path)
                p,_=model_scores(x[val],path);fpath=dest/f'{name}_fusion_{variant}.npz';fit(np.column_stack([p,frame.loc[val,'active_score']]),frame.loc[val,'label'].to_numpy(),fpath)
                score(testframe,x[test],name,variant,dest);score(validation,x[val],name,variant,dest)
                prefix='' if name=='resnet' else 'clip_';thresholds={system:select_threshold(validation.label.to_numpy(),validation[prefix+system].to_numpy()) for system in ['probe','fusion','gated']}
                for subset,part in [('overall',testframe),('coco',testframe.loc[testframe.label.eq(1)|testframe.collection.eq('coco')]),('openimages',testframe.loc[testframe.label.eq(1)|testframe.collection.eq('openimages')])]:
                    for system in ['probe','fusion','gated']:
                        col=prefix+system;result=metrics(part.label.to_numpy(),part[col+'_logit'].to_numpy(),part[col].to_numpy(),thresholds[system])
                        results.append({'seed':seed,'backbone':name,'variant':variant,'system':system,'real_source':subset,'test_rows':len(part),'test_sources':int(part.source_id.nunique()),
                            'train_rows':int(selected.sum()),'train_sources':int(frame.loc[selected,'source_id'].nunique()),**result})
                        marked=part.loc[~part.group.eq('S-U-A')];measure=metrics(marked.label.to_numpy(),marked[col+'_logit'].to_numpy(),marked[col].to_numpy(),thresholds[system])
                        markedresults.append({'seed':seed,'backbone':name,'variant':variant,'system':system,'real_source':subset,
                            'test_rows':len(marked),'test_sources':int(marked.source_id.nunique()),'train_rows':int(selected.sum()),'train_sources':int(frame.loc[selected,'source_id'].nunique()),**measure})
                savecols=['image_id','source_id','label','generator','collection','group','attack','severity','active_score']+[prefix+s+suffix for s in ['probe','fusion','gated'] for suffix in ['', '_logit']]
                testframe[savecols].to_csv(dest/f'{name}_{variant}_test_predictions.csv',index=False)
                pd.DataFrame(results).to_csv(root/'per_seed_metrics.partial.csv',index=False)
        del x
    result=pd.DataFrame(results)
    for name in backbones:result.loc[result.backbone.isin(['active',name])].to_csv(root/f'per_seed_metrics_{name}.csv',index=False)
    markedresult=pd.DataFrame(markedresults)
    for name in backbones:markedresult.loc[markedresult.backbone.isin(['active',name])].to_csv(archive/f'per_seed_metrics_{name}.csv',index=False)
    available=[name for name in ['resnet','clip'] if (root/f'per_seed_metrics_{name}.csv').exists()]
    result=pd.concat([pd.read_csv(root/f'per_seed_metrics_{name}.csv') for name in available],ignore_index=True).drop_duplicates(['seed','backbone','variant','system','real_source'])
    result.to_csv(root/'per_seed_metrics.csv',index=False)
    measures=['auc','tpr_at_test_roc_fpr_5pct','false_positive_rate','true_positive_rate']
    summary=result.groupby(['backbone','variant','system','real_source'])[measures].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in summary.columns];summary.to_csv(root/'mean_std_metrics.csv',index=False)
    markedresult=pd.concat([pd.read_csv(archive/f'per_seed_metrics_{name}.csv') for name in available],ignore_index=True).drop_duplicates(['seed','backbone','variant','system','real_source'])
    markedresult.to_csv(archive/'per_seed_metrics.csv',index=False)
    ms=markedresult.groupby(['backbone','variant','system','real_source'])[measures].agg(['mean','std']).reset_index();ms.columns=['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in ms.columns];ms.to_csv(archive/'mean_std_metrics.csv',index=False)
    if any(digest(FIT/name)!=value for name,value in frozen_heads.items()):raise ValueError('Original frozen heads changed during refits')
    if available==['resnet','clip']:write_json(root/'complete.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'seeds':SEEDS,'source_count':len(sources),'completed_backbones':available,'rows_per_source':{'train':13,'validation':13,'test_AI':30,'test_real':16},'test_rows_per_seed':49158,'result_rows':len(result),
        'std':'Sample standard deviation across five split seeds (ddof=1); test sources overlap and the same Midjourney sources stay held out. This is neither a standard error nor a confidence interval.',
        'model_head_sha256':{p.relative_to(root).as_posix():digest(p) for p in root.glob('*/*.npz')},'metric_sha256':digest(root/'per_seed_metrics.csv'),
        'original_frozen_heads_unchanged':True,'original_heads_sha256':frozen_heads,
        'fit_reuse_policy':'Existing seed heads are reused because the amendment adds test-only copies; the source assignments, thirteen-condition training/validation matrices, and fit specification are unchanged. Package versions below describe this analysis runtime.',
        'source_assignment_sha256':digest(root/'assignments.csv'),
        'control_feature_sha256':{p.name:digest(p) for p in OUT.glob('*_embeddings.npz')},
        'script_sha256':digest(Path(__file__)),'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']}})
    print(summary.to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['frozen','splits','all'],default='all');ap.add_argument('--backbone',choices=['resnet','clip','both'],default='both');args=ap.parse_args()
    if args.stage in ('frozen','all'):frozen()
    if args.stage in ('splits','all'):splits(('resnet','clip') if args.backbone=='both' else (args.backbone,))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
