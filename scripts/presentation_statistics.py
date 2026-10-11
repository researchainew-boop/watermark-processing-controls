"""Complete condition table and independently reconcile manuscript source counts."""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import roc_auc_score
from hard_gate_analysis import gated

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/extension/presentation'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=pd.read_csv(ROOT/'outputs/conference_clean/manifest_sources.csv',keep_default_na=False)
    derived=pd.read_csv(ROOT/'outputs/conference_clean/manifest_derived.csv',keep_default_na=False)
    test=pd.read_csv(ROOT/'outputs/conference_clean/test_predictions.csv',keep_default_na=False)
    assert len(sources)==sources.source_id.nunique()==5823
    assert len(derived)==80739
    assert derived.loc[derived.split=='test'].groupby('source_id').size().eq(16).all()
    assert derived.loc[derived.split!='test'].groupby('source_id').size().eq(13).all()
    assert test.source_id.nunique()==1680 and len(test)==1680*16
    counts=sources.groupby(['split','generator']).size().to_dict()
    audit=json.load(open(ROOT/'outputs/conference_clean/metrics.json'))['audit']['sources_by_split_generator']
    assert {f'{s}|{g}':int(n) for (s,g),n in counts.items()}==audit
    assert not sources.loc[sources.split.isin(['train','val']),'generator'].eq('midjourney').any()
    rows=[]
    for key,sub in [('identity:0',test.loc[test.group.isin(['R-C','S-W'])])]+[(f'{a}:{s}',b) for (a,s),b in test.loc[test.group.isin(['R-T','S-A'])].groupby(['attack','severity'])]:
        _,rank=gated(sub)
        row={'condition':key,'sources':int(sub.source_id.nunique()),'marked_mean_agreement':float(sub.loc[sub.label==1,'active_score'].mean()),
             'real_mean_agreement':float(sub.loc[sub.label==0,'active_score'].mean()),
             'active_auc':float(roc_auc_score(sub.label,sub.active_score)),
             'probe_auc':float(roc_auc_score(sub.label,sub.probe_logit)),
             'fusion_auc':float(roc_auc_score(sub.label,sub.fusion_logit)),
             'gated_auc':float(roc_auc_score(sub.label,rank))}
        rows.append(row)
    assert len(rows)==15
    pd.DataFrame(rows).to_csv(OUT/'full_condition_table.csv',index=False)
    (OUT/'table_i_verification.json').write_text(json.dumps({'retained_sources':len(sources),'derived_variants':len(derived),'test_variants':len(test),
        'counts':audit,'test_variants_per_source':16,'fit_variants_per_source':13,
        'held_out_three_conditions_excluded_from_train_and_validation':True,'table_i_matches_independent_manifests':True},indent=2),encoding='utf-8')
    print(next(x for x in rows if x['condition']=='crop:1'),flush=True)

if __name__=='__main__': main()
