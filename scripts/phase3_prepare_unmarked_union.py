"""Add unmarked edited AI for the union of five diagnostic test splits.

Existing original-test copies are reused. These additional copies are never
eligible for training or calibration; every seed gets the same test design.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
from phase3_prepare_controls import ROOT,OUT,unmarked_source,digest

def main():
    target=OUT/'unmarked_union_extra_manifest.csv'
    if target.exists():print('Expanded unmarked test union already locked',flush=True);return
    source=pd.read_csv(ROOT/'outputs/conference_clean/manifest_sources.csv',keep_default_na=False)
    assignment=pd.read_csv(OUT/'split_seeds/assignments.csv',keep_default_na=False)
    union=set(assignment.loc[assignment.split.eq('test'),'source_id'])
    prior=pd.read_csv(OUT/'unmarked_test_manifest.csv',keep_default_na=False)
    missing=source.loc[source.label.eq(1)&source.source_id.isin(union)&~source.source_id.isin(prior.source_id)].copy()
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,part in enumerate(pool.map(unmarked_source,missing.to_dict('records'))):
            rows.extend(part)
            if i%100==0:print('Additional unmarked test sources',i+1,'/',len(missing),flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==14*len(missing)
    assert not set(frame.image_id)&set(prior.image_id)
    frame.to_csv(target,index=False)
    seed=OUT/'split_seeds/protocol.json';protocol=json.loads(seed.read_text())
    protocol['expanded_test_amendment_before_extra_inference']={'amended_at_utc':datetime.now(timezone.utc).isoformat(),
        'old_test_design':'16 conditions per source, edited AI marked only.',
        'new_test_design':'16originalconditions/source plus14unmarkededits/AI source;30rows/AI and16rows/real.',
        'reason':'Matched unmarked AI editing control in every seed test, keeping training/validation unchanged.',
        'extra_unmarked_rows':len(frame),'extra_unmarked_source_units':len(missing),'manifest_sha256':digest(target),
        'extra_detector_inference_performed':False,'training_or_validation_eligibility':False}
    protocol['test_rows_per_seed']=49158;protocol['test_AI_sources_per_seed']=1405;protocol['test_real_sources_per_seed']=438
    protocol['source_unit']='Every derivative follows its source. Training/validation use13originalconditions; test uses30AI or16real rows. AllMidjourney always test.'
    seed.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    top=OUT/'protocol.json';initial=json.loads(top.read_text());initial['expanded_test_amendment']=protocol['expanded_test_amendment_before_extra_inference'];initial['split_study_conditions']='13originaltraining/validationconditions;16originaltestconditions plus14unmarkedAIedits.'
    top.write_text(json.dumps(initial,indent=2),encoding='utf-8')
    print(json.dumps(protocol['expanded_test_amendment_before_extra_inference'],indent=2),flush=True)

if __name__=='__main__':main()
