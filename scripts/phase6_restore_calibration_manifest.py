"""Recover pre-existing CSV bytes from unchanged pixels; never alter receipts."""
from __future__ import annotations
import hashlib,io,json
from pathlib import Path
import pandas as pd
from phase6_features import ROOT,OUT,digest,utc,write,check_final_lock
from phase6_attacks import quality,residual_budget
from benchmark_watermarks import read_image


def main():
    check_final_lock('calibration')
    directory=OUT/'calibration_attacks'
    target=directory/'manifest.csv'
    receipt=json.loads((directory/'features_complete.json').read_text())
    expected=receipt['manifest_sha256']
    if digest(target)==expected:
        print('Original manifest already matches its pre-existing receipt')
        return
    current=target.read_bytes()
    backup=directory/'manifest.reserialized.csv'
    if not backup.exists(): backup.write_bytes(current)
    frame=pd.read_csv(io.BytesIO(current),keep_default_na=False,dtype={'embedded_payload':str})
    if len(frame)!=900 or frame.image_id.nunique()!=900: raise ValueError('Unexpected calibration coverage')
    records=frame.to_dict('records')
    for row in records:
        for field,checksum in [('path','output_file_sha256'),('input_cover_path','input_cover_sha256'),('base_path','base_sha256')]:
            if digest(row[field])!=row[checksum]: raise ValueError('Recorded pixels changed')
        pixels=read_image(Path(row['input_cover_path']))
        result=read_image(Path(row['path']))
        row.update(quality(pixels,result))
        row['embedding_PSNR_dB']=float(row['embedding_PSNR_dB'])
        row['residual_gain']=float(row['residual_gain'])
        if row['condition'].endswith('_budget35'):
            raw_id=hashlib.sha256((row['cover_id']+'|'+row['condition'].replace('_budget35','_raw')).encode()).hexdigest()[:24]
            raw_path=Path(row['path']).parent.parent/(row['condition'].replace('_budget35','_raw'))/(raw_id+'.png')
            raw=read_image(raw_path)
            reconstructed,meta=residual_budget(pixels,raw,35.,seed=int(row['attack_seed']))
            import numpy as np
            if not np.array_equal(reconstructed,result): raise ValueError('Budgeted pixels changed')
            row.update(budget_gain=meta['gain'],budget_target_reachable=meta['target_reachable'],budget_within_tolerance=meta['within_tolerance'])
        elif row['condition'].endswith('_raw'):
            row.update(budget_gain=1.,budget_target_reachable=False,budget_within_tolerance=False)
        else:
            for field in ['budget_gain','budget_target_reachable','budget_within_tolerance']: row.pop(field,None)
    # Before the first SD run, the earlier 400 PGD/compression rows had passed
    # through one read_csv resume. The next 500 SD/noise rows were new values.
    old=[r for r in records if r['condition'].startswith(('waves_pgd8','waves_reconstruction_q3'))]
    new=[r for r in records if not r['condition'].startswith(('waves_pgd8','waves_reconstruction_q3'))]
    if len(old)!=400 or len(new)!=500: raise ValueError('Unexpected historical stage sizes')
    old_csv=pd.DataFrame(old,columns=frame.columns).sort_values('image_id').to_csv(index=False)
    old=pd.read_csv(io.StringIO(old_csv),keep_default_na=False,dtype={'embedded_payload':str}).to_dict('records')
    candidate=pd.DataFrame(old+new,columns=frame.columns).sort_values('image_id').to_csv(index=False).encode('utf-8')
    candidate_sha=hashlib.sha256(candidate).hexdigest()
    if candidate_sha!=expected:
        raise ValueError(f'Exact restoration refused: {candidate_sha} != {expected}')
    target.write_bytes(candidate)
    if digest(target)!=expected: raise ValueError('Restored file differs')
    write(directory/'manifest_restoration.json',{'utc':utc(),'preexisting_features_receipt_sha256':digest(directory/'features_complete.json'),
        'reserialized_sha256':hashlib.sha256(current).hexdigest(),'restored_sha256':expected,'images_unchanged_and_hash_checked':900,
        'method':'Recompute pixel quality/budget metadata; replay known single earlier read_csv stage; accept only original sealed SHA',
        'scores_models_cuts_or_receipts_changed':False,'script_sha256':digest(__file__)})
    print('Exact original calibration manifest restored; all old receipts unchanged')


if __name__=='__main__':main()
