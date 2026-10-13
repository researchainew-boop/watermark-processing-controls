"""Preserve interrupted, unrecorded attack files before an unchanged retry."""
from pathlib import Path
import pandas as pd
from phase6_features import ROOT,OUT,digest,write,utc,check_final_lock


def main():
    check_final_lock('test')
    directory=OUT/'test_attacks'
    if (directory/'inference_started.json').exists():
        raise RuntimeError('Do not move inputs after attack score inference starts')
    manifest=directory/'manifest.csv'
    frame=pd.read_csv(manifest,keep_default_na=False) if manifest.exists() else pd.DataFrame()
    recorded=set()
    for row in frame.to_dict('records'):
        path=Path(row['path']).resolve()
        if digest(path)!=row['output_file_sha256']: raise ValueError('Recorded attack changed')
        recorded.add(path)
    base=(ROOT/'data/phase6/experiment/test/extra').resolve()
    archive=(directory/'interrupted_uncommitted'/utc().replace(':','-')).resolve()
    base.relative_to(ROOT.resolve());archive.relative_to(ROOT.resolve())
    records=[]
    for path in sorted(base.rglob('*.png')):
        path=path.resolve();path.relative_to(base)
        if path in recorded: continue
        relative=path.relative_to(base)
        destination=(archive/relative).resolve();destination.relative_to(archive)
        fingerprint=digest(path)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists(): raise FileExistsError(destination)
        path.rename(destination)
        records.append({'relative_path':relative.as_posix(),'sha256':fingerprint})
    write(archive/'receipt.json',{'utc':utc(),'registered_manifest_sha256':digest(manifest) if manifest.exists() else None,
        'recorded_images_preserved':len(recorded),'uncommitted_images_archived':records,
        'reason':'Interrupted between ten-cover manifest checkpoints; GPU PGD retry is not byte deterministic',
        'no_test_score_inference_started':True,'scientific_driver_or_protocol_changed':False})
    print('Preserved recorded images:',len(recorded),'archived uncommitted images:',len(records))


if __name__=='__main__':main()
