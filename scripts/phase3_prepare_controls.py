"""Prepare new diagnostic controls without altering any previous experiment.

Selection is independent of detector scores; the 192-source frozen sample is
excluded, together with every original source, by identity and pixel/hash rules.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
cv2.setNumThreads(1)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aiwstudy.attacks import apply_attack
from aiwstudy.watermark import DwtDctWatermark
OUT=ROOT/'outputs/phase3/controls'
DATA=ROOT/'data/phase3/controls'
PAYLOAD='1011010010110100'
SEED=20261010

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def write_json(p,o): Path(p).write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def sid(key): return hashlib.sha256(f'phase3|{SEED}|{key}'.encode()).hexdigest()[:24]
def ahash(a):
    b=cv2.resize(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),(8,8),interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((b>=b.mean()).ravel()).tobytes(),'big')

def attacks():
    return [x for x in yaml.safe_load((ROOT/'configs/study.yaml').read_text())['attacks'] if x['name']!='identity']

def unmarked_source(src):
    base=ROOT/'data/processed/base'/f"{src['source_id']}.png"
    original=cv2.imread(str(base)); marker=DwtDctWatermark(PAYLOAD,12)
    if original is None: raise ValueError(base)
    rows=[]
    for a in attacks():
        name,severity=a['name'],a['severity']
        identity=sid(f"unmarked|{src['source_id']}|{name}|{severity}")
        path=DATA/'unmarked_edits'/f'{identity}.png'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            if not cv2.imwrite(str(path),apply_attack(original,name,severity)): raise ValueError(path)
        image=cv2.imread(str(path))
        rows.append({'image_id':identity,'source_id':src['source_id'],'path':str(path),'label':1,
            'generator':src['generator'],'collection':src['generator'],'split':'test','group':'S-U-A',
            'watermark_method':'','payload':'','attack':name,'severity':severity,'parent_id':src['source_id'],
            'active_score':marker.detect(image).bit_accuracy,'output_file_sha256':digest(path)})
    return rows

def prepare_unmarked():
    target=OUT/'unmarked_test_manifest.csv'
    if target.exists(): print('Unmarked edited manifest already locked',flush=True); return
    src=pd.read_csv(ROOT/'outputs/conference_clean/manifest_sources.csv',keep_default_na=False)
    src=src.loc[src.split.eq('test')&src.label.eq(1)]
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,part in enumerate(pool.map(unmarked_source,src.to_dict('records'))):
            rows.extend(part)
            if i%100==0: print('Unmarked edits',i+1,'/',len(src),flush=True)
    frame=pd.DataFrame(rows); assert len(frame)==1396*14 and frame.source_id.nunique()==1396
    frame.to_csv(target,index=False)
    write_json(OUT/'unmarked_test_preparation.json',{'completed_at_utc':utc(),'rows':len(frame),'sources':1396,
        'manifest_sha256':digest(target),'pixels':'Each of the same 14 deterministic transformations applied directly to the unmarked standardized original; no embed operation.',
        'inference_performed':False})

def fetch(entry):
    key=entry['ImageID']; target=DATA/'openimages_raw'/f'{key}.jpg'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        result=subprocess.run(['curl.exe','-sSfL','--retry','1','--connect-timeout','15','--max-time','60','-o',str(target),f'https://open-images-dataset.s3.amazonaws.com/test/{key}.jpg'],capture_output=True)
        if result.returncode: return entry,None,'download_failure'
    return entry,target,''

def prepare_oi(n):
    target=OUT/'openimages_sources.csv'
    if target.exists(): print('Open Images sources already locked',flush=True); return
    old=pd.read_csv(ROOT/'outputs/conference_clean/source_audit.csv',dtype={'ahash':str})
    fresh=pd.read_csv(ROOT/'outputs/extension/fresh/manifest.csv',keep_default_na=False,dtype={'prejpeg_ahash':str})
    hs=[int(x) for x in old.ahash]+[int(x) for x in fresh.prejpeg_ahash]
    seen=set(old.raw_pixel_sha256)|set(fresh.raw_pixel_sha256)
    excluded=set(fresh.loc[fresh.generator.eq('openimages'),'original_id'])
    selected,rejected=[],[]
    partial=OUT/'openimages_sources.partial.csv'
    if partial.exists():
        selected=pd.read_csv(partial,keep_default_na=False,dtype={'ahash':str}).to_dict('records')
        hs.extend(int(x['ahash']) for x in selected); seen.update(x['raw_pixel_sha256'] for x in selected)
    chosen=set(x['original_id'] for x in selected)
    metadata=pd.read_csv(ROOT/'.tools/extension-assets/openimages_test_metadata.csv',keep_default_na=False)
    metadata['order']=metadata.ImageID.map(sid)
    candidates=metadata.loc[~metadata.ImageID.isin(excluded|chosen)].sort_values('order').to_dict('records')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(candidates),32):
            if len(selected)>=n: break
            for entry,path,error in pool.map(fetch,candidates[start:start+32]):
                if len(selected)>=n: break
                key=entry['ImageID']
                if error: rejected.append({'original_id':key,'reason':error}); continue
                raw=cv2.imread(str(path))
                if raw is None or min(raw.shape[:2])<256:
                    rejected.append({'original_id':key,'reason':'invalid_or_small'}); continue
                rotation=float(entry.get('Rotation') or 0)
                if rotation not in (0,90,180,270):
                    rejected.append({'original_id':key,'reason':'unsupported_rotation'}); continue
                if rotation: raw=np.rot90(raw,int(rotation//90)).copy()
                rawhash=hashlib.sha256(str(raw.shape).encode()+raw.tobytes()).hexdigest()
                base=cv2.resize(raw,(256,256),interpolation=cv2.INTER_AREA); h=ahash(base)
                if rawhash in seen or any((h^x).bit_count()<=2 for x in hs):
                    rejected.append({'original_id':key,'reason':'old_fresh_or_new_overlap'}); continue
                identity=sid('openimages|'+key); image=DATA/'openimages_base'/f'{identity}.png'; image.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(image),base): raise ValueError(image)
                selected.append({'image_id':identity,'source_id':identity,'path':str(image),'native_path':str(path),
                    'label':0,'generator':'openimages','collection':'openimages','original_id':key,
                    'raw_pixel_sha256':rawhash,'ahash':str(h),'input_file_sha256':digest(path),'output_file_sha256':digest(image),
                    **{k:v for k,v in entry.items() if k!='order'}})
                hs.append(h); seen.add(rawhash)
            pd.DataFrame(selected).to_csv(partial,index=False)
            pd.DataFrame(rejected).to_csv(OUT/'openimages_rejections.csv',index=False)
            print('New Open Images selected',len(selected),'/',n,flush=True)
    if len(selected)!=n: raise RuntimeError('Not enough new sources')
    frame=pd.DataFrame(selected); frame.to_csv(target,index=False)
    write_json(OUT/'openimages_selection_complete.json',{'completed_at_utc':utc(),'sources':len(frame),'manifest_sha256':digest(target),'inference_performed':False,
        'selection':'SHA256 order by fixed seed; official unsigned S3 Open Images test subset; minimum side 256; exact raw pixel and 8x8 ahash <=2 exclusions against all original 6000, all fresh 192, and earlier selected new images.',
        'processing':'Native pixels read as BGR, metadata rotation applied, INTER_AREA resize to 256 square, PNG; matches Experiment 1 processing. JPEG control remains Experiment 2.'})

def oi_derived_source(row):
    src=row['source_id']; image=cv2.imread(row['path']); marker=DwtDctWatermark(PAYLOAD,12)
    result=[{'image_id':src,'source_id':src,'path':row['path'],'label':0,'generator':'openimages','collection':'openimages',
        'group':'R-C','watermark_method':'','payload':'','attack':'identity','severity':0,'parent_id':'',
        'active_score':marker.detect(image).bit_accuracy,'output_file_sha256':digest(row['path'])}]
    wrong=''.join('1' if x=='0' else '0' for x in PAYLOAD)
    specs=[('R-S','identity',0,DwtDctWatermark(wrong,12).embed(image))]
    specs.extend(('R-T',a['name'],a['severity'],apply_attack(image,a['name'],a['severity'])) for a in attacks())
    for group,name,severity,pixels in specs:
        identity=sid(f'{src}|{group}|{name}|{severity}'); path=DATA/'openimages_derived'/f'{identity}.png'; path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() and not cv2.imwrite(str(path),pixels): raise ValueError(path)
        result.append({'image_id':identity,'source_id':src,'path':str(path),'label':0,'generator':'openimages','collection':'openimages',
            'group':group,'watermark_method':'dwt_dct' if group=='R-S' else '', 'payload':wrong if group=='R-S' else '',
            'attack':name,'severity':severity,'parent_id':src,'active_score':marker.detect(pixels).bit_accuracy,'output_file_sha256':digest(path)})
    return result

def prepare_oi_derived():
    target=OUT/'openimages_derived_manifest.csv'
    if target.exists(): print('Open Images edits already locked',flush=True); return
    src=pd.read_csv(OUT/'openimages_sources.csv',keep_default_na=False); rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,part in enumerate(pool.map(oi_derived_source,src.to_dict('records'))):
            rows.extend(part)
            if i%100==0: print('Open Images derivatives',i+1,'/',len(src),flush=True)
    frame=pd.DataFrame(rows); assert len(frame)==len(src)*16
    frame.to_csv(target,index=False)
    write_json(OUT/'openimages_derivation_complete.json',{'completed_at_utc':utc(),'rows':len(frame),'sources':len(src),'manifest_sha256':digest(target),'inference_performed':False})

def audit_oi():
    """Canonicalize metadata orientation and recheck exclusion before features."""
    completed=OUT/'openimages_orientation_audit.json'
    target=OUT/'openimages_sources.csv'
    if completed.exists():
        audit=json.loads(completed.read_text())
        if audit['manifest_sha256']!=digest(target):raise RuntimeError('Audited Open Images manifest changed')
        print('Open Images orientation audit already locked',flush=True);return
    if any(OUT.glob('*_openimages_embeddings.npz')): raise RuntimeError('Cannot change sources after inference')
    if (OUT/'openimages_derived_manifest.csv').exists():raise RuntimeError('Audit must precede derivative generation')
    frame=pd.read_csv(target,keep_default_na=False,dtype={'ahash':str})
    old=pd.read_csv(ROOT/'outputs/conference_clean/source_audit.csv',dtype={'ahash':str})
    fresh=pd.read_csv(ROOT/'outputs/extension/fresh/manifest.csv',keep_default_na=False,dtype={'prejpeg_ahash':str})
    hs=[int(x) for x in old.ahash]+[int(x) for x in fresh.prejpeg_ahash]
    rawhashes=set(old.raw_pixel_sha256)|set(fresh.raw_pixel_sha256)
    changed=[];kept=[];rejected=[]
    for row in frame.to_dict('records'):
        raw=cv2.imread(row['native_path']);rotation=float(row.get('Rotation') or 0)
        if rotation:raw=np.rot90(raw,int(rotation//90)).copy()
        rawhash=hashlib.sha256(str(raw.shape).encode()+raw.tobytes()).hexdigest()
        base=cv2.resize(raw,(256,256),interpolation=cv2.INTER_AREA);h=ahash(base)
        if rawhash in rawhashes or any((h^v).bit_count()<=2 for v in hs):
            rejected.append({'original_id':row['original_id'],'reason':'canonical_orientation_overlap'});continue
        if rawhash!=row['raw_pixel_sha256']:
            cv2.imwrite(row['path'],base);changed.append(row['original_id'])
        row.update(raw_pixel_sha256=rawhash,ahash=str(h),output_file_sha256=digest(row['path']))
        kept.append(row);hs.append(h);rawhashes.add(rawhash)
    corrected=pd.DataFrame(kept);corrected.to_csv(target,index=False);corrected.to_csv(OUT/'openimages_sources.partial.csv',index=False)
    write_json(OUT/'openimages_orientation_audit.json',{'completed_at_utc':utc(),'corrected_metadata_orientation_IDs':changed,
        'rotation_rule':'Open Images metadata specifies counterclockwise degrees; numpy.rot90 uses positive k.',
        'rotation_primary_source':'https://storage.googleapis.com/openimages/web/download_v7.html',
        'overlap_exclusions':rejected,'sources':len(kept),'inference_performed':False,'manifest_sha256':digest(target),'script_sha256':digest(__file__)})
    if rejected:
        target.unlink();completed.unlink();prepare_oi(len(frame));audit_oi()
    else:
        complete=json.loads((OUT/'openimages_selection_complete.json').read_text());complete['orientation_audit_at_utc']=utc();complete['manifest_sha256']=digest(target)
        write_json(OUT/'openimages_selection_complete.json',complete)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['unmarked','openimages','audit','derive','all'],default='all');ap.add_argument('--openimages-count',type=int,default=1000); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'protocol.json'
    if not protocol.exists(): write_json(protocol,{'frozen_at_utc':utc(),'seed':SEED,'split_seeds':[20260920,20260921,20260922,20260923,20260924],
        'new_openimages_sources':args.openimages_count,'planned_openimages_splits':[700,150,150],
        'unmarked_test_sources':1396,'unmarked_test_edits':19544,'gate_threshold':13/16,
        'holdout_generator':'midjourney','split_study_conditions':'13 common conditions, same for every split; the three originally unseen attacks are not used in refitting.',
        'fresh_192_excluded':True,'interpretation':'Diagnostic retrospective re-splits of already examined sources; not a new untouched generalization test.',
        'script_sha256':digest(__file__),'original_heads_sha256':{p.name:digest(p) for p in (ROOT/'outputs/conference_clean').glob('*.npz')}})
    if args.stage in ('unmarked','all'): prepare_unmarked()
    if args.stage in ('openimages','all'): prepare_oi(args.openimages_count)
    if args.stage in ('audit','all'): audit_oi()
    if args.stage in ('derive','all'): prepare_oi_derived()

if __name__=='__main__': main()
