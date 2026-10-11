"""Prospectively locked replication on unused native archive/S3 items.

Never selects sources from detector scores or changes the frozen heads.
Same-collection replication is not an unseen-generator benchmark.
"""
from __future__ import annotations
import argparse, gc, hashlib, io, json, subprocess, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy.stats import beta
from threadpoolctl import threadpool_limits
from evaluate_encoding_controls import digest as file_digest, checkpoint_defaults, load_backbone, infer, model_scores, cached_features, original_caches

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase4/fresh_replication'
DATA=ROOT/'data/phase4/fresh_replication'
RAW=ROOT/'data/raw'
HEADS=ROOT/'outputs/phase3/controls/split_seeds/20260920'
CAL=ROOT/'outputs/phase4/domain_generalization/frozen_seed_thresholds.json'
SEED=20261010
PAYLOAD='1011010010110100'
AI={'stable-diffusion','dalle-3','midjourney'}
COLLECTIONS=['coco','openimages','stable-diffusion','dalle-3','midjourney']
sys.path.insert(0,str(ROOT/'src'))
from aiwstudy.attacks import apply_attack
from aiwstudy.watermark import DwtDctWatermark
cv2.setNumThreads(1)

def utc(): return datetime.now(timezone.utc).isoformat()
def digest(path):return file_digest(Path(path))
def write(p,o): Path(p).write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def order(c,k): return hashlib.sha256(f'phase4|{SEED}|{c}|{k}'.encode()).hexdigest()
def ahash(a):
    b=cv2.resize(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),(8,8),interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((b>=b.mean()).ravel()).tobytes(),'big')

def verify_calibration_heads():
    if not CAL.exists(): raise RuntimeError('Frozen validation calibration must precede source selection')
    calibration=json.loads(CAL.read_text(encoding='utf-8'))
    expected={f'{backbone}{suffix}_full.npz' for backbone in ['resnet','clip'] for suffix in ['','_fusion']}
    recorded={Path(k).name:v for k,v in calibration['head_sha256'].items()}
    if set(recorded)!=expected:raise RuntimeError('Calibration must identify exactly the four frozen seed heads')
    current={name:digest(HEADS/name) for name in sorted(expected)}
    if current!=recorded:raise RuntimeError('Current seed heads differ from frozen calibration heads')
    return current

def prior_native_ids(old,fresh,oi):
    """Replay original eligible export IDs; JPEG re-saving makes raw hashes insufficient."""
    target=OUT/'excluded_native_ids.csv';audit_path=OUT/'native_exclusion_audit.json'
    source_paths=[ROOT/'outputs/conference_clean/source_audit.csv',ROOT/'outputs/extension/fresh/manifest.csv',
                  ROOT/'outputs/phase3/controls/openimages_sources.csv',ROOT/'scripts/prepare_native_sources.py']
    inputs={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in source_paths}
    archives=[RAW/'downloads/Dalle3-dataset.zip']+[RAW/'downloads'/f'mj-{i:03d}.parquet' for i in range(3)]
    inputs.update({p.name:digest(p) for p in archives})
    if target.exists():
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if audit['input_sha256']!=inputs or audit['excluded_ids_sha256']!=digest(target):
            raise RuntimeError('Frozen native exclusions or their input archives changed')
        frozen=pd.read_csv(target,keep_default_na=False)
        return set(zip(frozen.collection,frozen.original_id)),audit
    records=[]
    def add(c,key,reason):records.append({'collection':c,'original_id':key,'exclusion_reason':reason})
    for path in old.loc[old.label.eq(0),'path']:
        add('coco','val2017.zip::val2017/'+Path(path).name,'original_6000_source_pool')
    for r in fresh.to_dict('records'):
        key=r['original_id']
        if r['collection']!='openimages':key=Path(r['archive']).name+'::'+key
        add(r['collection'],key,'earlier_frozen_fresh_sample')
    for key in oi.original_id:add('openimages',key,'phase3_openimages_training_pool')
    written=0
    with zipfile.ZipFile(archives[0]) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(('.jpg','.jpeg','.png','.webp')) or 'thumb' in name.lower() or name.endswith('/'):continue
            with Image.open(io.BytesIO(archive.read(name))) as image:
                image.load()
                if min(image.size)<256:continue
                image.convert('RGB')
            add('dalle-3',archives[0].name+'::'+name,'original_first1000_eligible_native_exports')
            written+=1
            if written==1000:break
    if written!=1000:raise RuntimeError('Original DALL-E source export could not be replayed')
    import pyarrow.parquet as pq
    written=0
    for path in archives[1:]:
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16,columns=['image','label']):
            images=batch.column('image').to_pylist();labels=batch.column('label').to_pylist()
            for i,(cell,label) in enumerate(zip(images,labels)):
                if int(label)!=0:continue
                payload=cell.get('bytes') if isinstance(cell,dict) else cell
                if not payload:continue
                with Image.open(io.BytesIO(payload)) as image:
                    if min(image.size)<256:continue
                    image.convert('RGB')
                add('midjourney',path.name+'::'+str(offset+i),'original_first1000_eligible_native_exports')
                written+=1
                if written==1000:break
            offset+=len(batch)
            if written==1000:break
        if written==1000:break
    if written!=1000:raise RuntimeError('Original Midjourney source export could not be replayed')
    frozen=pd.DataFrame(records).sort_values(['collection','original_id','exclusion_reason']).drop_duplicates(['collection','original_id'])
    frozen.to_csv(target,index=False)
    audit={'completed_at_utc':utc(),'input_sha256':inputs,'excluded_ids_sha256':digest(target),
           'unique_excluded_native_ids':len(frozen),'excluded_ids_per_collection':frozen.collection.value_counts().to_dict(),
           'Dalle_original_eligible_exports_replayed':1000,'Midjourney_original_eligible_exports_replayed':1000,
           'source_hash_screen_count':len(old)+len(fresh)+len(oi),
           'rule':'Original ZIP order/ordered parquet rows; same label0, embedded-image and min-side256 eligibility as original native export. Also exclude all original COCO basenames and earlier fresh/OI IDs. New SD part3 shares no archive ID with original part1/part2, but all sources still undergo exact/aHash exclusion.',
           'detector_inference_performed':False}
    write(audit_path,audit)
    return set(zip(frozen.collection,frozen.original_id)),audit

def freeze():
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    model_hashes=verify_calibration_heads()
    if (OUT/'protocol.json').exists():
        saved=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        if saved['model_hashes']!=model_hashes or saved['calibration_sha256']!=digest(CAL):raise RuntimeError('Frozen replication/calibration changed')
        return
    write(OUT/'protocol.json',{'frozen_at_utc':utc(),'seed':SEED,'sources_per_collection':80,
        'collections':COLLECTIONS,'primary_heads':'Existing joint-collection seed20260920 full heads; no new fitting',
        'model_hashes':model_hashes,
        'calibration_path':str(CAL),'calibration_sha256':digest(CAL),
        'selection':'SHA256 order of native archive IDs; minimum short side256; explicitly exclude original eligible DALL-E/MJ exports, original COCO IDs and earlier fresh/OI IDs, then reject exact pixels or average-hash Hamming<=2 overlap with all prior sources and within this sample; no score-based rejection',
        'explicit_native_id_exclusions':True,'calibration_head_hashes_verified':True,
        'preprocessing':'PIL RGB; official metadata rotation for Open Images; INTER_AREA 256 square; OpenCV JPEG95 encode/decode equally for all origins; PNG lossless staging',
        'conditions':[{'name':'identity','severity':0},{'name':'jpeg','severity':2},{'name':'crop','severity':1},{'name':'lowres','severity':2}],
        'AI_variants':'Unmarked and marked independently per condition; mark applied after common JPEG95',
        'real_variants':'Unmarked only; identical four conditions',
        'payload':PAYLOAD,'strength':12,'gate_threshold':13/16,
        'planned_source_count':400,'planned_image_count':2560,'bootstrap_seed':20261011,'bootstrap_replicates':2000,
        'bootstrap_reporting':'Paired AUC differences, marginal AUC, and fixed-threshold TPR/FPR source-cluster percentile95% intervals; common source draws across systems.',
        'test_inference_passes':1,'test_retuning':False,'scope':'New source items from existing collections; no prior study inference on selected sources; pretraining and semantic overlap not excluded',
        'script_sha256':digest(__file__)})

def select():
    freeze()
    if (OUT/'sources.csv').exists(): print('Source manifest locked; no reselection',flush=True); return
    old=pd.read_csv(ROOT/'outputs/conference_clean/source_audit.csv',dtype={'ahash':str})
    fresh=pd.read_csv(ROOT/'outputs/extension/fresh/manifest.csv',dtype={'prejpeg_ahash':str},keep_default_na=False)
    oi=pd.read_csv(ROOT/'outputs/phase3/controls/openimages_sources.csv',dtype={'ahash':str},keep_default_na=False)
    excluded_ids,exclusion_audit=prior_native_ids(old,fresh,oi)
    hashes=[int(x) for x in old.ahash]+[int(x) for x in fresh.prejpeg_ahash]+[int(x) for x in oi.ahash]
    exact=set(old.raw_pixel_sha256)|set(fresh.raw_pixel_sha256)|set(oi.raw_pixel_sha256)
    rows=[]; rejections=[]; partial=OUT/'sources.partial.csv'
    if (OUT/'selection_rejections.csv').exists():
        prior_rejections=pd.read_csv(OUT/'selection_rejections.csv',keep_default_na=False)
        rejections=prior_rejections.to_dict('records')
    if partial.exists():
        rows=pd.read_csv(partial,keep_default_na=False,dtype={'prejpeg_ahash':str}).to_dict('records')
        hashes.extend(int(x['prejpeg_ahash']) for x in rows);exact.update(x['raw_pixel_sha256'] for x in rows)
    chosen={(r['collection'],r['original_id']) for r in rows}
    if chosen & excluded_ids:raise RuntimeError('Partial sample contains a previously used native ID')
    def count(c):return sum(r['collection']==c for r in rows)
    def consider(c,key,payload,metadata):
        if (c,key) in chosen:return False
        if (c,key) in excluded_ids:
            rejections.append({'collection':c,'original_id':key,'reason':'previously_used_native_id'});return False
        try:
            im=Image.open(io.BytesIO(payload)).convert('RGB')
            rotation=float(metadata.get('Rotation') or 0)
            if rotation not in (0,90,180,270):raise ValueError('unsupported_metadata_rotation')
            if rotation: im=im.rotate(rotation,expand=True)
            if min(im.size)<256:raise ValueError('short_side_below256')
            pixels=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
            rawhash=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
            base=cv2.resize(pixels,(256,256),interpolation=cv2.INTER_AREA); h=ahash(base)
            if rawhash in exact or any((h^x).bit_count()<=2 for x in hashes):raise ValueError('prior_or_within_sample_hash_overlap')
            ok,jpeg=cv2.imencode('.jpg',base,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok:raise RuntimeError('JPEG encoding failed')
            sid=order(c,key)[:24];path=DATA/'base'/f'{sid}.png';path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path),cv2.imdecode(jpeg,cv2.IMREAD_COLOR)):raise RuntimeError('PNG write failed')
            rows.append({'source_id':sid,'image_id':sid,'collection':c,'generator':c,'label':int(c in AI),'original_id':key,
                'path':str(path),'input_file_sha256':hashlib.sha256(payload).hexdigest(),'raw_pixel_sha256':rawhash,
                'prejpeg_ahash':str(h),'output_file_sha256':digest(path),**metadata})
            hashes.append(h);exact.add(rawhash);chosen.add((c,key));return True
        except (ValueError,OSError) as e:
            rejections.append({'collection':c,'original_id':key,'reason':str(e)});return False
    def checkpoint():
        pd.DataFrame(rows).to_csv(partial,index=False);pd.DataFrame(rejections).to_csv(OUT/'selection_rejections.csv',index=False)
    archives={'coco':[RAW/'val2017.zip'],'stable-diffusion':[ROOT/'.tools/extension-assets/part-000003.zip'],
              'dalle-3':[RAW/'downloads/Dalle3-dataset.zip']}
    for c,paths in archives.items():
        if count(c)==80:continue
        with __import__('contextlib').ExitStack() as stack:
            zs={p.name:stack.enter_context(zipfile.ZipFile(p)) for p in paths}
            candidates=[(p.name,n) for p in paths for n in zs[p.name].namelist() if n.lower().endswith(('.jpg','.png','.jpeg','.webp')) and 'thumb' not in n.lower()]
            for archive,name in sorted(candidates,key=lambda v:order(c,v[0]+'::'+v[1])):
                if count(c)==80:break
                consider(c,archive+'::'+name,zs[archive].read(name),{'archive':str(next(p for p in paths if p.name==archive))})
            if count(c)!=80:raise RuntimeError(f'Insufficient unused {c} items: {count(c)}')
        checkpoint();print('Selected',c,count(c),flush=True)
    if count('midjourney')<80:
        import pyarrow.parquet as pq
        candidates=[]
        for p in sorted((RAW/'downloads').glob('mj-*.parquet')):
            pf=pq.ParquetFile(p);offset=0
            for batch in pf.iter_batches(batch_size=512,columns=['label']):
                for i,label in enumerate(batch.column(0).to_pylist()):
                    if int(label)==0:candidates.append((p,offset+i))
                offset+=len(batch)
        # Read only the next deterministic candidate blocks, never inspect scores.
        for start in range(0,len(candidates),128):
            if count('midjourney')==80:break
            eligible=[v for v in candidates if ('midjourney',v[0].name+'::'+str(v[1])) not in excluded_ids]
            chunk=sorted(eligible,key=lambda v:order('midjourney',v[0].name+'::'+str(v[1])))[start:start+128]
            if not chunk:break
            payloads={}
            for p in set(x[0] for x in chunk):
                wanted={idx for path,idx in chunk if path==p};offset=0
                for batch in pq.ParquetFile(p).iter_batches(batch_size=16,columns=['image']):
                    selected=[(idx,idx-offset) for idx in wanted if offset<=idx<offset+len(batch)]
                    for idx,j in selected:
                        cell=batch.column(0)[j].as_py();payloads[p,idx]=cell['bytes'] if isinstance(cell,dict) else cell
                    offset+=len(batch)
                    if offset>max(wanted):break
            for p,idx in chunk:
                if count('midjourney')==80:break
                consider('midjourney',p.name+'::'+str(idx),payloads[p,idx],{'archive':str(p),'archive_row':idx})
            checkpoint();print('Selected midjourney',count('midjourney'),flush=True)
        if count('midjourney')!=80:raise RuntimeError('Insufficient unused Midjourney items')
    if count('openimages')<80:
        metadata=pd.read_csv(ROOT/'.tools/extension-assets/openimages_test_metadata.csv',keep_default_na=False)
        excluded={key for c,key in excluded_ids if c=='openimages'}
        entries=metadata.loc[~metadata.ImageID.isin(excluded)].to_dict('records')
        entries=sorted(entries,key=lambda e:order('openimages',e['ImageID']))
        def fetch(e):
            key=e['ImageID'];path=DATA/'raw_openimages'/f'{key}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                temporary=path.with_suffix('.download')
                result=subprocess.run(['curl.exe','-sSfL','--retry','1','--connect-timeout','10','--max-time','30','-o',str(temporary),f'https://open-images-dataset.s3.amazonaws.com/test/{key}.jpg'],capture_output=True)
                if result.returncode:return e,None
                temporary.replace(path)
            return e,path.read_bytes()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(entries),32):
                if count('openimages')==80:break
                for e,payload in pool.map(fetch,entries[start:start+32]):
                    if count('openimages')==80:break
                    if payload is None:rejections.append({'collection':'openimages','original_id':e['ImageID'],'reason':'download_failure'});continue
                    consider('openimages',e['ImageID'],payload,{k:v for k,v in e.items() if k in ['Rotation','License','ImageID']})
                checkpoint();print('Selected openimages',count('openimages'),flush=True)
        if count('openimages')!=80:raise RuntimeError('Insufficient unused Open Images items')
    frame=pd.DataFrame(rows);assert len(frame)==400 and frame.source_id.nunique()==400
    assert not set(zip(frame.collection,frame.original_id)) & excluded_ids
    frame.to_csv(OUT/'sources.csv',index=False)
    write(OUT/'selection_complete.json',{'completed_at_utc':utc(),'sources':400,'sources_per_collection':frame.collection.value_counts().to_dict(),
        'manifest_sha256':digest(OUT/'sources.csv'),'protocol_sha256':digest(OUT/'protocol.json'),'prior_source_count':len(old)+len(fresh)+len(oi),
        'native_exclusion_audit_sha256':digest(OUT/'native_exclusion_audit.json'),'prior_native_id_intersections':0,
        'exact_or_ahash_overlap_screen_passed':True,'source_hashes_screened':len(old)+len(fresh)+len(oi),
        'detector_inference_performed':False})

def derive():
    if (OUT/'manifest.csv').exists():return
    sources=pd.read_csv(OUT/'sources.csv',keep_default_na=False);marker=DwtDctWatermark(PAYLOAD,12);rows=[]
    selection=json.loads((OUT/'selection_complete.json').read_text(encoding='utf-8'))
    assert digest(OUT/'sources.csv')==selection['manifest_sha256'] and selection['sources']==400
    for src in sources.to_dict('records'):
        image=cv2.imread(src['path']);assert digest(src['path'])==src['output_file_sha256']
        covers=[('unmarked',image)]+([('marked',marker.embed(image))] if src['label']==1 else [])
        for name,severity in [('identity',0),('jpeg',2),('crop',1),('lowres',2)]:
            for mark,cover in covers:
                pixels=apply_attack(cover,name,severity);iid=order(src['source_id'],f'{mark}|{name}|{severity}')[:24]
                path=DATA/'derived'/f'{iid}.png';path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),pixels):raise RuntimeError(path)
                rows.append({'image_id':iid,'source_id':src['source_id'],'path':str(path),'collection':src['collection'],'generator':src['generator'],
                    'label':src['label'],'mark':mark,'attack':name,'severity':severity,'active_score':marker.detect(pixels).bit_accuracy,'output_file_sha256':digest(path)})
    frame=pd.DataFrame(rows);assert len(frame)==2560 and frame.image_id.nunique()==2560
    assert sources.collection.value_counts().to_dict()=={c:80 for c in COLLECTIONS}
    assert frame.groupby('source_id').size().to_dict()==sources.set_index('source_id').label.map({0:4,1:8}).to_dict()
    frame.to_csv(OUT/'manifest.csv',index=False);write(OUT/'derivation_complete.json',{'completed_at_utc':utc(),'images':len(frame),'manifest_sha256':digest(OUT/'manifest.csv'),'inference_performed':False})

def extract(device,batch_size):
    import torch
    torch.set_num_threads(2)
    frame=pd.read_csv(OUT/'manifest.csv',keep_default_na=False); _,checkpoint=checkpoint_defaults();protocol=json.loads((OUT/'protocol.json').read_text())
    assert all(digest(HEADS/k)==v for k,v in protocol['model_hashes'].items())
    anchors=pd.read_csv(ROOT/'outputs/conference_clean/manifest_derived.csv',keep_default_na=False)
    anchors=anchors.loc[anchors.group.eq('R-C')|anchors.group.eq('S-U')].groupby('generator').head(1)
    for name in ['resnet','clip']:
        path=OUT/f'{name}_embeddings.npz'
        if path.exists():
            with np.load(path,allow_pickle=True) as z:assert str(z['manifest_sha256'])==digest(OUT/'manifest.csv') and np.array_equal(z['image_id'].astype(str),frame.image_id)
            continue
        model,transform,forward,preproc,identity=load_backbone(name,device,checkpoint)
        current=infer(anchors,transform,forward,device,batch_size).astype(np.float64)
        cached=cached_features(original_caches()[:1] if name=='resnet' else original_caches()[1:],anchors.image_id.tolist()).astype(np.float64)
        cosine=np.sum(current*cached,axis=1)/(np.linalg.norm(current,axis=1)*np.linalg.norm(cached,axis=1))
        relative=np.linalg.norm(current-cached,axis=1)/np.linalg.norm(cached,axis=1)
        assert cosine.min()>.99999 and relative.max()<.005
        for r in frame.to_dict('records'):assert digest(r['path'])==r['output_file_sha256']
        test_started=utc()
        write(OUT/f'{name}_inference_started.json',{'test_inference_started_at_utc':test_started,
            'manifest_sha256':digest(OUT/'manifest.csv'),'script_sha256':digest(__file__),
            'device':device,'images':len(frame),'model_hashes':protocol['model_hashes']})
        vectors=infer(frame,transform,forward,device,batch_size)
        np.savez(path,image_id=frame.image_id.to_numpy(),embedding=vectors,manifest_sha256=digest(OUT/'manifest.csv'))
        write(OUT/f'{name}_inference.json',{'completed_at_utc':utc(),'test_inference_started_at_utc':test_started,'device':device,'model':identity,'preprocessing':preproc,
            'images':len(frame),'manifest_sha256':digest(OUT/'manifest.csv'),'embedding_sha256':digest(path),'anchor_cosine':cosine.tolist(),'anchor_relative_l2':relative.tolist(),'checkpoint_sha256':digest(checkpoint) if name=='clip' else None})
        del model,forward,transform,vectors;gc.collect();torch.cuda.empty_cache()

def exact(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def weighted_auc_draws(y,score,source_index,counts):
    """Exact tied-rank AUC for bootstrap source multiplicities, without refitting."""
    order_idx=np.argsort(score,kind='stable');s=score[order_idx];labels=y[order_idx];ids=source_index[order_idx]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];draws=[]
    for start in range(0,len(counts),32):
        weight=counts[start:start+32,ids]
        positives=np.add.reduceat(weight*labels,starts,axis=1)
        negatives=np.add.reduceat(weight*(1-labels),starts,axis=1)
        below=np.cumsum(negatives,axis=1)-negatives
        draws.extend(((positives*(below+.5*negatives)).sum(axis=1)/(positives.sum(axis=1)*negatives.sum(axis=1))).tolist())
    return np.asarray(draws)

def rate_draws(y,prediction,source_index,counts,label):
    chosen=y==label
    numerator=np.bincount(source_index[chosen],weights=prediction[chosen],minlength=counts.shape[1])
    denominator=np.bincount(source_index[chosen],minlength=counts.shape[1])
    return counts@numerator/(counts@denominator)

def analyze():
    frame=pd.read_csv(OUT/'manifest.csv',keep_default_na=False);protocol=json.loads((OUT/'protocol.json').read_text());cal=json.loads(CAL.read_text())
    assert digest(CAL)==protocol['calibration_sha256'] and all(digest(HEADS/k)==v for k,v in protocol['model_hashes'].items())
    # Calibration file schema is inspected before evaluation; no test score tuning.
    thresholds=cal['systems']
    for name in ['resnet','clip']:
        with np.load(OUT/f'{name}_embeddings.npz',allow_pickle=True) as z:
            assert np.array_equal(z['image_id'].astype(str),frame.image_id)
            assert str(z['manifest_sha256'])==digest(OUT/'manifest.csv');x=z['embedding']
        p,logit=model_scores(x,HEADS/f'{name}_full.npz');fp,fl=model_scores(np.column_stack([p,frame.active_score]),HEADS/f'{name}_fusion_full.npz')
        frame[name+'_probe']=logit;frame[name+'_fusion']=fl;frame[name+'_gated']=np.where(frame.active_score>=13/16,fl,logit)
    frame.to_csv(OUT/'predictions.csv',index=False)
    records=[];paired=[];saved_draws={}
    source_ids=np.sort(frame.source_id.unique());source_lookup={sid:i for i,sid in enumerate(source_ids)}
    source_collections=frame[['source_id','collection']].drop_duplicates().set_index('source_id').collection
    groups=[source_ids[source_collections.loc[source_ids].to_numpy()==c] for c in COLLECTIONS]
    rng=np.random.default_rng(20261011);counts=np.zeros((2000,len(source_ids)),dtype=np.float64)
    for rep in range(2000):
        for group in groups:
            chosen=[source_lookup[sid] for sid in rng.choice(group,len(group),replace=True)]
            counts[rep]+=np.bincount(chosen,minlength=len(source_ids))
    saved_draws['source_ids']=source_ids;saved_draws['source_multiplicities']=counts
    for attack in ['overall','identity','jpeg','crop','lowres']:
        for mark in ['unmarked','marked']:
            part=frame.loc[(frame.mark.eq(mark)|frame.label.eq(0)) & (True if attack=='overall' else frame.attack.eq(attack))]
            ids=part.source_id.to_numpy();collections=part.collection.to_numpy();y=part.label.to_numpy()
            source_index=np.array([source_lookup[sid] for sid in ids])
            scores={name+'_'+s:part[name+'_'+s].to_numpy() for name in ['resnet','clip'] for s in ['probe','fusion','gated']}
            predictions={system:a>=float(thresholds[system]['threshold_logit']) for system,a in scores.items()}
            boot=np.column_stack([weighted_auc_draws(y,a,source_index,counts) for a in scores.values()])
            boot_tpr=np.column_stack([rate_draws(y,a,source_index,counts,1) for a in predictions.values()])
            boot_fpr=np.column_stack([rate_draws(y,a,source_index,counts,0) for a in predictions.values()])
            ci=np.percentile(boot,[2.5,97.5],axis=0)
            tpr_ci=np.percentile(boot_tpr,[2.5,97.5],axis=0);fpr_ci=np.percentile(boot_fpr,[2.5,97.5],axis=0)
            prefix=f'{attack}_{mark}'
            saved_draws[prefix+'_auc']=boot;saved_draws[prefix+'_tpr']=boot_tpr;saved_draws[prefix+'_fpr']=boot_fpr
            columns={system:j for j,system in enumerate(scores)}
            for backbone in ['resnet','clip']:
                for left,right in [('gated','probe'),('fusion','probe'),('gated','fusion')]:
                    l,r=backbone+'_'+left,backbone+'_'+right
                    paired.append({'attack':attack,'AI_mark':mark,'backbone':backbone,'comparison':left+'-minus-'+right,
                        'sources':int(part.source_id.nunique()),'images':len(part),
                        'auc_difference':float(roc_auc_score(y,scores[l])-roc_auc_score(y,scores[r])),
                        'auc_difference_ci95':np.percentile(boot[:,columns[l]]-boot[:,columns[r]],[2.5,97.5]).tolist(),
                        'fixed_threshold_TPR_difference':float(predictions[l][y==1].mean()-predictions[r][y==1].mean()),
                        'fixed_threshold_TPR_difference_ci95':np.percentile(boot_tpr[:,columns[l]]-boot_tpr[:,columns[r]],[2.5,97.5]).tolist(),
                        'fixed_threshold_FPR_difference':float(predictions[l][y==0].mean()-predictions[r][y==0].mean()),
                        'fixed_threshold_FPR_difference_ci95':np.percentile(boot_fpr[:,columns[l]]-boot_fpr[:,columns[r]],[2.5,97.5]).tolist(),
                        'paired_source_draws':True})
            for j,(system,score) in enumerate(scores.items()):
                threshold=float(thresholds[system]['threshold_logit']);prediction=predictions[system]
                record={'attack':attack,'AI_mark':mark,'system':system,'sources':int(part.source_id.nunique()),'images':len(part),
                    'auc':float(roc_auc_score(y,score)),'auc_ci95':ci[:,j].tolist(),'threshold_logit':threshold,
                    'test_TPR':float(prediction[y==1].mean()),'test_TPR_ci95':tpr_ci[:,j].tolist(),
                    'test_FPR':float(prediction[y==0].mean()),'test_FPR_ci95':fpr_ci[:,j].tolist(),'per_collection':{}}
                for c in COLLECTIONS:
                    ix=collections==c;count=int(ix.sum());success=int(prediction[ix].sum());
                    # Overall means include derivatives; exact source intervals only for one copy/source.
                    record['per_collection'][c]={'images':count,'sources':int(part.loc[ix,'source_id'].nunique()),'AI_decisions':success,'AI_decision_rate':success/count,
                        'exact_ci95':exact(success,count) if attack!='overall' else None}
                records.append(record)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**saved_draws,systems=np.array(list(scores)),
        seed=np.array(20261011),replicates=np.array(2000),manifest_sha256=np.array(digest(OUT/'manifest.csv')),
        calibration_sha256=np.array(digest(CAL)))
    write(OUT/'summary.json',{'completed_at_utc':utc(),'source_count':400,'images':2560,'replication':'new items in existing five collections, not new generators',
        'models_refitted':False,'thresholds_retuned':False,'protocol_sha256':digest(OUT/'protocol.json'),'manifest_sha256':digest(OUT/'manifest.csv'),
        'calibration_sha256':digest(CAL),'model_hashes_unchanged':True,'bootstrap':'2000 source-cluster draws stratified by collection; CIs conditional on frozen models',
        'bootstrap_draws_sha256':digest(OUT/'bootstrap_draws.npz'),
        'measurements':records,'paired_comparisons':paired})
    print(json.dumps([r for r in records if r['attack']=='identity'],indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['select','derive','extract','analyze'],required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=16);a=ap.parse_args()
    if a.stage=='select':select()
    if a.stage=='derive':derive()
    if a.stage=='extract':extract(a.device,a.batch_size)
    if a.stage=='analyze':analyze()
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
