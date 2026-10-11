"""Lock a new source sample before inference, including independent Open Images.

Candidates are ordered by SHA256(seed|collection|ID), rejected only for
availability, size, or conservative pixel/hash overlap with all old sources.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import zipfile
import urllib.request
import subprocess
import numpy as np
import pandas as pd
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/extension/fresh'
DATA = ROOT/'data/extension_fresh'
RAW = ROOT/'data/raw'
SEED = 20261009

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def order(collection,key): return hashlib.sha256(f'{SEED}|{collection}|{key}'.encode()).hexdigest()
def ahash(pixels):
    small = cv2.resize(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY),(8,8),interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((small>=small.mean()).ravel()).tobytes(),'big')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.csv').exists():
        print('Fresh manifest already frozen; no reselection performed.',flush=True)
        return
    fits = list((ROOT/'outputs/conference_clean').glob('*.npz'))
    protocol = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'planned_sources':{'openimages':48,'coco':48,'dalle-3':96},
        'selection':'SHA256 order; source availability/minimum side256/old and within-new hash screening only',
        'models_sha256':{p.name:digest(p) for p in fits},
        'gate_threshold':13/16,'jpeg_quality':95,'jpeg_library':cv2.__version__,
        'processing':'RGB->BGR, INTER_AREA resize256x256, OpenCV JPEG95 encode/decode for every source',
        'fresh_ai_scope':'Unused items in the same DALL-E3 archive, not an unseen synthetic collection',
        'tuning_on_this_sample':False,'untouched_means':'No prior detector inference on these sources; pretrained-model overlap cannot be ruled out',
        'inference_passes_allowed':1,'script_sha256':digest(__file__)}
    freeze = OUT/'protocol.json'
    if not freeze.exists(): freeze.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    old = pd.read_csv(ROOT/'outputs/conference_clean/source_audit.csv',dtype={'ahash':str})
    hashes = [int(x) for x in old.ahash]
    seenraw = set(old.raw_pixel_sha256)
    rows,rejections = [],[]

    def consider(collection,key,payload,metadata):
        im = Image.open(io.BytesIO(payload)).convert('RGB')
        if min(im.size)<256: return False,'small'
        rotation = metadata.get('Rotation',0)
        if rotation and not pd.isna(rotation): im = im.rotate(-float(rotation),expand=True)
        pixels = cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)
        rawhash = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
        base = cv2.resize(pixels,(256,256),interpolation=cv2.INTER_AREA)
        h = ahash(base)
        if rawhash in seenraw or any((h^x).bit_count()<=2 for x in hashes): return False,'old_or_new_hash_overlap'
        ok,jpeg = cv2.imencode('.jpg',base,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok: raise ValueError('JPEG encode failure')
        decoded = cv2.imdecode(jpeg,cv2.IMREAD_COLOR)
        source = order(collection,key)[:24]
        target = DATA/f'{source}.png'
        cv2.imwrite(str(target),decoded)
        rows.append({'image_id':source,'source_id':source,'path':str(target),'label':int(collection=='dalle-3'),
            'generator':collection,'collection':collection,'split':'fresh_test','group':'S-U' if collection=='dalle-3' else 'R-C',
            'attack':'identity','severity':0,'original_id':key,'input_file_sha256':hashlib.sha256(payload).hexdigest(),
            'raw_pixel_sha256':rawhash,'prejpeg_ahash':str(h),'output_file_sha256':digest(target),'jpeg_quality':95,**metadata})
        hashes.append(h); seenraw.add(rawhash)
        return True,''

    for collection,archive,n in [('dalle-3',RAW/'downloads/Dalle3-dataset.zip',96),('coco',RAW/'val2017.zip',48)]:
        with zipfile.ZipFile(archive) as z:
            names = [x for x in z.namelist() if x.lower().endswith(('.jpg','.png','.jpeg','.webp')) and 'thumb' not in x.lower()]
            selected = 0
            for name in sorted(names,key=lambda x:order(collection,x)):
                if selected==n: break
                try: accepted,reason = consider(collection,name,z.read(name),{'archive':str(archive)})
                except Exception as e: accepted,reason=False,type(e).__name__
                if accepted: selected+=1
                else: rejections.append({'collection':collection,'id':name,'reason':reason})
            if selected!=n: raise RuntimeError(f'Only {selected}/{n} sources available in {collection}')
        print(collection,selected,flush=True)
    metadata = pd.read_csv(ROOT/'.tools/extension-assets/openimages_test_metadata.csv',keep_default_na=False)
    metadata['selection_order'] = metadata.ImageID.map(lambda x:order('openimages',x))
    selected = 0
    for entry in metadata.sort_values('selection_order').to_dict('records'):
        if selected==48: break
        key = entry['ImageID']
        url = f'https://open-images-dataset.s3.amazonaws.com/test/{key}.jpg'
        try:
            native = DATA/'raw_openimages'/f'{key}.jpg'
            native.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['curl.exe','-sSfL','--connect-timeout','15','--max-time','45','-o',str(native),url],check=True)
            payload = native.read_bytes()
            accepted,reason = consider('openimages',key,payload,{k:v for k,v in entry.items() if k!='selection_order'}|{'download_url':url})
        except Exception as e: accepted,reason=False,type(e).__name__
        if accepted:
            selected+=1
            if selected%8==0: print('Open Images',selected,flush=True)
        else: rejections.append({'collection':'openimages','id':key,'reason':reason})
    if selected!=48: raise RuntimeError('Insufficient Open Images sources')
    pd.DataFrame(rejections).to_csv(OUT/'selection_rejections.csv',index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'manifest.csv',index=False)
    (OUT/'selection_complete.json').write_text(json.dumps({'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'manifest_sha256':digest(OUT/'manifest.csv'),'sources':len(frame),'model_inference_performed':False},indent=2),encoding='utf-8')
    print('Locked fresh sources',len(frame),flush=True)

if __name__=='__main__': main()
