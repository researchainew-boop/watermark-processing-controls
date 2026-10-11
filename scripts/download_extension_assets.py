"""Download official published models and Open Images metadata, archive hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.tools/extension-assets'
OUT.mkdir(parents=True,exist_ok=True)

def download(url,name):
    target = OUT/name
    if not target.exists():
        part = target.with_suffix(target.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=90) as source, part.open('wb') as out:
            while chunk := source.read(1024*1024):
                out.write(chunk)
        part.replace(target)
    h = hashlib.sha256(target.read_bytes()).hexdigest()
    print(name,target.stat().st_size,h,flush=True)
    return {'url':url,'filename':name,'bytes':target.stat().st_size,'sha256':h}

def main():
    req = urllib.request.Request('https://api.github.com/repos/WisconsinAIVision/UniversalFakeDetect/commits/main',headers={'User-Agent':'research-reproducibility'})
    commit = json.load(urllib.request.urlopen(req))['sha']
    base = f'https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/{commit}/'
    assets = [('https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt','ViT-L-14.pt'),
        (base+'pretrained_weights/fc_weights.pth','univfd_fc_weights.pth'),
        ('https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv','openimages_test_metadata.csv'),
        (base+'models/clip/model.py','official_clip_model.py'),
        (base+'validate.py','official_univfd_validate.py')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item:download(*item),assets))
    model = next(x for x in results if x['filename']=='ViT-L-14.pt')
    assert model['sha256']=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    (OUT/'provenance.json').write_text(json.dumps({'univfd_commit':commit,'assets':results},indent=2),encoding='utf-8')

if __name__ == '__main__': main()
