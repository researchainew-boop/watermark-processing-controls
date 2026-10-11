"""Verify fresh DALL-E archive IDs lie outside the exact old prefix selection."""
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    path=ROOT.parent/'synthetic-image-watermark-study/data/raw/downloads/Dalle3-dataset.zip'
    original=[]
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(('.jpg','.jpeg','.png','.webp')) or 'thumb' in name.lower() or name.endswith('/'): continue
            im=Image.open(io.BytesIO(z.read(name)))
            if min(im.size)<256: continue
            original.append(name)
            if len(original)==1000: break
    fresh=pd.read_csv(ROOT/'outputs/extension/fresh/manifest.csv',keep_default_na=False)
    new=set(fresh.loc[fresh.collection=='dalle-3','original_id'])
    overlap=set(original)&new
    report={'reconstructed_original_prefix_count':len(original),'fresh_dalle_count':len(new),'original_archive_id_overlap':sorted(overlap),
        'selection_rule':'Exact ordered qualifying archive entries used by prepare_native_sources.py'}
    (ROOT/'outputs/extension/fresh/archive_identity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report,flush=True)
    assert len(original)==1000 and not overlap

if __name__=='__main__': main()
