"""Expanded, fixed 204-AI/204-real watermark experiment; old run preserved."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
import benchmark_watermarks as base
from prepare_encoding_controls import clean_clip_vectors, cosine_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase3/watermarks'
DATA=ROOT/'data/phase3_watermarks'
SEED=20261012

def select():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selected_pairs.csv').exists(): return
    sources=pd.read_csv(ROOT/'outputs/conference_clean/manifest_sources.csv',keep_default_na=False)
    test=sources.loc[sources.split=='test'].copy()
    selected=[]
    for gen in base.GENERATORS:
        pool=test.loc[test.generator==gen].copy()
        pool['order']=pool.source_id.map(lambda x: hashlib.sha256(f'{SEED}|{x}'.encode()).hexdigest())
        selected.append(pool.sort_values(['order','source_id']).head(68))
    ai=pd.concat(selected).sort_values('source_id').reset_index(drop=True)
    real=test.loc[test.label==0].sort_values('source_id').reset_index(drop=True)
    vectors=clean_clip_vectors(set(ai.source_id)|set(real.source_id))
    sim=cosine_matrix(ai,real,vectors); si,ri=linear_sum_assignment(-sim)
    pairs=[]
    for i,j in zip(si,ri):
        a=ai.iloc[i]; r=real.iloc[j]
        pairs.append({'pair_id':base.identity('phase3',a.source_id,r.source_id),
                      'synthetic_source_id':a.source_id,'real_source_id':r.source_id,
                      'generator':a.generator,'split':'test','clip_cosine':float(sim[i,j])})
    pairs=pd.DataFrame(pairs)
    assert len(pairs)==204 and pairs.synthetic_source_id.nunique()==pairs.real_source_id.nunique()==204
    ids=set(pairs.synthetic_source_id)|set(pairs.real_source_id)
    chosen=test.loc[test.source_id.isin(ids)].copy()
    chosen['path']=chosen.source_id.map(lambda x: str(ROOT/'data/processed/base'/f'{x}.png'))
    chosen['output_file_sha256']=chosen.path.map(lambda x: base.digest(Path(x)))
    pairs.to_csv(OUT/'selected_pairs.csv',index=False); chosen.to_csv(OUT/'selected_sources.csv',index=False)
    base.write_json(OUT/'selection.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'seed':SEED,'ai_sources':204,'real_sources':204,'ai_per_collection':68,
        'selection':'ID-SHA order within each AI collection; maximum total CLIP cosine assignment to distinct COCO controls',
        'test_only':True,'reuses_previously_examined_sources':True,'new_independent_dataset':False,
        'payloads':list(base.PAYLOADS),'noise_seeds':list(base.NOISE_SEEDS),
        'classical_rule':'agreement >=13/16','trustmark_rule':'exact61-bit data message plus valid schema/ECC',
        'pairs_sha256':base.digest(OUT/'selected_pairs.csv'),'sources_sha256':base.digest(OUT/'selected_sources.csv')})

def prepare(marks):
    select()
    if (OUT/'base_covers.csv').exists(): raise RuntimeError('Covers already frozen')
    pairs=pd.read_csv(OUT/'selected_pairs.csv',keep_default_na=False)
    sources=pd.read_csv(OUT/'selected_sources.csv',keep_default_na=False).set_index('source_id')
    rows=[]
    for index,pair in enumerate(pairs.to_dict('records')):
        for origin,source_id in [(1,pair['synthetic_source_id']),(0,pair['real_source_id'])]:
            source=sources.loc[source_id]; path=Path(source.path)
            assert base.digest(path)==source.output_file_sha256
            pixels=base.read_image(path)
            methods=['unmarked','dwt_dct','trustmark'] if origin else ['unmarked']
            for method in methods:
                for payload in base.PAYLOADS if method!='unmarked' else ['']:
                    marks.set_payload(payload or base.PAYLOAD)
                    cover_id=base.identity('phase3_watermark',source_id,method,payload)
                    target=DATA/'covers'/method/f'{cover_id}.png'
                    result=marks.encode(pixels,method); base.save_pixels(target,result)
                    rows.append({'cover_id':cover_id,'image_id':cover_id,'source_id':source_id,'parent_id':source_id,
                        'pair_id':pair['pair_id'],'split':'test','generator':pair['generator'] if origin else 'real',
                        'paired_generator':pair['generator'],'label':origin,'input_origin_label':origin,
                        'watermark_method':method,'payload':payload,'group':'R-C' if not origin else ('S-U' if method=='unmarked' else 'S-W'),
                        'path':str(target),'original_path':str(path),'original_file_sha256':base.digest(path),
                        'output_file_sha256':base.digest(target),**base.quality(pixels,result)})
        if (index+1)%12==0: print('Expanded covers',index+1,'/204',flush=True)
    assert len(rows)==1632
    pd.DataFrame(rows).to_csv(OUT/'base_covers.csv',index=False)
    base.write_json(OUT/'provenance.json',{**marks.provenance(),'sources':408,'ai_sources':204,'covers':1632,
        'driver_sha256':base.digest(Path(__file__))})

def analyze(args,marks):
    covers=base.csv(OUT/'base_covers.csv').set_index('cover_id',verify_integrity=True)
    for column in ['label','input_origin_label']:
        covers[column]=covers[column].astype(int)
    conditions=base.import_conditions(args)
    assert set(conditions.cover_id).issubset(covers.index)
    previous=OUT/'measurements.csv'
    partial=OUT/'measurements.partial.csv'
    saved=previous if previous.exists() else partial
    rows=pd.read_csv(saved,keep_default_na=False,dtype={c:str for c in
        ['cover_id','source_id','payload','payload_bits','decoded_message']}).to_dict('records') if saved.exists() else []
    if rows:
        retained=pd.DataFrame(rows)
        if 'learned_decoder_device' not in retained:
            if args.prior_decode_device is None:
                raise RuntimeError('Resume of older rows requires their explicit prior decoder device')
            retained['learned_decoder_device']=args.prior_decode_device
        assert not retained.duplicated(['cover_id','condition','detector']).any()
        assert retained.groupby(['cover_id','condition']).detector.nunique().eq(2).all()
        expected_hash=conditions.set_index(['cover_id','condition']).output_file_sha256.to_dict()
        for r in rows:
            assert r['output_file_sha256']==expected_hash[(r['cover_id'],r['condition'])]
        for column in ['label','input_origin_label','verified']:
            retained[column]=retained[column].astype(int)
        rows=retained.to_dict('records')
    done={(r['cover_id'],r['condition']) for r in rows}
    work=[r for r in conditions.to_dict('records') if (r['cover_id'],r['condition']) not in done]
    anchor_file=OUT/'public_decode_anchors.json'
    anchors=json.loads(anchor_file.read_text()) if anchor_file.exists() else []
    for anchor in anchors:
        if 'learned_decoder_device' not in anchor:
            anchor['learned_decoder_device']=args.prior_decode_device or args.device
    selected=conditions.merge(covers.reset_index()[['cover_id','paired_generator','watermark_method','payload']],on='cover_id')
    representatives=selected.drop_duplicates(['paired_generator','watermark_method','payload'])
    anchor_keys=set(zip(representatives.cover_id,representatives.condition))
    representatives=selected.drop_duplicates('condition')
    anchor_keys.update(zip(representatives.cover_id,representatives.condition))
    checked={(r['cover_id'],r['condition']) for r in anchors}
    new_device_anchors=0
    for start in range(0,len(work),args.batch_size):
        block=work[start:start+args.batch_size]; pixels=[]; patches=[]
        for row in block:
            assert base.digest(Path(row['path']))==row['output_file_sha256']
            im=base.read_image(row['path']); pixels.append(im)
            patch=marks.tm.get_the_image_for_processing(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))).resize(
                (marks.tm.model_resolution_dec,marks.tm.model_resolution_dec),Image.Resampling.BILINEAR)
            patches.append(marks.transforms.ToTensor()(patch)*2-1)
        with torch.inference_mode():
            raw=(marks.tm.decoder.decoder(torch.stack(patches).to(marks.tm.decoder.device))>0).cpu().numpy().astype(np.uint8)
        decoded=marks.tm.ecc.decode_bitstream(raw,MODE='binary')
        for i,(transformed,im) in enumerate(zip(block,pixels)):
            cover={'cover_id':transformed['cover_id'],**covers.loc[transformed['cover_id']].to_dict()}
            payload=cover['payload'] or base.PAYLOAD
            assert base.digest(Path(cover['path']))==cover['output_file_sha256']
            assert base.digest(Path(cover['original_path']))==cover['original_file_sha256']
            marks.set_payload(payload)
            d=marks.dwt.detect(im,threshold=13/16)
            message,valid,schema=decoded[i]
            learned_valid=bool(valid and int(schema)==int(marks.tm.enctyp) and message==marks.expected)
            anchor_key=(cover['cover_id'],transformed['condition'])
            if (anchor_key in anchor_keys and anchor_key not in checked) or new_device_anchors<32:
                public=marks.tm.decode(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),MODE='binary',ROTATION=False,DETECTFIRST=False)
                assert bool(public[1])==bool(valid)
                if valid: assert public[0]==message and int(public[2])==int(schema)
                anchors.append({'cover_id':cover['cover_id'],'condition':transformed['condition'],
                    'learned_decoder_device':args.device,'batch_matches_public':True})
                checked.add(anchor_key)
                new_device_anchors+=1
            input_pixels=base.read_image(Path(cover['path']))
            origin=int(cover['input_origin_label']); clean=transformed['condition']=='identity'
            group=('R-C' if clean else 'R-T') if not origin else (
                'S-U' if cover['watermark_method']=='unmarked' else ('S-W' if clean else 'S-A'))
            shared={**transformed,**cover,'path':transformed['path'],
                'output_file_sha256':transformed['output_file_sha256'],'input_cover_path':cover['path'],
                'cover_group':cover['group'],'group':group,
                'learned_decoder_device':args.device,
                'image_id':base.identity('phase3_attack',cover['cover_id'],transformed['condition']),
                'psnr_vs_input_cover':float(cv2.PSNR(input_pixels,im)),
                'ssim_vs_input_cover':float(base.structural_similarity(input_pixels,im,channel_axis=2,data_range=255))}
            rows.append({**shared,'detector':'dwt_dct','verified':int(d.verified),'payload_bits':d.bits,
                'payload_bit_accuracy':d.bit_accuracy,'raw_codeword_accuracy':d.bit_accuracy,
                'ecc_valid':'','decoded_schema':'','full_message_match':int(d.bits==payload)})
            rows.append({**shared,'detector':'trustmark','verified':int(learned_valid),
                'payload_bits':''.join(map(str,raw[i,:16].tolist())),
                'payload_bit_accuracy':float(np.mean(raw[i,:16]==np.array(list(payload),dtype=np.uint8))),
                'raw_codeword_accuracy':float(np.mean(raw[i]==marks.codeword)),
                'ecc_valid':int(valid),'decoded_schema':int(schema),'full_message_match':int(learned_valid),'decoded_message':message if valid else ''})
        if (start//args.batch_size)%32==0:
            pd.DataFrame(rows).to_csv(OUT/'measurements.partial.csv',index=False)
            base.write_json(anchor_file,anchors)
            print('Expanded decodes',start+len(block),'/',len(work),flush=True)
    frame=pd.DataFrame(rows)
    assert len(frame)==2*len(conditions)
    assert not frame.duplicated(['cover_id','condition','detector']).any()
    assert frame.groupby(['cover_id','condition']).detector.nunique().eq(2).all()
    for column in ['label','input_origin_label','verified']:
        frame[column]=frame[column].astype(int)
    frame.to_csv(previous,index=False)
    base.write_json(OUT/'summary.json',base.summarize(frame))
    exact=[]
    for (condition,detector,origin),block in frame.groupby(['condition','detector','input_origin_label']):
        relevant=block.loc[(block.watermark_method==detector) if origin else block.watermark_method.eq('unmarked')]
        if relevant.empty: continue
        success=relevant.groupby('source_id').verified.min()
        exact.append({'condition':condition,'detector':detector,'input_origin_label':int(origin),
            'n_sources':len(success),'all_payload_successes':int(success.sum()),'ci95':base.rate_interval(int(success.sum()),len(success))})
    base.write_json(OUT/'source_verification_intervals.json',{'target':'All evaluated payloads verify for one source',
        'source_unit':True,'common_success_probability_descriptive_assumption':True,'measurements':exact})
    base.write_json(OUT/'analysis_provenance.json',{**marks.provenance(),'outputs':len(conditions),'rows':len(frame),
        'driver_sha256':base.digest(Path(__file__)),'batched_official_ECC':True,'public_decode_anchors':anchors,
        'learned_decoder_device_rows':{str(k):int(v) for k,v in frame.loc[frame.detector=='trustmark'].learned_decoder_device.value_counts().items()},
        'external_manifests':{str(p):base.digest(p) for p in args.diffusion_manifest}})
    base.write_json(anchor_file,anchors)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['select','prepare','attacks','analyze'],required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu'); ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--diffusion-manifest',type=Path,action='append',default=[])
    ap.add_argument('--prior-decode-device',choices=['cpu','cuda'],help='Recorded device for older partial rows lacking that column')
    args=ap.parse_args(); args.output=OUT; args.data=DATA
    if args.stage=='select': select(); return
    if args.stage=='attacks': base.attacks(args); return
    marks=base.Watermarks(args.device,args.threads)
    prepare(marks) if args.stage=='prepare' else analyze(args,marks)

if __name__=='__main__': main()
